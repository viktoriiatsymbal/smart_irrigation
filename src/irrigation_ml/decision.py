"""Decision layer: forecast distribution -> {irrigate, postpone, no_action} with reasons.

Dry trigger is calibrated per field on the probe's own scale (FAO-56 RAW logic with sensor-derived anchors):
    theta_trigger = fc_s - p_mad * (fc_s - pwp_s)
where fc_s / pwp_s are in-season upper / lower plateaus of the field's own readings. Action thresholds follow
asymmetric cost: irrigate when P(dry within horizon) >= c_fp / (c_fp + c_fn).
"""
from dataclasses import dataclass, field
import json

import numpy as np
import pandas as pd

from .config import ARTIFACTS, HORIZONS

P_MAD = {"corn": 0.55, "sorghum": 0.55}      # from ref_crop.csv
EFFICIENCY = {"sprinkler": 0.75, "drip": 0.90}
MIN_PROBE_RANGE = 15.0                     # fc_s - pwp_s below this: probe too flat to define a trigger
INFIL_MM_H = {"sprinkler": 13.0, "drip": 13.0}    # ref_fields_soil.csv (loam)


@dataclass
class Policy:
    c_fn: float = 3.0                 # cost of missing a needed irrigation (crop stress)
    c_fp: float = 1.0                 # cost of an unnecessary irrigation (water, energy)
    rain_effective_mm: float = 5.0    # forecast rain that justifies postponing
    battery_min_v: float = 3.8        # below this the probe degrades (DE finding)
    min_obs_share: float = 0.5        # share of trusted readings in last 24 h
    max_event_mm: float = 40.0        # largest single application (gross); a bigger deficit is split over several days
    offline_max_age_h: float = 240.0  # no-probe estimate is only offered for outages up to 10 days (no skill beyond that)

    @property
    def tau(self):
        return self.c_fp / (self.c_fp + self.c_fn)


def sensor_thresholds(df: pd.DataFrame, until=None, q_fc=0.75, q_pwp=0.02) -> dict:
    """Per-field {fc_s, pwp_s, trigger} from in-season valid readings (optionally only before `until`)."""
    d = df[df["kc"].notna() & df["sm_clean"].notna()]
    if until is not None:
        d = d[d["timestamp"] < pd.Timestamp(until)]
    out = {}
    for fid, g in d.groupby("field_id"):
        p = P_MAD.get(g["crop"].iloc[-1], 0.55)
        fc_s, pwp_s = g["sm_clean"].quantile(q_fc), g["sm_clean"].quantile(q_pwp)
        out[fid] = dict(fc_s=float(fc_s), pwp_s=float(pwp_s), trigger=float(fc_s - p * (fc_s - pwp_s)), p_mad=p,
                        reliable=bool(fc_s - pwp_s >= MIN_PROBE_RANGE))
    return out


def calibrate_intervals(oof: pd.DataFrame, level=0.8) -> dict:
    """Conformalised quantile regression: widen [q10, q90] so that coverage reaches `level` out of fold."""
    off = {}
    for H, g in oof.dropna(subset=["q10", "q90", "y"]).groupby("H"):
        s = np.maximum(g["q10"] - g["y"], g["y"] - g["q90"]).to_numpy()
        off[int(H)] = float(np.quantile(s, min(1.0, level * (1 + 1 / len(s)))))
    return off


def prob_below(theta, q10, q50, q90):
    """P(sm < theta) by interpolating the predictive CDF through (q10,.1), (q50,.5), (q90,.9); tails extrapolated."""
    lo, mid, hi = min(q10, q50), q50, max(q90, q50)
    xs = [lo - 2.0 * max(mid - lo, 1e-3), lo, mid, hi, hi + 2.0 * max(hi - mid, 1e-3)]
    return float(np.interp(theta, xs, [0.01, 0.10, 0.50, 0.90, 0.99]))


def gross_need_mm(sm, th, taw_mm, system="sprinkler"):
    """Water to bring the root zone back to field capacity, divided by system efficiency."""
    frac = np.clip((th["fc_s"] - sm) / (th["fc_s"] - th["pwp_s"]), 0, 1)
    return float(frac * taw_mm / EFFICIENCY[system])


def decide(sm0, p_dry: dict, th: dict, rain_fc: dict, trust: dict, policy: Policy = Policy(),
           median_fc: dict = None, taw_mm=None, system="sprinkler") -> dict:
    """p_dry[H] = classifier P(probe < trigger within H h); median_fc[H] = forecast median moisture at t+H;
    rain_fc[H] = forecast rain (mm) over next H h; trust = {battery_min, obs_share, anomaly}.

    1. no usable reading (missing, <50% coverage, battery < 3.8 V) -> check_sensor (recommend() may then try the no-probe
       fallback); an anomaly-detector flag alone only lowers confidence
    2. no dry risk through 72 h             -> no_action
    3. rain forecast AND the moisture forecast recovers above the trigger at that horizon -> postpone
       (rain alone is never enough: ERA5 rain often misses the field)
    4. dry now or P(dry in 24 h) >= tau     -> irrigate (with amount)
    5. risk first reaches tau at 48 or 72 h -> irrigate_soon: the forward-looking recommendation. dry_expected_h is the model's
       window; irrigate_in_h = dry_expected_h - 24 is the deadline we advise, one step earlier, because flagged dry-downs
       arrive earlier than the model says. NOTE: measured later, this one-step margin is still too small (the probe crosses
       within 24 h in 94% of 48 h flags and 57% of 72 h flags); see docs/PIPELINE_REPORT.md sections 5.4 and 10.1. The amount is estimated
       from the forecast moisture at the dry window.
    """
    if sm0 is None or np.isnan(sm0) or trust.get("obs_share", 1) < policy.min_obs_share \
            or (trust.get("battery_min") is not None and not np.isnan(trust["battery_min"]) and trust["battery_min"] < policy.battery_min_v):
        return dict(action="check_sensor", irrigate_in_h=None, dry_expected_h=None, confidence="low", p_dry={}, amount_mm=None, deficit_mm=None, capped=False,
                    mode="none", reasons=["probe reading unavailable or untrusted"])
    theta, tau = th["trigger"], policy.tau
    p, run = dict(p_dry), 0.0
    for H in HORIZONS:                    # "dry within 24 h" is contained in "dry within 72 h": risk cannot fall with horizon,
        run = max(run, p[H]); p[H] = run  # but the three classifiers are fitted separately, so enforce it here
    p[0] = 1.0 if sm0 < theta else 0.0
    risk = max(p.values())
    first = next(H for H in (0,) + HORIZONS if p[H] >= tau) if risk >= tau else None
    rain_h = next((H for H in HORIZONS if rain_fc[H] >= policy.rain_effective_mm), None)
    recovers = rain_h is not None and median_fc is not None and median_fc[rain_h] >= theta
    notes = [] if th.get("reliable", True) else ["probe range is narrow for this field: trigger is approximate"]
    if trust.get("anomaly"):
        notes.append("sensor readings look unusual (anomaly detector flagged part of the last 24 h); treat this advice with caution")
    amount, deficit, in_h, dry_h = None, None, None, None
    if risk < tau:
        action, why = "no_action", f"max P(dry) = {risk:.2f} < {tau:.2f} through {max(HORIZONS)} h"
    elif recovers and (first or 0) >= rain_h:
        action, why = "postpone", f"{rain_fc[rain_h]:.0f} mm rain forecast within {rain_h} h and forecast moisture {median_fc[rain_h]:.0f} stays above trigger {theta:.0f}"
    elif first in (0, 24):
        action, in_h = "irrigate", 0
        why = f"moisture {sm0:.0f} vs trigger {theta:.0f}; P(dry within 24 h) = {max(p[0], p[24]):.2f} >= {tau:.2f}"
        if taw_mm:
            deficit = gross_need_mm(sm0, th, taw_mm, system)
            amount = min(deficit, policy.max_event_mm)
    else:
        action, in_h, dry_h = "irrigate_soon", first - 24, first
        why = (f"forecast: soil moisture is expected to reach the irrigation level within {first // 24} days "
               f"(P(dry within {first} h) = {p[first]:.2f} >= {tau:.2f}); irrigate within {in_h // 24} day{'s' if in_h > 24 else ''} to stay ahead of it")
        if taw_mm:                       # deficit expected at that time, from the forecast median
            at_h = median_fc[first] if median_fc else sm0
            deficit = gross_need_mm(min(at_h, sm0), th, taw_mm, system)
            amount = min(deficit, policy.max_event_mm)
    conf = "low" if notes else ("high" if (risk < 0.5 * tau or risk > 2 * tau) else "medium")
    return dict(action=action, irrigate_in_h=in_h, dry_expected_h=dry_h, confidence=conf, reasons=[why] + notes, p_dry=p, amount_mm=amount, deficit_mm=deficit,
                capped=bool(amount is not None and deficit > amount), mode="sensor")


def fmt_duration(hours: float) -> str:
    """Human wording for a span: 'less than an hour', '5 hours', '8.3 days' (days only from 2 days up)."""
    if hours < 1:
        return "less than an hour"
    if hours < 48:
        h = int(round(hours))
        return f"{h} hour" + ("" if h == 1 else "s")
    return f"{hours / 24:.1f} days"


def decide_offline(p_dry: dict, age_h, rain_fc: dict, th: dict, policy: Policy = Policy()) -> dict:
    """Advice while the probe is down, from the no-probe model. Offered only for outages up to policy.offline_max_age_h;
    always low confidence, never gives an amount (the soil deficit is unknown) and never postpones (no state to confirm recovery)."""
    if age_h is None or np.isnan(age_h) or age_h > policy.offline_max_age_h:
        return dict(action="check_sensor", irrigate_in_h=None, dry_expected_h=None, confidence="low", p_dry={}, amount_mm=None, deficit_mm=None, capped=False,
                    mode="none", reasons=["probe has been offline too long for a reliable estimate"])
    tau, p, run = policy.tau, dict(p_dry), 0.0
    for H in HORIZONS:
        run = max(run, p[H]); p[H] = run
    risk = max(p.values())
    if p[24] >= tau:
        action, in_h, why = "irrigate", 0, f"no trusted probe reading for {fmt_duration(age_h)}; weather-based risk of dry soil within 24 h is {p[24]:.2f} (threshold {tau:.2f})"
    elif risk >= tau:
        first = next(H for H in HORIZONS if p[H] >= tau)
        action, in_h, why = "irrigate_soon", first - 24, f"no trusted probe reading for {fmt_duration(age_h)}; weather-based risk of dry soil reaches {risk:.2f} within {first} h"
    else:
        action, in_h, why = "no_action", None, f"no trusted probe reading for {fmt_duration(age_h)}; weather-based risk of dry soil stays below {tau:.2f} for 72 h"
    notes = ["estimated from the last trusted reading and the weather since, not from the sensor"]
    if rain_fc.get(24, 0) >= policy.rain_effective_mm or rain_fc.get(48, 0) >= policy.rain_effective_mm:
        notes.append("rain is forecast, but without the probe it cannot be confirmed that it will reach the soil")
    return dict(action=action, irrigate_in_h=in_h, dry_expected_h=(in_h + 24 if action == "irrigate_soon" else None), confidence="low", reasons=[why] + notes, p_dry=p, amount_mm=None, deficit_mm=None,
                capped=False, mode="weather_only")
