"""Inference: python -m irrigation_ml.recommend [--at 2025-09-01T06:00] [--c-fn 5] [--system sprinkler] [--json]

Replays the latest (or a chosen) issue time from the processed hourly table. In production the table is rebuilt
from live probe telemetry plus a weather-forecast feed (rain_fc_next*h); everything downstream is unchanged.
"""
import argparse
import json

import joblib
import numpy as np
import pandas as pd

from .config import ARTIFACTS, HORIZONS
from .decision import Policy, decide, decide_offline, fmt_duration
from . import offline as offline_mod
from .forecast import prepare
from .risk import design

ACTIONS = {"irrigate": "IRRIGATE NOW", "irrigate_soon": "PLAN TO IRRIGATE SOON", "postpone": "POSTPONE (rain expected)", "no_action": "NO ACTION",
           "check_sensor": "CHECK SENSOR"}


def load_artifacts():
    return (joblib.load(ARTIFACTS / "forecast_models.joblib"), joblib.load(ARTIFACTS / "risk_models.joblib"),
            json.loads((ARTIFACTS / "calibration.json").read_text()))


def load_offline():
    f = ARTIFACTS / "offline_models.joblib"
    return joblib.load(f) if f.exists() else None


def diagnose_sensor(df, fid, t, battery_min_v=3.8):
    """Why is the probe not usable at t? -> dict(text, age_h): age_h = hours since the last trusted reading (None if never)."""
    g = df[(df["field_id"] == fid) & (df["timestamp"] <= t)]
    w = g.tail(24)
    ok = g["sm_obs_causal"].notna() & (g["battery_voltage"].isna() | (g["battery_voltage"] >= battery_min_v))
    age_h = float((t - g.loc[ok, "timestamp"].iloc[-1]).total_seconds() / 3600) if ok.any() else None
    batt = w["battery_voltage"].min()
    share = lambda c: float(w[c].fillna(False).astype(bool).mean()) if len(w) else 0.0
    no_record = float(w["battery_voltage"].isna().mean()) if len(w) else 1.0
    if pd.notna(batt) and batt < battery_min_v:
        why = f"The sensor battery is low ({batt:.2f} V), so its readings cannot be trusted"
    elif no_record >= 0.5:
        why = "The sensor has not sent data for most of the last 24 hours"
    elif share("sm_dead") >= 0.3:
        why = "The sensor is reporting near-zero values, which means a probe fault"
    elif share("sm_flat_stuck") >= 0.3:
        why = "The sensor reading is stuck at one value"
    else:
        why = "Most readings in the last 24 hours were rejected as unreliable"
    n_ok = int(ok.tail(24).sum())
    if age_h is not None and age_h < 1 and not (pd.notna(batt) and batt < battery_min_v):
        # a trusted reading exists right now, but the 24 h before it were flagged: the sensor has only just recovered
        return dict(text=f"Only {n_ok} of the last 24 hours {'has' if n_ok == 1 else 'have'} a trusted reading: the sensor was flagged as unreliable until a moment ago.", age_h=age_h)
    when = f"The last trusted reading was {fmt_duration(age_h)} ago." if age_h is not None else "There is no earlier trusted reading."
    return dict(text=f"{why}. {when}", age_h=age_h)


def recommend(at=None, policy=Policy(), system="sprinkler", prepared=None):
    fm, rm, cal = load_artifacts()
    off_models = load_offline()
    df, feat = prepared or prepare()
    t = pd.Timestamp(at) if at else feat["timestamp"].max()
    out = []
    for fid, g in feat.groupby("field_id"):
        g = g[g["timestamp"] <= t]
        if g.empty:
            continue
        row = g.iloc[[-1]]
        th = cal["thresholds"][fid]
        sm0 = float(row["sm0"].iloc[0]) if pd.notna(row["sm0"].iloc[0]) else None
        stale_h = (t - row["timestamp"].iloc[0]).total_seconds() / 3600
        fc = {}
        if sm0 is not None:
            X = row[fm["cols"]]
            for H in HORIZONS:
                off = cal["interval_offsets"][str(H)]
                m = fm["horizons"][H]
                q = {k: float(np.clip(sm0 + m[k].predict(X)[0], 0, 100)) for k in ("mean", "q10", "q90")}
                fc[H] = (max(q["q10"] - off, 0.0), q["mean"], min(q["q90"] + off, 100.0))
        p_dry = {}
        if sm0 is not None:
            Xr = design(row, pd.DataFrame({"trigger": [th["trigger"]]}, index=row.index), rm["cols"])
            p_dry = {H: float(rm["models"][H].predict_proba(Xr)[0, 1]) for H in HORIZONS}
        rain = {H: float(row[f"rain_fc_next{H}h"].iloc[0]) for H in HORIZONS}
        trust = dict(battery_min=row["batt_min_24"].iloc[0], obs_share=row["obs_share_24"].iloc[0],
                     anomaly=bool(row["anom_iforest_24"].iloc[0] > 0.25))
        taw = df.loc[(df.field_id == fid) & (df.timestamp == row["timestamp"].iloc[0]), "taw_mm"].iloc[0]
        d = decide(sm0, p_dry, th, rain, trust, policy, {H: fc[H][1] for H in fc} if fc else None,
                   None if pd.isna(taw) else float(taw), system) if sm0 is not None else \
            decide(None, {}, th, rain, trust, policy)
        issue = None
        if d["action"] == "check_sensor":
            fc = {}                       # never present a forecast built on an unusable reading
            dg = diagnose_sensor(df, fid, row["timestamp"].iloc[0], policy.battery_min_v)
            issue = dg["text"]
            if off_models is not None and dg["age_h"] is not None and dg["age_h"] <= policy.offline_max_age_h:
                x = offline_mod.infer_features(df, feat, fid, row["timestamp"].iloc[0], dg["age_h"], cal["thresholds"])
                if x is not None:
                    p_off = {H: float(off_models["models"][H].predict_proba(x[off_models["features"]])[0, 1]) for H in HORIZONS}
                    d = decide_offline(p_off, dg["age_h"], rain, th, policy)
            if d["mode"] == "none":
                d["reasons"] = ["no estimate is possible: " + ("the probe has been offline too long" if dg["age_h"] is not None and dg["age_h"] > policy.offline_max_age_h else "no earlier trusted reading")]
        elif trust["anomaly"]:
            issue = "Sensor readings look unusual (the anomaly detector flagged part of the last 24 hours)."
        d["sensor_issue"] = issue
        d["irrigate_by"] = (str((row["timestamp"].iloc[0] + pd.Timedelta(hours=d["irrigate_in_h"])).date())
                            if d["irrigate_in_h"] is not None else None)
        if stale_h > 6:
            d["reasons"].append(f"latest probe data is {stale_h:.0f} h old")
            d["confidence"] = "low"
        out.append(dict(field_id=fid, issued_at=str(row["timestamp"].iloc[0]), crop=row["crop"].iloc[0],
                        soil_moisture=sm0, trigger=th["trigger"], forecast=fc, rain_forecast_mm=rain, **d))
    return out


def render(recs) -> str:
    lines = []
    for r in recs:
        lines += [f"=== {r['field_id']} ({r['crop']}) @ {r['issued_at']}  [mode: {r['mode']}]",
                  f"  RECOMMENDATION: {ACTIONS[r['action']]}" + (f"  [irrigate by {r['irrigate_by']}; dry soil expected within {r['dry_expected_h'] // 24} days]" if r["action"] == "irrigate_soon" else "")
                  + f"   (confidence: {r['confidence']})"]
        for x in r["reasons"]:
            lines.append(f"  - {x}")
        if r.get("sensor_issue"):
            lines.append(f"  - sensor: {r['sensor_issue']}")
        if r["amount_mm"]:
            more = f" (soil deficit ~{r['deficit_mm']:.0f} mm: repeat on the following days)" if r.get("capped") else ""
            lines.append(f"  - suggested amount: ~{r['amount_mm']:.0f} mm gross{more}")
        if r["mode"] == "weather_only":
            for H in HORIZONS:
                lines.append(f"    +{H:>2}h: P(dry)={r['p_dry'][H]:.2f}  rain fc {r['rain_forecast_mm'][H]:.1f} mm")
        if r["soil_moisture"] is not None:
            lines.append(f"  moisture now {r['soil_moisture']:.0f} (trigger {r['trigger']:.0f})")
            for H, (lo, mid, hi) in r["forecast"].items():
                lines.append(f"    +{H:>2}h: {mid:5.1f}  [{lo:5.1f}, {hi:5.1f}]  P(dry)={r['p_dry'][H]:.2f}  rain fc {r['rain_forecast_mm'][H]:.1f} mm")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--at"); ap.add_argument("--c-fn", type=float, default=5.0)
    ap.add_argument("--system", default="sprinkler", choices=["sprinkler", "drip"]); ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    recs = recommend(a.at, Policy(c_fn=a.c_fn), a.system)
    print(json.dumps(recs, indent=1, default=str) if a.json else render(recs))
