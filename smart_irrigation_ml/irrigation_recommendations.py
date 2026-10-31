"""Offline FAO-56-style irrigation DECISION PROTOTYPE for the supplied dataset.

Outputs conditional recommendations, NEVER a validated irrigate/no-irrigate label.
Do not use for automated valve control. See README for assumptions.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Assumptions transcribed from cells 18 and 20 of analysis_advanced_real_info.ipynb.
# No claim that these are measured field soil parameters or actual farm dates.
CROPS = {
    "corn": {"kc_ini": .30, "kc_mid": 1.20, "kc_end": .35, "root_m": 1.2, "p": .55,
             "stages": ("04-20", "05-20", "07-01", "08-20", "09-30")},
    "sorghum": {"kc_ini": .30, "kc_mid": 1.05, "kc_end": .55, "root_m": 1.3, "p": .55,
                "stages": ("05-10", "06-05", "07-10", "08-25", "10-05")},
}
SOIL_PARAMS = {"loam": {"fc": .25, "pwp": .12, "fc_range": (.20, .30), "pwp_range": (.07, .17)}}
DECISION_HOUR = 6
MAX_SMAP_AGE_HOURS = 6


def usda_texture(sand_percent: float, clay_percent: float) -> str:
    silt = 100 - sand_percent - clay_percent
    if 7 <= clay_percent < 27 and 28 <= silt < 50 and sand_percent <= 52:
        return "loam"
    return "unknown"  # Never silently apply loam values to an unrecognized soil texture.


def crop_state(ts: pd.Timestamp, crop: str) -> tuple[bool, float, float]:
    params = CROPS.get(str(crop).lower())
    if params is None:
        return False, np.nan, np.nan
    days = [pd.Timestamp(f"{ts.year}-{md}") for md in params["stages"]]
    if not days[0] <= ts.normalize() <= days[-1]:
        return False, np.nan, np.nan
    current = ts.normalize()
    if current < days[1]:
        kc = params["kc_ini"]
    elif current < days[2]:
        w = (current - days[1]).days / max(1, (days[2] - days[1]).days)
        kc = params["kc_ini"] + w * (params["kc_mid"] - params["kc_ini"])
    elif current < days[3]:
        kc = params["kc_mid"]
    elif current < days[4]:
        w = (current - days[3]).days / max(1, (days[4] - days[3]).days)
        kc = params["kc_mid"] + w * (params["kc_end"] - params["kc_mid"])
    else:
        kc = params["kc_end"]
    frac = np.clip((current - days[0]).days / max(1, (days[2] - days[0]).days), 0, 1)
    root = .30 + (params["root_m"] - .30) * frac
    return True, float(kc), float(root)


def threshold_range(p: float, soil: dict) -> tuple[float, float]:
    thresholds = []
    for fc in soil["fc_range"]:
        for pwp in soil["pwp_range"]:
            if fc > pwp:
                thresholds.append(fc - p * (fc - pwp))
    return min(thresholds), max(thresholds)


def load_forecasts(path: Path | None) -> pd.DataFrame | None:
    """Future forecast files must contain forecasts genuinely issued at or BEFORE decision time.

    Schema: field_id, issue_time, decision_timestamp, rain_next24h_mm, et0_next24h_mm.
    No forecast -> no 'postpone' recommendation. Historical observed future weather is forbidden.
    """
    if path is None:
        return None
    fc = pd.read_csv(path)
    required = {"field_id", "issue_time", "decision_timestamp", "rain_next24h_mm", "et0_next24h_mm"}
    if not required <= set(fc):
        raise ValueError(f"Forecast file missing {sorted(required - set(fc))}")
    fc["issue_time"] = pd.to_datetime(fc["issue_time"], errors="coerce")
    fc["decision_timestamp"] = pd.to_datetime(fc["decision_timestamp"], errors="coerce")
    fc = fc.dropna(subset=["issue_time", "decision_timestamp"])
    fc = fc[(fc.issue_time <= fc.decision_timestamp) &
            (fc.issue_time >= fc.decision_timestamp - pd.Timedelta(hours=24))]
    for c in ["rain_next24h_mm", "et0_next24h_mm"]:
        fc[c] = pd.to_numeric(fc[c], errors="coerce")
        fc = fc[fc[c].ge(0)]
    fc = fc.sort_values("issue_time").drop_duplicates(["field_id", "decision_timestamp"], keep="last")
    return fc


def make_decisions(raw: pd.DataFrame, forecasts: pd.DataFrame | None = None) -> pd.DataFrame:
    required = {"timestamp", "field_id", "crop", "smap_sm_rootzone", "smap_timestamp", "smap_static_mwrtm_sand", "smap_static_mwrtm_clay"}
    if not required <= set(raw):
        raise ValueError(f"Source missing columns {sorted(required - set(raw))}")
    d = raw.copy()
    d["timestamp"] = pd.to_datetime(d.timestamp, errors="coerce")
    d["smap_timestamp"] = pd.to_datetime(d.smap_timestamp, errors="coerce")
    d = d.dropna(subset=["timestamp", "field_id"]).sort_values(["field_id", "timestamp"])
    d = d.drop_duplicates(["field_id", "timestamp"], keep="first")
    # Only lagged ET0 measurements, excluding the decision-hour value. Do not extrapolate gaps.
    et_col = "openmeteo_et0_fao_evapotranspiration"
    if et_col in d.columns:
        et = pd.to_numeric(d[et_col], errors="coerce")
        # Reindex to preserve hourly time alignment within each field before rolling.
        d["et0_h"] = et
    else:
        d["et0_h"] = np.nan
    groups = []
    for field, sub in d.groupby("field_id", sort=False):
        ix = pd.date_range(sub.timestamp.min(), sub.timestamp.max(), freq="h")
        sub = sub.set_index("timestamp").reindex(ix).rename_axis("timestamp").reset_index()
        sub["field_id"] = field
        sub["et0_last24h"] = sub["et0_h"].shift(1).rolling(24, min_periods=18).sum()
        groups.append(sub.loc[sub.timestamp.dt.hour.eq(DECISION_HOUR)].copy())
    snap = pd.concat(groups, ignore_index=True).sort_values(["timestamp", "field_id"]).reset_index(drop=True)
    if forecasts is not None:
        fc = forecasts.rename(columns={"decision_timestamp": "timestamp"})
        snap = snap.merge(fc, on=["field_id", "timestamp"], how="left", validate="one_to_one")
    snap["day"] = snap.timestamp.dt.normalize()
    rows = []
    for r in snap.itertuples(index=False):
        t = r.timestamp
        field, crop = r.field_id, str(getattr(r, "crop", "")).lower()
        theta = pd.to_numeric(getattr(r, "smap_sm_rootzone", np.nan), errors="coerce")
        smap_time = getattr(r, "smap_timestamp", pd.NaT)
        age_h = (t - smap_time).total_seconds() / 3600 if pd.notna(smap_time) else np.nan
        smap_good = bool(np.isfinite(theta) and 0 <= theta <= .7 and np.isfinite(age_h) and 0 <= age_h <= MAX_SMAP_AGE_HOURS)
        sand = getattr(r, "smap_static_mwrtm_sand", np.nan)
        clay = getattr(r, "smap_static_mwrtm_clay", np.nan)
        soil_class = usda_texture(float(sand) * 100, float(clay) * 100) if pd.notna(sand) and pd.notna(clay) else "unknown"
        soil = SOIL_PARAMS.get(soil_class)
        in_season, kc, root = crop_state(t, crop)
        et0_recent = getattr(r, "et0_last24h", np.nan)
        etc_recent = float(et0_recent * kc) if np.isfinite(et0_recent) and np.isfinite(kc) else np.nan
        p_base = CROPS[crop]["p"] if crop in CROPS else np.nan
        p = float(np.clip(p_base + .04 * (5 - etc_recent), .1, .8)) if np.isfinite(etc_recent) else p_base
        valid = in_season and smap_good and (soil is not None) and np.isfinite(p)
        taw = raw_mm = dep = fraction = thr = lo = hi = np.nan
        action = "insufficient_data"
        midpoint_action = "unknown"
        robust = "not_assessed"
        reason = "no soil/SMAP/crop state compatible with a decision"
        rain_fc = getattr(r, "rain_next24h_mm", np.nan)
        et0_fc = getattr(r, "et0_next24h_mm", np.nan)
        used_forecast = False
        if crop not in CROPS:
            action, reason = "insufficient_data", "unknown crop"
        elif not in_season:
            action, reason = "no_action_out_of_season", "outside ASSUMED crop calendar"
        elif not smap_good:
            action, reason = "insufficient_data", "SMAP moisture missing/stale/future or invalid"
        elif soil is None:
            action, reason = "insufficient_data", "unrecognized coarse-grid soil class"
        elif valid:
            taw = 1000 * (soil["fc"] - soil["pwp"]) * root
            raw_mm = p * taw
            dep = float(np.clip(1000 * (soil["fc"] - float(theta)) * root, 0, taw))
            fraction = dep / raw_mm if raw_mm else np.nan
            thr = soil["fc"] - p * (soil["fc"] - soil["pwp"])
            lo, hi = threshold_range(p, soil)
            robust = ("dry_all_assumptions" if theta <= lo else
                      "wet_all_assumptions" if theta >= hi else "assumption_sensitive")
            if dep >= raw_mm:
                midpoint_action = "irrigate_candidate"
                action = "review_irrigation"  # never an automatic command
                reason = "estimated root-zone depletion >= RAW; check local soil and irrigation history"
                if np.isfinite(rain_fc) and np.isfinite(et0_fc) and rain_fc >= 0 and et0_fc >= 0:
                    # Forecast is only used when proven issued <= decision time by load_forecasts.
                    projected = np.clip(dep + kc * et0_fc - .8 * rain_fc, 0, taw)
                    used_forecast = True
                    if projected < raw_mm:
                        action = "postpone_review"
                        reason = "issued future rain forecast MAY remove water stress; verify forecast and soil"
            elif fraction >= .90:
                midpoint_action, action = "monitor", "monitor"
                reason = "approaching provisional RAW trigger"
            else:
                midpoint_action, action = "no_action", "no_action"
                reason = "below provisional depletion trigger"
            if action == "no_action" and robust == "assumption_sensitive":
                action = "monitor"
                reason = "nominally wet, but soil uncertainty changes the decision"
        sensor = getattr(r, "soil_moisture_sensor", np.nan)
        battery = getattr(r, "battery_voltage", np.nan)
        low_battery = bool(pd.notna(battery) and float(battery) <= 3.8)
        sensor_valid = bool(pd.notna(sensor) and 10 <= float(sensor) <= 100 and not low_battery)
        # Never claim high confidence: model soil coefficients are not field calibrated.
        confidence = ("very_low" if not valid or robust == "assumption_sensitive" or not sensor_valid else "low")
        rows.append({
            "field_id": field, "decision_timestamp": t, "crop": crop, "soil_class_coarse": soil_class,
            "in_assumed_growing_season": bool(in_season), "smap_theta_rootzone_m3m3": theta,
            "smap_age_hours": age_h, "sensor_moisture_raw": sensor,
            "battery_voltage": battery, "sensor_valid_indicator": sensor_valid,
            "kc_assumed": kc, "root_depth_m_assumed": root, "et0_past24h_mm": et0_recent,
            "etc_past24h_mm": etc_recent, "depletion_fraction_p": p,
            "fc_assumed_m3m3": soil["fc"] if soil else np.nan,
            "pwp_assumed_m3m3": soil["pwp"] if soil else np.nan,
            "taw_mm": taw, "raw_mm": raw_mm, "depletion_mm": dep,
            "depletion_over_raw": fraction, "theta_trigger_nominal": thr,
            "theta_trigger_min_sensitivity": lo, "theta_trigger_max_sensitivity": hi,
            "assumption_robustness": robust, "nominal_action": midpoint_action,
            "recommendation": action, "confidence": confidence, "reason": reason,
            "forecast_used": used_forecast, "rain_forecast_next24h_mm": rain_fc,
            "et0_forecast_next24h_mm": et0_fc,
            "validated_irrigation_label": False,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--forecast", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs/decision_prototype_daily.csv"))
    args = parser.parse_args()
    raw = pd.read_parquet(args.data) if args.data.suffix == ".parquet" else pd.read_csv(args.data, low_memory=False)
    fc = load_forecasts(args.forecast)
    output = make_decisions(raw, fc)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    recent = output.sort_values("decision_timestamp").groupby("field_id").tail(1)
    recent.to_csv(args.output.with_name("decision_prototype_last_historical.csv"), index=False)
    summary = {
        "source": str(args.data), "status": "exploratory_physics_based_rule_ENGINE_NOT_VERIFIED_ML_POLICY",
        "total_snapshots": len(output), "dates": [str(output.decision_timestamp.min()), str(output.decision_timestamp.max())],
        "fields": sorted(output.field_id.unique().tolist()),
        "actions": {str(k): int(v) for k, v in output.recommendation.value_counts().items()},
        "growing_season_actions": {str(k): int(v) for k, v in output.loc[output.in_assumed_growing_season, "recommendation"].value_counts().items()},
        "assumption_robustness": {str(k): int(v) for k, v in output.assumption_robustness.value_counts().items()},
        "has_historical_irrigation_truth": False,
        "live_advice": False, "note": "Historical data ends in 2025. No live 2026 readings. Never actuate irrigation.",
        "limits": ["Sensor scale 0-100 not calibrated to volumetric moisture", "SMAP coarse grid, not field probe",
                   "Loam FC/PWP and crop dates/root depth assumed in teammate notebook",
                   "No true irrigation events or crop-stress outcomes", "Forecast unavailable unless supplied as an issued forecast"],
    }
    args.output.with_name("decision_prototype_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print('\nLast known HISTORICAL snapshots (NOT current 2026 recommendations):')
    print(recent[["field_id", "decision_timestamp", "recommendation", "reason", "confidence"]].to_string(index=False))


if __name__ == "__main__":
    main()
