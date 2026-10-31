import numpy as np
import pandas as pd

from .config import SATURATION

TRUST = ["battery_voltage", "solar_panel_voltage", "anom_iforest", "anom_sm_any", "sm_dead", "sm_post_gap",
         "trust_bad_past_48h", "rng_soil_moisture_sensor", "sm_flat_stuck", "sm_spike", "sm_jump_up_no_rain",
         "sm_drop_fast", "st_flat", "pr_dry_gauge"]
WEATHER_OBS = ["air_temperature_sensor", "air_humidity_sensor", "air_pressure_sensor", "era5_ssrd_wm2",
               "era5_wind_speed_10m", "et0_mm_h", "etc_mm_h"]
FORECAST = ["rain_fc_next24h", "rain_fc_next48h", "rain_fc_next72h"]
# Reanalysis / satellite: high latency (ERA5-Land ~5 d, SMAP L4 ~2.5 d) -> not available at issue time in production.
RETRO = ["era5_swvl1", "era5_swvl2", "era5_swvl3", "smap_sm_surface", "smap_sm_rootzone", "smap_sm_profile",
         "era5_stl1_c", "era5_t2m_c"]
STATIC = ["zr_m", "kc", "taw_mm", "season_frac"]

GROUPS = {"sensor": [], "weather": ["wx"], "forecast": ["wx", "fc"], "retro": ["wx", "fc", "retro"]}


def _roll(g, col, win, fn):
    return getattr(g[col].rolling(win, min_periods=max(1, win // 4)), fn)()


def build_features(df: pd.DataFrame):
    """Causal features at issue time t. Returns (frame, {group: [columns]})."""
    df = df.sort_values(["field_id", "timestamp"]).reset_index(drop=True)
    out = df[["field_id", "timestamp", "crop"]].copy()
    g = df.groupby("field_id", sort=False)
    sm = df["sm_obs_causal"]                      # untrusted/anomalous readings are NaN: no look-ahead via cleaning
    df = df.assign(_sm=sm)
    g = df.groupby("field_id", sort=False)
    cols = {"sensor": [], "wx": [], "fc": [], "retro": []}

    def put(group, name, s):
        out[name] = s.to_numpy(); cols[group].append(name)

    put("sensor", "sm0", sm)
    for L in (1, 3, 6, 12, 24, 48, 72, 168):
        put("sensor", f"dsm_{L}", sm - g["_sm"].shift(L))
    for win in (24, 72):
        put("sensor", f"sm_min_{win}", g["_sm"].transform(lambda s, w=win: s.rolling(w, min_periods=w // 4).min()))
        put("sensor", f"sm_max_{win}", g["_sm"].transform(lambda s, w=win: s.rolling(w, min_periods=w // 4).max()))
        put("sensor", f"sm_std_{win}", g["_sm"].transform(lambda s, w=win: s.rolling(w, min_periods=w // 4).std()))
    sat = (sm >= SATURATION).astype(float).where(sm.notna())
    put("sensor", "sat_now", sat)
    put("sensor", "sat_share_72", sat.groupby(df["field_id"]).transform(lambda s: s.rolling(72, min_periods=12).mean()))
    obs = sm.notna()
    last = df["timestamp"].where(obs).groupby(df["field_id"]).ffill()
    put("sensor", "hours_since_obs", (df["timestamp"] - last).dt.total_seconds() / 3600)
    put("sensor", "obs_share_24", obs.astype(float).groupby(df["field_id"]).transform(lambda s: s.rolling(24, min_periods=1).mean()))
    put("sensor", "st_clean", df["st_clean"])
    hr, doy = df["timestamp"].dt.hour, df["timestamp"].dt.dayofyear
    for n, v, p in (("hour", hr, 24), ("doy", doy, 365.25)):
        put("sensor", f"{n}_sin", np.sin(2 * np.pi * v / p)); put("sensor", f"{n}_cos", np.cos(2 * np.pi * v / p))
    out["crop_code"] = df["crop"].map({"sorghum": 0, "corn": 1}).astype(float); cols["sensor"].append("crop_code")
    for c in STATIC:
        put("sensor", c, df[c])
    for c in TRUST:
        put("sensor", f"trust_{c}", df[c].astype(float))
    put("sensor", "batt_min_24", g["battery_voltage"].transform(lambda s: s.rolling(24, min_periods=1).min()))
    put("sensor", "anom_iforest_24", df["anom_iforest"].astype(float).groupby(df["field_id"]).transform(lambda s: s.rolling(24, min_periods=1).mean()))

    rain = df["p_era5_mm_h"].fillna(0)            # rain history (gauge is unreliable; ERA5 stands in for a weather feed)
    for win in (1, 6, 24, 72, 168):
        put("wx", f"rain_{win}h", rain.groupby(df["field_id"]).transform(lambda s, w=win: s.rolling(w, min_periods=1).sum()))
    wet = df["timestamp"].where(rain > 0.2)
    put("wx", "hours_since_rain", (df["timestamp"] - wet.groupby(df["field_id"]).ffill()).dt.total_seconds() / 3600)
    for c in WEATHER_OBS:
        put("wx", c, df[c])
    for win in (24, 72):
        put("wx", f"et0_{win}h", df["et0_mm_h"].groupby(df["field_id"]).transform(lambda s, w=win: s.rolling(w, min_periods=1).sum()))
    for c in FORECAST:
        put("fc", c, df[c])
    for c in RETRO:
        put("retro", c, df[c])
    return out, cols


def columns_for(cols: dict, group: str) -> list:
    keep = ["sensor"] + [{"wx": "wx", "fc": "fc", "retro": "retro"}[k] for k in GROUPS[group]]
    return [c for k in keep for c in cols[k]]
