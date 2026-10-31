"""No-probe fallback: P(dry within H h) when the probe is down, from the last trusted reading, its age and the weather since.

Trained by simulating outages of random length (6 h .. 25 d) on the history: for each hour t an age a is drawn, the model only sees
the reading at t-a (as if the probe went silent then) plus weather up to t, and is scored against what the probe did after t.
"""
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import ARTIFACTS, FOLDS, HORIZONS
from .forecast import recency_weights

PARAMS = dict(n_estimators=150, learning_rate=0.03, num_leaves=6, min_child_samples=100, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.7, reg_lambda=5.0, verbose=-1, n_jobs=4, random_state=0)
AGE_MIN, AGE_MAX = 6, 600
FEATURES = ["age_h", "last_sm", "gap_last", "last_trend_24", "rain_since", "et0_since", "rain_24h", "rain_72h", "rain_168h",
            "et0_24h", "et0_72h", "hours_since_rain", "air_temp_24h", "crop_code", "doy_sin", "doy_cos", "kc", "zr_m", "taw_mm",
            "season_frac", "rain_fc_next24h", "rain_fc_next48h", "rain_fc_next72h"]


def last_trusted(feat):
    """Per row: last trusted reading, its age in hours (0 if observed now) and the reading 24 h before it."""
    sm = feat["sm0"]
    last = sm.groupby(feat["field_id"]).ffill()
    age = feat["hours_since_obs"]
    return last, age


def _cums(df):
    out = {}
    for fid, g in df.groupby("field_id", sort=False):
        out[fid] = dict(idx=g.index.to_numpy(), rain=np.cumsum(g["p_era5_mm_h"].fillna(0).to_numpy()), et0=np.cumsum(g["et0_mm_h"].fillna(0).to_numpy()))
    return out


def build(df, feat, th, ages=None, seed=0):
    """Feature frame (index = feat index). ages=None -> draw random ages (training / evaluation); else use given ages (inference)."""
    rng = np.random.default_rng(seed)
    n = len(feat); X = pd.DataFrame(index=feat.index, columns=FEATURES, dtype=float)
    base = feat.copy()
    base["_row"] = np.arange(n)
    for fid, g in base.groupby("field_id", sort=False):
        pos = g["_row"].to_numpy(); m = len(pos)
        sm = g["sm0"].to_numpy(float)
        rain = df.loc[g.index, "p_era5_mm_h"].fillna(0).to_numpy(float); et0 = df.loc[g.index, "et0_mm_h"].fillna(0).to_numpy(float)
        cr, ce = np.concatenate([[0], np.cumsum(rain)]), np.concatenate([[0], np.cumsum(et0)])
        t = np.arange(m)
        a = (np.exp(rng.uniform(np.log(AGE_MIN), np.log(AGE_MAX), m)).round().astype(int) if ages is None else np.asarray(ages)[pos].astype(int))
        s0 = t - a
        ok = s0 >= 0
        s0c = np.clip(s0, 0, m - 1)
        last = np.where(ok, sm[s0c], np.nan)
        prev = np.where(s0 - 24 >= 0, sm[np.clip(s0 - 24, 0, m - 1)], np.nan)
        trig = th[fid]["trigger"]
        X.iloc[pos, FEATURES.index("age_h")] = np.where(ok, a, np.nan)
        X.iloc[pos, FEATURES.index("last_sm")] = last
        X.iloc[pos, FEATURES.index("gap_last")] = last - trig
        X.iloc[pos, FEATURES.index("last_trend_24")] = last - prev
        X.iloc[pos, FEATURES.index("rain_since")] = np.where(ok, cr[t + 1] - cr[s0c + 1], np.nan)
        X.iloc[pos, FEATURES.index("et0_since")] = np.where(ok, ce[t + 1] - ce[s0c + 1], np.nan)
    for c in ("rain_24h", "rain_72h", "rain_168h", "et0_24h", "et0_72h", "hours_since_rain", "crop_code", "doy_sin", "doy_cos",
              "kc", "zr_m", "taw_mm", "season_frac", "rain_fc_next24h", "rain_fc_next48h", "rain_fc_next72h"):
        X[c] = feat[c].to_numpy() if c in feat else np.nan
    X["air_temp_24h"] = df["air_temperature_sensor"].groupby(df["field_id"]).transform(lambda s: s.rolling(24, min_periods=6).mean()).to_numpy()
    return X


def walk_forward(df, feat, labels, th, half_life=120.0):
    X = build(df, feat, th, seed=1)
    ts = feat["timestamp"]; res = []
    for H in HORIZONS:
        y = labels[f"stress{H}"]; ok = y.notna() & X["last_sm"].notna()
        for a, b in FOLDS:
            a, b = pd.Timestamp(a), pd.Timestamp(b)
            tr = ok & (ts < a - pd.Timedelta(hours=H)); te = ok & (ts >= a) & (ts < b)
            if te.sum() < 50 or y[tr].sum() < 20:
                continue
            m = lgb.LGBMClassifier(**PARAMS).fit(X.loc[tr], y[tr], sample_weight=recency_weights(ts[tr], a, half_life))
            res.append(pd.DataFrame({"field_id": feat.loc[te, "field_id"], "timestamp": ts[te], "H": H, "fold": str(a.date()), "y": y[te],
                                     "p": m.predict_proba(X.loc[te])[:, 1], "age_h": X.loc[te, "age_h"], "last_sm": X.loc[te, "last_sm"],
                                     "trigger": labels.loc[te, "trigger"], "true_sm0": feat.loc[te, "sm0"]}))
    return pd.concat(res, ignore_index=True)


def train_final(df, feat, labels, th, half_life=120.0):
    X = build(df, feat, th, seed=1); ts = feat["timestamp"]; models = {}
    for H in HORIZONS:
        y = labels[f"stress{H}"]; ok = y.notna() & X["last_sm"].notna()
        models[H] = lgb.LGBMClassifier(**PARAMS).fit(X.loc[ok], y[ok], sample_weight=recency_weights(ts[ok], ts.max(), half_life))
    joblib.dump({"features": FEATURES, "models": models}, ARTIFACTS / "offline_models.joblib")


def summarize(oof: pd.DataFrame) -> pd.DataFrame:
    """Skill by horizon and outage age. 'hard' = the hidden probe is still above the trigger (the real anticipation problem)."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    oof = oof.assign(age_bucket=pd.cut(oof.age_h, [0, 72, 240, 600], labels=["<=3d", "3-10d", "10-25d"]),
                     stale=(oof.last_sm < oof.trigger).astype(float))
    rows = []
    for H, gh in oof.groupby("H"):
        for nm, g in (("all", gh), *[(str(k), v) for k, v in gh.groupby("age_bucket", observed=True)]):
            hard = g[g.true_sm0 >= g.trigger]
            r = dict(H=H, outage_age=nm, n=len(g), events=int(g.y.sum()), base_rate=g.y.mean(), AUC_model=roc_auc_score(g.y, g.p),
                     PR_AUC_model=average_precision_score(g.y, g.p), AUC_stale_rule=roc_auc_score(g.y, g.stale), hard_n=len(hard),
                     hard_events=int(hard.y.sum()), hard_base_rate=hard.y.mean())
            if hard.y.nunique() > 1:
                r.update(hard_AUC=roc_auc_score(hard.y, hard.p), hard_PR_AUC=average_precision_score(hard.y, hard.p))
            rows.append(r)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from .decision import sensor_thresholds
    from .forecast import prepare
    from .risk import build_labels
    df, feat = prepare(); th = sensor_thresholds(df, until="2025-04-01")
    oof = walk_forward(df, feat, build_labels(df, th), th); oof.to_parquet(ARTIFACTS / "offline_oof.parquet")
    print(summarize(oof).round(3).to_string())


def infer_features(df, feat, fid, t, age_h, th):
    """Single-row version of build() for inference: probe silent for age_h hours before t (same arithmetic as the training frame)."""
    g = df[df["field_id"] == fid]
    pos = g.index[g["timestamp"] == t]
    if len(pos) == 0:
        return None
    pt = int(np.where(g.index == pos[0])[0][0]); a = int(max(round(age_h), AGE_MIN)); s0 = pt - a
    sm = feat.loc[g.index, "sm0"].to_numpy(float); rain = g["p_era5_mm_h"].fillna(0).to_numpy(float); et0 = g["et0_mm_h"].fillna(0).to_numpy(float)
    row = feat.loc[pos[0]]
    x = {c: row[c] if c in row else np.nan for c in FEATURES}
    ok = s0 >= 0
    last = sm[s0] if ok else np.nan
    x["age_h"] = a if ok else np.nan
    x["last_sm"] = last
    x["gap_last"] = last - th[fid]["trigger"]
    x["last_trend_24"] = last - sm[s0 - 24] if s0 - 24 >= 0 else np.nan
    x["rain_since"] = rain[s0 + 1: pt + 1].sum() if ok else np.nan
    x["et0_since"] = et0[s0 + 1: pt + 1].sum() if ok else np.nan
    w = g["air_temperature_sensor"].to_numpy(float)[max(0, pt - 23): pt + 1]
    x["air_temp_24h"] = np.nanmean(w) if np.isfinite(w).sum() >= 6 else np.nan
    return pd.DataFrame([x], columns=FEATURES).astype(float)
