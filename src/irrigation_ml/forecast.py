"""Soil-moisture forecasting: walk-forward evaluation and final training.

Target = y(t+H) - sm(t) (change), predicted by LightGBM with an L1 (median) objective: the change is heavy-tailed
(rain jumps up to +57) and an L2 loss is dominated by it, biasing quiet-day forecasts. Persistence is the baseline in every table.
Missing targets are masked (no interpolation); LightGBM consumes NaN features natively.
"""
import argparse
import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import ARTIFACTS, FOLDS, HORIZONS, QUANTILES
from .data import add_targets, load_hourly
from .features import build_features, columns_for

# Small trees / strong regularisation chosen on walk-forward folds (larger models overfit and lose to persistence).
PARAMS = dict(n_estimators=100, learning_rate=0.03, num_leaves=4, min_child_samples=500, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0, verbose=-1, n_jobs=4, random_state=0)


def recency_weights(ts: pd.Series, ref: pd.Timestamp, half_life_days):
    if not half_life_days:
        return None
    age = (ref - ts).dt.total_seconds() / 86400
    return np.power(0.5, age / half_life_days).to_numpy()


def make_xy(feat, df, H, cols):
    y = df[f"y{H}"]
    sm0 = feat["sm0"]
    ok = y.notna() & sm0.notna()
    return feat.loc[ok, cols], (y - sm0)[ok], sm0[ok], feat.loc[ok, "timestamp"], ok


def fit(X, d, w=None, objective="l1", alpha=None):
    p = dict(PARAMS, objective=objective)
    if alpha is not None:
        p["alpha"] = alpha
    return lgb.LGBMRegressor(**p).fit(X, d, sample_weight=w)


def predict_sm(model, X, sm0):
    return np.clip(sm0.to_numpy() + model.predict(X), 0, 100)


def metrics(y, p):
    e = np.asarray(p) - np.asarray(y)
    return dict(MAE=np.abs(e).mean(), RMSE=np.sqrt((e ** 2).mean()), n=len(e))


def walk_forward(feat, df, group="forecast", half_life=None, with_quantiles=True):
    cols = columns_for(feat.attrs["cols"], group)
    rows, preds = [], []
    for H in HORIZONS:
        X, d, sm0, ts, ok = make_xy(feat, df, H, cols)
        for a, b in FOLDS:
            a, b = pd.Timestamp(a), pd.Timestamp(b)
            tr = (ts < a - pd.Timedelta(hours=H))            # purge: train targets end before the test window
            te = (ts >= a) & (ts < b)
            if te.sum() < 50 or tr.sum() < 500:
                continue
            w = recency_weights(ts[tr], a, half_life)
            m = fit(X[tr], d[tr], w)
            out = pd.DataFrame({"field_id": feat.loc[X.index[te], "field_id"], "timestamp": ts[te], "H": H,
                                "fold": f"{a.date()}", "y": (d + sm0)[te], "sm0": sm0[te],
                                "persist": sm0[te], "model": predict_sm(m, X[te], sm0[te]),
                                "rain_fc": feat.loc[X.index[te], f"rain_fc_next{H}h"]})
            if with_quantiles:
                for q in QUANTILES:
                    mq = fit(X[tr], d[tr], w, "quantile", q)
                    out[f"q{int(q * 100)}"] = predict_sm(mq, X[te], sm0[te])
            preds.append(out)
    return pd.concat(preds, ignore_index=True)


def summarize(p: pd.DataFrame) -> pd.DataFrame:
    rows = []
    segs = {"all": p["y"].notna(),
            "change>3": (p["y"] - p["sm0"]).abs() > 3,
            "rain_fc>2mm": p["rain_fc"] > 2,
            "dry(fc<0.1mm)": p["rain_fc"] < 0.1}
    for H, gh in p.groupby("H"):
        for s, mask in segs.items():
            g = gh[mask.loc[gh.index]]
            if len(g) < 30:
                continue
            mp, mm = metrics(g["y"], g["persist"]), metrics(g["y"], g["model"])
            r = dict(H=H, segment=s, n=len(g), MAE_persist=mp["MAE"], MAE_model=mm["MAE"],
                     RMSE_persist=mp["RMSE"], RMSE_model=mm["RMSE"],
                     skill_MAE=1 - mm["MAE"] / mp["MAE"], skill_RMSE=1 - mm["RMSE"] / mp["RMSE"])
            if "q10" in g:
                r["cover_10_90"] = ((g["y"] >= g["q10"]) & (g["y"] <= g["q90"])).mean()
            rows.append(r)
    return pd.DataFrame(rows)


def train_final(feat, df, group="forecast", half_life=None):
    """Fit on all labelled history; persist models with their feature order."""
    cols = columns_for(feat.attrs["cols"], group)
    art = {"cols": cols, "group": group, "horizons": {}}
    for H in HORIZONS:
        X, d, sm0, ts, _ = make_xy(feat, df, H, cols)
        w = recency_weights(ts, ts.max(), half_life)
        art["horizons"][H] = {"mean": fit(X, d, w), **{f"q{int(q * 100)}": fit(X, d, w, "quantile", q) for q in QUANTILES}}
    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump(art, ARTIFACTS / "forecast_models.joblib")
    return art


def prepare():
    df = add_targets(load_hourly())
    df = df.sort_values(["field_id", "timestamp"]).reset_index(drop=True)
    feat, cols = build_features(df)
    feat.attrs["cols"] = cols
    return df, feat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablate", action="store_true", help="compare feature groups and recency weighting")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--group", default="forecast")
    ap.add_argument("--half-life", type=float, default=120.0, help="recency weighting (drift adaptation); 0 = off")
    a = ap.parse_args()
    a.half_life = a.half_life or None
    ARTIFACTS.mkdir(exist_ok=True)
    df, feat = prepare()
    pd.set_option("display.width", 220, "display.max_columns", 30)
    if a.ablate:
        res = []
        for group, hl in [("sensor", None), ("weather", None), ("forecast", None), ("retro", None),
                          ("forecast", 180), ("forecast", 90)]:
            s = summarize(walk_forward(feat, df, group, hl, with_quantiles=False))
            s["config"] = f"{group}|hl={hl}"; res.append(s)
        r = pd.concat(res)
        r.to_csv(ARTIFACTS / "forecast_ablation.csv", index=False)
        print(r[r.segment == "all"].pivot(index="config", columns="H", values=["MAE_model", "skill_MAE", "skill_RMSE"]).round(3))
        print(r[r.segment != "all"].pivot_table(index=["segment", "config"], columns="H", values="skill_MAE").round(3))
        print("persistence MAE (all):", r[(r.segment == "all") & (r.config == "sensor|hl=None")].set_index("H")["MAE_persist"].round(3).to_dict())
    else:
        p = walk_forward(feat, df, a.group, a.half_life)
        p.to_parquet(ARTIFACTS / "forecast_oof.parquet")
        s = summarize(p); s.to_csv(ARTIFACTS / "forecast_metrics.csv", index=False)
        print(s.round(3).to_string())
    if a.final:
        train_final(feat, df, a.group, a.half_life)
        print("saved", ARTIFACTS / "forecast_models.joblib")


if __name__ == "__main__":
    main()
