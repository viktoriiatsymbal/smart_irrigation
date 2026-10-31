"""Direct dry-risk classifier: P(probe falls below the field's trigger within H hours).

Label = lowest trusted reading in (t, t+H] < field trigger (realised outcome; masked when too few trusted future
readings). Features are the forecasting features plus distance-to-trigger. Walk-forward validated like the forecaster.
"""
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .config import ARTIFACTS, FOLDS, HORIZONS
from .decision import sensor_thresholds
from .features import columns_for
from .forecast import recency_weights

PARAMS = dict(n_estimators=150, learning_rate=0.03, num_leaves=6, min_child_samples=100, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.7, reg_lambda=5.0, verbose=-1, n_jobs=4, random_state=0)


def build_labels(df: pd.DataFrame, th: dict) -> pd.DataFrame:
    out = df[["field_id", "timestamp"]].copy()
    trig = df["field_id"].map({k: v["trigger"] for k, v in th.items()})
    out["trigger"] = trig
    sm = df["sm_clean"]
    for H in HORIZONS:
        # window (t, t+H]  == rolling(H) evaluated at t+H, shifted back by H
        fmin = sm.groupby(df["field_id"]).transform(lambda s: s.rolling(H, min_periods=H // 4).min().shift(-H))
        out[f"stress{H}"] = (fmin < trig).astype(float).where(fmin.notna())
    return out


def design(feat, labels, cols):
    X = feat[cols].copy()
    X["gap_to_trigger"] = feat["sm0"] - labels["trigger"]
    return X


def walk_forward(feat, labels, group="forecast", half_life=120.0):
    cols = columns_for(feat.attrs["cols"], group)
    X = design(feat, labels, cols)
    res = []
    for H in HORIZONS:
        y = labels[f"stress{H}"]
        ok = y.notna() & feat["sm0"].notna()
        for a, b in FOLDS:
            a, b = pd.Timestamp(a), pd.Timestamp(b)
            ts = feat["timestamp"]
            tr = ok & (ts < a - pd.Timedelta(hours=H)); te = ok & (ts >= a) & (ts < b)
            if te.sum() < 50 or y[tr].sum() < 20:
                continue
            m = lgb.LGBMClassifier(**PARAMS).fit(X[tr], y[tr], sample_weight=recency_weights(ts[tr], a, half_life))
            res.append(pd.DataFrame({"field_id": feat.loc[te, "field_id"], "timestamp": ts[te], "H": H, "fold": str(a.date()),
                                     "y": y[te], "p": m.predict_proba(X[te])[:, 1], "gap": X.loc[te, "gap_to_trigger"]}))
    return pd.concat(res, ignore_index=True)


def train_final(feat, labels, group="forecast", half_life=120.0):
    cols = columns_for(feat.attrs["cols"], group)
    X = design(feat, labels, cols); ts = feat["timestamp"]
    models = {}
    for H in HORIZONS:
        y = labels[f"stress{H}"]; ok = y.notna() & feat["sm0"].notna()
        models[H] = lgb.LGBMClassifier(**PARAMS).fit(X[ok], y[ok], sample_weight=recency_weights(ts[ok], ts.max(), half_life))
    joblib.dump({"cols": cols, "models": models}, ARTIFACTS / "risk_models.joblib")


if __name__ == "__main__":
    from .forecast import prepare
    df, feat = prepare()
    th = sensor_thresholds(df, until="2025-04-01")
    labels = build_labels(df, th)
    for H in HORIZONS:
        print(H, "positive rate", labels[f"stress{H}"].mean().round(3))
    oof = walk_forward(feat, labels)
    oof.to_parquet(ARTIFACTS / "risk_oof.parquet")
    for H, g in oof.groupby("H"):
        a = g[g.gap > 0]            # anticipation: currently above trigger
        print(f"H={H}: all PR-AUC={average_precision_score(g.y, g.p):.3f} (base {g.y.mean():.3f}) | "
              f"anticipation n={len(a)} pos={int(a.y.sum())} PR-AUC={average_precision_score(a.y, a.p):.3f} (base {a.y.mean():.3f})")
