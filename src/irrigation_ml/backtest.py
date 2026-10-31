"""Backtest the decision policy on out-of-fold forecasts against the realised sensor outcome."""
import json

import numpy as np
import pandas as pd

from .config import ARTIFACTS, HORIZONS
from .decision import Policy, calibrate_intervals, decide, sensor_thresholds
from .forecast import prepare


DECISION_HOURS = (0, 6, 12, 18)


def build_cases(oof, df, feat, th, offsets, risk):
    d6 = oof[oof["timestamp"].dt.hour.isin(DECISION_HOURS)]
    wide = d6.pivot_table(index=["field_id", "timestamp"], columns="H", values=["q10", "model", "q90", "rain_fc"])
    wide = wide.dropna()
    rk = risk.pivot_table(index=["field_id", "timestamp"], columns="H", values="p")
    base = feat.set_index(["field_id", "timestamp"])
    rows = []
    # realised outcome: lowest trusted reading in (t, t+H]
    sm = df.set_index(["field_id", "timestamp"])["sm_clean"]
    for (fid, t), r in wide.iterrows():
        f = sm.loc[fid]
        fut = {H: f.loc[t + pd.Timedelta(hours=1): t + pd.Timedelta(hours=H)].min() for H in HORIZONS}
        w72 = f.loc[t + pd.Timedelta(hours=1): t + pd.Timedelta(hours=max(HORIZONS))].dropna()
        hit = w72[w72 < th[fid]["trigger"]]
        first_cross = (hit.index[0] - t).total_seconds() / 3600 if len(hit) else np.nan      # hours until the probe really crosses
        b = base.loc[(fid, t)]
        pr = rk.loc[(fid, t)] if (fid, t) in rk.index else None
        if pr is None or pr.isna().any():
            continue
        rows.append(dict(field_id=fid, timestamp=t, sm0=b["sm0"], trigger=th[fid]["trigger"],
                         batt=b["batt_min_24"], obs_share=b["obs_share_24"], anom=b["anom_iforest_24"] > 0.25,
                         taw=df.loc[(df.field_id == fid) & (df.timestamp == t), "taw_mm"].iloc[0],
                         **{f"fc{H}": (r[("q10", H)] - offsets[H], r[("model", H)], r[("q90", H)] + offsets[H]) for H in HORIZONS},
                         **{f"p{H}": pr[H] for H in HORIZONS},
                         **{f"rain{H}": r[("rain_fc", H)] for H in HORIZONS},
                         first_cross_h=first_cross, **{f"min{H}": fut[H] for H in HORIZONS}))
    return pd.DataFrame(rows)


def run(policy=Policy(), verbose=True):
    df, feat = prepare()
    oof = pd.read_parquet(ARTIFACTS / "forecast_oof.parquet")
    th = sensor_thresholds(df, until="2025-04-01")
    off = calibrate_intervals(oof)
    risk = pd.read_parquet(ARTIFACTS / "risk_oof.parquet")
    cases = build_cases(oof, df, feat, th, off, risk)
    cases = cases[cases["sm0"].notna()]
    thf = lambda fid: th[fid]
    dec = [decide(r.sm0, {H: r[f"p{H}"] for H in HORIZONS}, thf(r.field_id), {H: r[f"rain{H}"] for H in HORIZONS},
                  dict(battery_min=r.batt, obs_share=r.obs_share, anomaly=r.anom), policy,
                  {H: r[f"fc{H}"][1] for H in HORIZONS}, r.taw) for _, r in cases.iterrows()]
    cases["action"] = [x["action"] for x in dec]
    cases["stress72"] = cases["min72"] < cases["trigger"]
    cases["stress24"] = cases["min24"] < cases["trigger"]
    cases["naive"] = cases["sm0"] < cases["trigger"]
    cases = cases[cases["min72"].notna()]
    return cases, th, off


def score(cases, flag, truth="stress72"):
    tp = (flag & cases[truth]).sum(); fn = (~flag & cases[truth]).sum(); fp = (flag & ~cases[truth]).sum()
    rec = tp / max(tp + fn, 1); pre = tp / max(tp + fp, 1)
    return dict(flagged=int(flag.sum()), TP=int(tp), FN=int(fn), FP=int(fp), recall=rec, precision=pre,
                F1=2 * rec * pre / max(rec + pre, 1e-9))


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    cases, th, off = run()
    print("thresholds:", json.dumps(th, indent=1)); print("interval offsets:", off)
    print("cases", len(cases), "| actions:", cases.action.value_counts().to_dict(), "| stress72 rate:", cases.stress72.mean().round(3))
    for fid, g in list(cases.groupby("field_id")) + [("ALL", cases)]:
        print(f"\n== {fid} (n={len(g)}, stress72={g.stress72.sum()})")
        res = {"naive sm0<trigger": score(g, g.naive),
               "ML policy (irrigate)": score(g, g.action == "irrigate"),
               "ML policy (irrigate|irrigate_soon)": score(g, g.action.isin(["irrigate", "irrigate_soon"]))}
        print(pd.DataFrame(res).T.round(3))
