"""Train everything and write artifacts: python -m irrigation_ml.pipeline"""
import json

import pandas as pd

from . import forecast, offline, risk
from .config import ARTIFACTS
from .decision import calibrate_intervals, sensor_thresholds

WF_THRESHOLD_CUTOFF = "2025-04-01"      # walk-forward: trigger defined only from data before the first test window


def main():
    ARTIFACTS.mkdir(exist_ok=True)
    df, feat = forecast.prepare()
    # 1) forecaster: walk-forward report, conformal interval offsets, final models
    oof = forecast.walk_forward(feat, df, "forecast", 120.0)
    oof.to_parquet(ARTIFACTS / "forecast_oof.parquet")
    forecast.summarize(oof).to_csv(ARTIFACTS / "forecast_metrics.csv", index=False)
    forecast.train_final(feat, df, "forecast", 120.0)
    # 2) dry-risk classifier: walk-forward (trigger from pre-test data) and final (all data)
    th_wf = sensor_thresholds(df, until=WF_THRESHOLD_CUTOFF)
    risk.walk_forward(feat, risk.build_labels(df, th_wf)).to_parquet(ARTIFACTS / "risk_oof.parquet")
    th = sensor_thresholds(df)
    risk.train_final(feat, risk.build_labels(df, th))
    # 3) no-probe fallback: simulated outages, walk-forward report, final models
    off_oof = offline.walk_forward(df, feat, risk.build_labels(df, th_wf), th_wf)
    off_oof.to_parquet(ARTIFACTS / "offline_oof.parquet")
    offline.summarize(off_oof).to_csv(ARTIFACTS / "offline_metrics.csv", index=False)
    offline.train_final(df, feat, risk.build_labels(df, th), th)
    (ARTIFACTS / "calibration.json").write_text(json.dumps(
        {"interval_offsets": calibrate_intervals(oof), "thresholds": th, "thresholds_walkforward": th_wf,
         "data_end": str(df["timestamp"].max())}, indent=1))
    print("artifacts written to", ARTIFACTS)


if __name__ == "__main__":
    main()
