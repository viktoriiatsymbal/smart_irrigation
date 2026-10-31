"""Smart irrigation ML: auditable retrospective benchmark (not yet operational)."""
from __future__ import annotations
import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, f1_score, fbeta_score, recall_score,
    precision_score, balanced_accuracy_score, average_precision_score,
    confusion_matrix,
)

SEED = 42
HORIZONS = (24, 72, 168)
FOLDS = (("2025-04-01", "2025-06-15"), ("2025-06-15", "2025-08-15"), ("2025-08-15", "2025-10-13"))
DEAD_SENSOR_BELOW = 10.0  # Provisional engineering QC threshold, not an agronomic threshold.
DEMO_ALERT_THRESHOLD = 70.0  # NOT a validated irrigation threshold.
DECISION_HOUR = 6
MIN_TEST_SAMPLES = 40


def read_source(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def true_boolean(frame: pd.DataFrame, key: str, index=None) -> pd.Series:
    if key not in frame.columns:
        return pd.Series(False, index=frame.index if index is None else index)
    c = frame[key]
    if pd.api.types.is_bool_dtype(c):
        return c.fillna(False)
    return c.astype("string").str.lower().isin(["true", "1", "yes"])


def load_hourly(input_path: Path) -> tuple[pd.DataFrame, dict]:
    raw = read_source(input_path)
    # Optional static DE reference: no leakage from future measurements.
    ref = input_path.parent / "ref_fields_soil.csv"
    if ref.exists() and "field_id" in raw.columns:
        r = pd.read_csv(ref)
        cols = [c for c in ["fc", "pwp", "infil_mm_h", "usda_class"] if c in r.columns]
        if cols and r.field_id.is_unique:
            r = r[["field_id", *cols]].rename(columns={
                "fc": "soil_fc_ref", "pwp": "soil_pwp_ref",
                "infil_mm_h": "soil_infiltration_ref", "usda_class": "soil_class_ref"})
            raw = raw.merge(r, on="field_id", how="left", validate="many_to_one")
    required = {"timestamp", "field_id"}
    if not required.issubset(raw.columns):
        raise ValueError(f"Missing required columns {required - set(raw.columns)}")
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="coerce")
    raw = raw.dropna(subset=["timestamp", "field_id"]).sort_values(["field_id", "timestamp"])
    duplicates = int(raw.duplicated(["field_id", "timestamp"]).sum())
    raw = raw.drop_duplicates(["field_id", "timestamp"], keep="first").copy()

    # Full hourly grid per field; preserves missing periods as NaN, never fills them.
    groups = []
    for field, sub in raw.groupby("field_id", sort=False):
        start, end = sub.timestamp.min(), sub.timestamp.max()
        index = pd.date_range(start, end, freq="h")
        reindexed = sub.set_index("timestamp").reindex(index).rename_axis("timestamp").reset_index()
        reindexed["field_id"] = field
        groups.append(reindexed)
    d = pd.concat(groups, ignore_index=True).sort_values(["field_id", "timestamp"]).reset_index(drop=True)
    raw_target = "sm_clean" if "sm_clean" in d else "soil_moisture_sensor"
    if raw_target not in d:
        raise ValueError("Neither sm_clean nor soil_moisture_sensor is available. Please provide an hourly table with cleaned moisture measurements.")
    sm = pd.to_numeric(d[raw_target], errors="coerce")
    bad_range = (~sm.between(DEAD_SENSOR_BELOW, 100)) & sm.notna()
    low_battery = pd.to_numeric(d.get("battery_voltage", pd.Series(np.nan, index=d.index)), errors="coerce").le(3.8)
    # Causal flat-line detector: flags appear only after 72 full consecutive hours.
    group_id = d["field_id"]
    rolls = d.assign(_sm=sm).groupby(group_id, sort=False)["_sm"]
    flat = rolls.transform(lambda s: (s.rolling(72, min_periods=72).max() - s.rolling(72, min_periods=72).min()).abs().lt(1e-8))
    flat &= sm.lt(99.5) & sm.notna()
    flag = true_boolean(d, "flag_soil_moisture_range") | true_boolean(d, "duplicate_timestamp")
    core_bad = bad_range | flag | flat
    # Do not smooth through missing data or censor points solely on low battery.
    # If the DE table already contains sm_clean, preserve its quality decisions.
    d["sm_clean"] = sm.where(~core_bad)
    d["sensor_invalid_now"] = core_bad.astype("int8")
    d["sensor_stuck_now"] = flat.astype("int8")
    d["battery_low"] = low_battery.astype("int8")
    d["sm_observed"] = d["sm_clean"].notna().astype("int8")
    d["sensor_record_missing"] = d["soil_moisture_sensor"].isna().astype("int8") if "soil_moisture_sensor" in d else d["sm_clean"].isna().astype("int8")
    d["quality_any"] = true_boolean(d, "any_quality_flag").astype("int8")
    # Include all rule and IF scores if DE exported them, but avoid recomputing IF on test data.
    d["anom_iforest_indicator"] = true_boolean(d, "anom_iforest").astype("int8")
    d["anom_rule_indicator"] = true_boolean(d, "anom_sm_any").astype("int8") | d["sensor_invalid_now"]
    qa = {
        "input": str(input_path), "source_target": raw_target,
        "duplicate_rows_removed": duplicates, "raw_rows": int(len(raw)),
        "full_hourly_rows": int(len(d)), "n_fields": int(d.field_id.nunique()),
        "clean_target_coverage_pct": round(float(d.sm_clean.notna().mean() * 100), 2),
        "bad_sensor_records": int(core_bad.sum()),
        "stuck_sensor_records": int(flat.sum()),
        "low_battery_records": int(low_battery.sum()),
        "notes": [
            "Cleaning reconstructed from available raw data; not identical to teammate's sm_clean.",
            "sm < 10 assumed faulty sensor readings; engineering assumption pending confirmation.",
            "No linear interpolation. NaN targets excluded from supervised loss.",
            "Retrospective benchmark only; real-time sensor access and forecasting weather feeds require verification.",
        ],
    }
    return d, qa


def make_features(d: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    f = d.copy()
    ts = f.timestamp
    f["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24)
    f["doy_sin"] = np.sin(2 * np.pi * ts.dt.dayofyear / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * ts.dt.dayofyear / 365.25)
    # A site's current sensor state is known at forecast issuance, not t+H.
    measured = ["sm_clean", "battery_voltage", "air_temperature_sensor", "air_humidity_sensor",
                "soil_temperature_sensor", "precipitation_sensor", "solar_panel_voltage"]
    flags = ["sm_observed", "battery_low", "sensor_invalid_now", "sensor_stuck_now",
             "quality_any", "anom_iforest_indicator", "anom_rule_indicator", "sensor_record_missing"]
    cols = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"] + flags
    for c in measured:
        if c in f.columns:
            f[c] = pd.to_numeric(f[c], errors="coerce")
            cols.append(c)
    for c in ["sm_clean", "air_temperature_sensor", "precipitation_sensor"]:
        if c not in f:
            continue
        for lag in [1, 6, 24, 72, 168]:
            out = f"{c}_lag{lag}"
            f[out] = f.groupby("field_id", sort=False)[c].shift(lag)
            cols.append(out)
    for win in [6, 24, 72]:
        grouped = f.groupby("field_id", sort=False)["sm_clean"]
        shifted = grouped.shift(1)
        fid = f["field_id"]
        f[f"sm_mean_prev{win}h"] = shifted.groupby(fid, sort=False).transform(
            lambda v: v.rolling(win, min_periods=max(2, win//2)).mean())
        f[f"sm_count_prev{win}h"] = shifted.notna().groupby(fid, sort=False).transform(
            lambda v: v.rolling(win, min_periods=1).sum())
        cols.extend([f"sm_mean_prev{win}h", f"sm_count_prev{win}h"])
    for c in ["precipitation_sensor"]:
        if c in f:
            for win in [24, 72]:
                f[f"rain_past{win}h"] = f.groupby("field_id", sort=False)[c].transform(
                    lambda v: v.clip(lower=0).rolling(win, min_periods=max(1, win//2)).sum())
                cols.append(f"rain_past{win}h")
    # Field indicators allow within-site training without exposing a numeric field_id.
    # Their site generalization remains unverified (only two sites).
    f["field_3335"] = (f["field_id"] == "field_3335").astype("int8")
    cols.append("field_3335")
    for c in ["soil_fc_ref", "soil_pwp_ref", "soil_infiltration_ref"]:
        if c in f:
            f[c] = pd.to_numeric(f[c], errors="coerce")
            cols.append(c)
    # Lag labels are computed on exact hourly grids to prevent 20-day gaps being mistaken for H hours.
    for h in HORIZONS:
        f[f"target_{h}h"] = f.groupby("field_id", sort=False)["sm_clean"].shift(-h)
    assert not any(c.startswith("target_") for c in cols)
    assert len(cols) == len(set(cols))
    return f, cols


def build_fold(frame: pd.DataFrame, target: str, start: str, end: str, horizon_hours: int):
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    valid = frame[target].notna()
    # Remove train rows whose label t+H reaches test, and test rows whose label is beyond the fold end.
    train = valid & (frame.timestamp < t0 - pd.Timedelta(hours=horizon_hours))
    test = valid & frame.timestamp.ge(t0) & (frame.timestamp + pd.Timedelta(hours=horizon_hours)).lt(t1)
    return train, test


def regressor(mode="level"):
    return LGBMRegressor(n_estimators=200, learning_rate=0.055, num_leaves=15,
                         max_depth=7, min_child_samples=75, colsample_bytree=0.9,
                         reg_lambda=2.0, verbosity=-1, random_state=SEED, n_jobs=2)


def calc_regression(y, yhat):
    return {"MAE": float(mean_absolute_error(y, yhat)),
            "RMSE": float(np.sqrt(mean_squared_error(y, yhat)))}


def train_forecasting(f, features, root: Path):
    rows, predictions = [], []
    for horizon in HORIZONS:
        target = f"target_{horizon}h"
        for i, (start, end) in enumerate(FOLDS, 1):
            train, test = build_fold(f, target, start, end, horizon)
            # Train only on currently credible sensor data: current and future targets known.
            train &= f.sm_clean.notna()
            test &= f.sm_clean.notna()
            if train.sum() < 200 or test.sum() < MIN_TEST_SAMPLES:
                rows.append({"horizon_h": horizon, "fold": i, "model": "SKIPPED", "n_train": int(train.sum()), "n_test": int(test.sum()), "reason": "not enough clean observations"})
                continue
            Xtr, Xte = f.loc[train, features], f.loc[test, features]
            ytr, yte = f.loc[train, target], f.loc[test, target]
            base = f.loc[test, "sm_clean"].clip(0, 100)
            for name, yp in [("persistence", base)]:
                met = calc_regression(yte, yp)
                rows.append(dict(horizon_h=horizon, fold=i, model=name, n_train=int(train.sum()), n_test=int(test.sum()), **met))
            for mode in ["level", "delta"]:
                mdl = regressor(mode)
                y_train = ytr if mode == "level" else ytr - f.loc[train, "sm_clean"]
                mdl.fit(Xtr, y_train)
                yp = mdl.predict(Xte)
                if mode == "delta":
                    yp += f.loc[test, "sm_clean"].to_numpy()
                yp = np.clip(yp, 0, 100)
                met = calc_regression(yte, yp)
                rows.append(dict(horizon_h=horizon, fold=i, model=f"lightgbm_{mode}", n_train=int(train.sum()), n_test=int(test.sum()), **met))
                part = f.loc[test, ["timestamp", "field_id"]].copy()
                part["horizon_h"] = horizon
                part["fold"] = i
                part["model"] = f"lightgbm_{mode}"
                part["y_true"] = yte.to_numpy()
                part["y_pred"] = yp
                part["y_persistence"] = base.to_numpy()
                predictions.append(part)
    res = pd.DataFrame(rows)
    res.to_csv(root / "outputs" / "forecast_walkforward_metrics.csv", index=False)
    if predictions:
        pd.concat(predictions, ignore_index=True).to_csv(root / "outputs" / "forecast_predictions.csv", index=False)
    # Fit final models only after all walk-forward test evaluations, for future inference.
    for horizon in HORIZONS:
        valid = f[f"target_{horizon}h"].notna() & f.sm_clean.notna()
        if valid.sum() < 200:
            continue
        # Two modes are exported; winner should be selected by out-of-sample folds, not in-sample score.
        for mode in ["level", "delta"]:
            model = regressor(mode)
            y = f.loc[valid, f"target_{horizon}h"]
            if mode == "delta":
                y = y - f.loc[valid, "sm_clean"]
            model.fit(f.loc[valid, features], y)
            joblib.dump({"model": model, "features": features, "target_mode": mode, "horizon_h": horizon,
                         "issued_at": "historical backtest; check feature freshness for actual deployment"},
                        root / "models" / f"soil_moisture_{horizon}h_{mode}.joblib")
    return res


def find_external_daily(root: Path):
    candidates = [
        root / "input" / "decision_labels_daily.csv", root.parent / "decision_labels_daily.csv",
        root / "input" / "decision_labels_daily.parquet", root.parent / "decision_labels_daily.parquet",
        root / "input" / "daily_decision_table.csv", root.parent / "daily_decision_table.csv",
        root / "input" / "daily_decision_table.parquet", root.parent / "daily_decision_table.parquet",
    ]
    for p in candidates:
        if p.exists():
            t = read_source(p)
            if "field_id" not in t:
                continue
            label = next((c for c in ["needs_irrigation", "irrigation_needed", "label", "act_h24", "act_h24_noisy"] if c in t), None)
            if label is None:
                continue
            t["day"] = pd.to_datetime(t.get("day", t.get("date", t.get("timestamp"))), errors="coerce").dt.normalize()
            if "system" in t.columns:
                t = t[t["system"].astype(str).eq("sprinkler")].copy()
            t = t.dropna(subset=[label, "day"])
            vals = t[label].astype(str).str.lower().str.strip()
            recognized_true = {"irrigate", "1", "true", "yes", "y", "так"}
            recognized_false = {"no_action", "postpone", "0", "false", "no", "n", "ні"}
            is_valid = vals.isin(recognized_true | recognized_false)
            t = t.loc[is_valid, ["field_id", "day", label]].copy()
            t["label"] = vals.loc[is_valid].isin(recognized_true).astype(int)
            t = t.drop_duplicates(["field_id", "day"], keep="first")
            return t[["field_id", "day", "label"]], f"external:{p.name} ({label}; system=sprinkler if present)"
    return None, None


def build_daily_labels(f: pd.DataFrame, root: Path, features: list[str]):
    day = f[f.timestamp.dt.hour.eq(DECISION_HOUR)].copy().reset_index(drop=True)
    day["day"] = day.timestamp.dt.normalize()
    daily_features = list(features)
    # The DE daily table is optional. Only static / decision-time known
    # properties are admitted; same-day full aggregates would be leakage.
    tables = [root / "input" / "daily_decision_table.csv",
              root / "input" / "daily_decision_table.parquet",
              root.parent / "daily_decision_table.csv",
              root.parent / "daily_decision_table.parquet"]
    for p in tables:
        if not p.exists():
            continue
        dt = read_source(p)
        if not {"field_id"}.issubset(dt.columns):
            continue
        dt["day"] = pd.to_datetime(dt.get("day", dt.get("timestamp")), errors="coerce").dt.normalize()
        if "system" in dt:
            dt = dt[dt.system.astype(str).eq("sprinkler")]
        safe = [c for c in ["kc", "zr_m", "p_mad", "taw_mm"] if c in dt.columns and c not in day.columns]
        if safe:
            add = dt[["field_id", "day", *safe]].drop_duplicates(["field_id", "day"])
            add = add.rename(columns={c: "reference_" + c for c in safe})
            day = day.merge(add, on=["field_id", "day"], how="left", validate="one_to_one")
            daily_features.extend("reference_" + c for c in safe)
        break
    truth, source = find_external_daily(root)
    if truth is None:
        # Illustrative next-day low-moisture RISK task, NOT ground truth for irrigation!
        day["label"] = np.where(day["target_24h"].notna(),
                                (day["target_24h"] < DEMO_ALERT_THRESHOLD).astype(float), np.nan)
        source = f"DEMO_PROXY: next-day clean sensor moisture < {DEMO_ALERT_THRESHOLD}; not actual irrigation need"
    else:
        day = day.merge(truth, on=["field_id", "day"], how="left", validate="one_to_one")
    return day, source, daily_features


def classifier():
    return LGBMClassifier(n_estimators=170, learning_rate=.05, num_leaves=11, max_depth=6,
                          min_child_samples=30, reg_lambda=2., random_state=SEED,
                          class_weight="balanced", verbosity=-1, n_jobs=2)


def choose_threshold(y_val: np.ndarray, probability: np.ndarray) -> float:
    if len(np.unique(y_val)) != 2:
        return .5
    candidates = np.arange(.15, .851, .05)
    vals = [(fbeta_score(y_val, probability >= z, beta=2, zero_division=0), z) for z in candidates]
    return float(max(vals, key=lambda x: (x[0], x[1]))[1])


def class_metrics(y, pred, probs):
    return dict(F1=float(f1_score(y, pred, zero_division=0)),
                Recall=float(recall_score(y, pred, zero_division=0)),
                Precision=float(precision_score(y, pred, zero_division=0)),
                F2=float(fbeta_score(y, pred, beta=2, zero_division=0)),
                PR_AUC=float(average_precision_score(y, probs)) if len(np.unique(y)) > 1 else float("nan"),
                FN=int(((y == 1) & (pred == 0)).sum()),
                FP=int(((y == 0) & (pred == 1)).sum()))


def train_classification(f, features, root: Path):
    day, source, features = build_daily_labels(f, root, features)
    results, outs = [], []
    day.to_csv(root / "outputs" / "daily_label_audit.csv", columns=["field_id", "timestamp", "day", "label"], index=False)
    for i, (start, end) in enumerate(FOLDS, 1):
        tr, te = build_fold(day, "label", start, end, 24)
        if tr.sum() < 60 or te.sum() < 15 or day.loc[tr, "label"].nunique() < 2:
            results.append({"fold": i, "status": "SKIPPED", "n_train": int(tr.sum()), "n_test": int(te.sum()), "reason": "insufficient rows or class variation"})
            continue
        Xtr, ytr = day.loc[tr, features], day.loc[tr, "label"].astype(int)
        Xte, yte = day.loc[te, features], day.loc[te, "label"].astype(int)
        # Chronological threshold tuning; tune on late train only; purge a 24h boundary.
        end_train = day.loc[tr, "timestamp"].quantile(.80)
        fit_mask = day.loc[tr, "timestamp"].lt(end_train - pd.Timedelta(hours=24))
        val_mask = day.loc[tr, "timestamp"].ge(end_train)
        threshold = .5
        if fit_mask.sum() > 40 and val_mask.sum() > 20 and ytr[fit_mask].nunique() == 2:
            cal = classifier()
            cal.fit(Xtr.loc[fit_mask], ytr.loc[fit_mask])
            threshold = choose_threshold(ytr.loc[val_mask].to_numpy(), cal.predict_proba(Xtr.loc[val_mask])[:, 1])
        mdl = classifier()
        mdl.fit(Xtr, ytr)
        prob = mdl.predict_proba(Xte)[:, 1]
        pred = (prob >= threshold).astype(int)
        results.append(dict(fold=i, status="OK", n_train=int(tr.sum()), n_test=int(te.sum()),
                            positive_train=int(ytr.sum()), positive_test=int(yte.sum()), threshold=threshold,
                            **class_metrics(yte.to_numpy(), pred, prob)))
        out = day.loc[te, ["field_id", "timestamp", "day"]].copy()
        out["fold"] = i
        out["true_label"] = yte.to_numpy()
        out["probability"] = prob
        out["prediction"] = pred
        outs.append(out)
    res = pd.DataFrame(results)
    res.to_csv(root / "outputs" / "classification_walkforward_metrics.csv", index=False)
    if outs:
        pd.concat(outs).to_csv(root / "outputs" / "classification_predictions.csv", index=False)
    valid = day.label.notna()
    if day.loc[valid, "label"].nunique() == 2:
        model = classifier()
        model.fit(day.loc[valid, features], day.loc[valid, "label"].astype(int))
        joblib.dump({"model": model, "features": features, "label_source": source,
                     "decision_hour": DECISION_HOUR,
                     "WARNING": "Do NOT deploy for irrigation without validated agronomic labels and live prospective evaluation"},
                    root / "models" / "daily_risk_classifier_DEMO_OR_REAL.joblib")
    return res, source, day


def drift_report(f, features, root):
    # Train/test feature shift: no data leakage because these statistics are post-hoc audit only.
    train = f.timestamp.lt("2025-04-01")
    test = f.timestamp.ge("2025-04-01")
    rows = []
    for name in ["sm_clean", "battery_voltage", "air_temperature_sensor", "soil_temperature_sensor"]:
        if name not in f:
            continue
        for field_id, d in f.groupby("field_id"):
            before = pd.to_numeric(d.loc[train.loc[d.index], name], errors="coerce").dropna()
            after = pd.to_numeric(d.loc[test.loc[d.index], name], errors="coerce").dropna()
            if len(before) < 20 or len(after) < 20:
                continue
            rows.append({"feature": name, "field_id": field_id,
                         "train_median": float(before.median()), "test_median": float(after.median()),
                         "difference": float(after.median() - before.median()),
                         "n_train": len(before), "n_test": len(after)})
    report = pd.DataFrame(rows)
    report.to_csv(root / "outputs" / "drift_train_vs_test.csv", index=False)
    return report


def save_figures(f, root):
    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for ax, (field, d) in zip(axs, f.groupby("field_id")):
        ax.plot(d.timestamp, d["sm_clean"], label="clean (heuristic)", lw=.65)
        if "soil_moisture_sensor" in d:
            ax.plot(d.timestamp, d.soil_moisture_sensor, label="raw", alpha=.25, lw=.4)
        ax.set_title(field)
        ax.set_ylabel("Sensor moisture (%)")
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(root / "figures" / "sensor_cleaning.png", dpi=140)
    plt.close(fig)

    preds = root / "outputs" / "forecast_predictions.csv"
    if preds.exists():
        p = pd.read_csv(preds, parse_dates=["timestamp"])
        for horizon in HORIZONS:
            view = p[(p.horizon_h == horizon) & (p.model == "lightgbm_delta")].sort_values("timestamp")
            if view.empty:
                continue
            field = view.field_id.value_counts().index[0]
            view = view[view.field_id == field]
            fig, ax = plt.subplots(figsize=(13, 4))
            ax.plot(view.timestamp, view.y_true, lw=1.0, label="observed t+H")
            ax.plot(view.timestamp, view.y_pred, lw=.7, label="model")
            ax.plot(view.timestamp, view.y_persistence, lw=.5, alpha=.6, label="persistence")
            ax.set_title(f"{horizon}h horizon - {field} | walk-forward, held-out only")
            ax.set_ylabel("Soil moisture (%)")
            ax.legend()
            fig.tight_layout()
            fig.savefig(root / "figures" / f"forecast_{horizon}h.png", dpi=140)
            plt.close(fig)
    p = root / "outputs" / "classification_predictions.csv"
    if p.exists():
        c = pd.read_csv(p)
        cm = confusion_matrix(c.true_label, c.prediction, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(5, 4))
        heat = ax.imshow(cm)
        fig.colorbar(heat, ax=ax)
        ax.set_xticks([0, 1], labels=["no", "yes"])
        ax.set_yticks([0, 1], labels=["no", "yes"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Out-of-fold classification (proxy unless external labels supplied)")
        fig.tight_layout()
        fig.savefig(root / "figures" / "classification_confusion.png", dpi=140)
        plt.close(fig)


def main(input_path: str | None = None, root: str | None = None):
    root = Path(root) if root else Path(__file__).resolve().parent
    for sub in ["models", "figures", "outputs", "input"]:
        (root/sub).mkdir(parents=True, exist_ok=True)
    if input_path:
        source = Path(input_path)
    else:
        possible = [root/"input"/"hourly_model_table.parquet", root/"input"/"hourly_model_table.csv",
                    root/"input"/"irrigation_ml_dataset.csv", root.parent/"irrigation_ml_dataset.csv"]
        source = next((p for p in possible if p.exists()), None)
        if source is None:
            raise FileNotFoundError("Place hourly_model_table.csv or irrigation_ml_dataset.csv in the input/ folder or pass --data.")
    f, qa = load_hourly(source)
    f, features = make_features(f)
    qa["feature_count"] = len(features)
    qa["features"] = features
    qa["forecasts"] = list(HORIZONS)
    qa["decision_time"] = f"{DECISION_HOUR:02d}:00 (assumed same naive time basis as input; timezone unverified)"
    with open(root/"outputs"/"quality_and_config.json", "w") as out:
        json.dump(qa, out, indent=2, ensure_ascii=False)
    # Keep intermediate only as a compact audit table, not the massive raw 131-col CSV.
    f[["field_id", "timestamp", "sm_clean", "sensor_invalid_now", "sensor_stuck_now", "battery_low", "sm_observed"]].to_csv(root/"outputs"/"clean_target_audit.csv", index=False)
    print("LOADED", source, "fields", qa["n_fields"], "rows", qa["full_hourly_rows"], "clean coverage", qa["clean_target_coverage_pct"])
    print("REGRESSION: horizons", HORIZONS)
    reg = train_forecasting(f, features, root)
    print(reg.to_string(index=False))
    print("CLASSIFICATION")
    cls, source, daily = train_classification(f, features, root)
    print("LABEL SOURCE:", source)
    print(cls.to_string(index=False))
    drift_report(f, features, root)
    save_figures(f, root)
    result = {"data_quality": qa, "classification_label_source": source, "classification_is_real_irrigation": source.startswith("external:"),
              "notes": ["Model benchmark must beat persistence, checked independently for each horizon.",
                        "Only 2 fields; transfer to unseen fields NOT established.",
                        "Synthetic/demo labels cannot establish irrigation decision performance.",
                        "Advanced notebook constructs simulated FAO-56 irrigation labels; these are also not actual irrigation events.",
                        "Only measurements at issuance and earlier are used; no future ERA5, SMAP, or Open-Meteo reanalysis."],
              "regression_metrics": reg.replace({np.nan: None}).to_dict(orient="records"),
              "classification_metrics": cls.replace({np.nan: None}).to_dict(orient="records")}
    with open(root/"outputs"/"run_summary.json", "w") as out:
        json.dump(result, out, indent=2, ensure_ascii=False, default=str)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", help="CSV or Parquet of hourly data")
    parser.add_argument("--root", help="Output root for project")
    args = parser.parse_args()
    main(args.data, args.root)
