"""Train a retrospective ML surrogate of a PHYSICS-BASED irrigation rule.

This is NOT a model trained against verified irrigation actions, agronomic ground
truth, or observed downstream crop outcomes. It learns provisional FAO-56-style
rule outputs computed with approximate soil/crop parameters. Do not actuate
irrigation or report agreement metrics as real-world recommendation accuracy.

Usage:
  python train_decision_policy.py --data input/irrigation_ml_dataset.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             fbeta_score, precision_score, recall_score)

from irrigation_recommendations import make_decisions

SEED = 42
TARGET_NAME = "proxy_action_from_assumed_FAO56_rule_NOT_GROUND_TRUTH"
CLASSES = ["no_action", "monitor", "review_irrigation"]
# Growing-season test only; two historical fields, one source of positives.
OUTER_FOLDS = [
    ("2025-06-01", "2025-07-15"),
    ("2025-07-15", "2025-08-15"),
    ("2025-08-15", "2025-10-06"),
]
GAP_DAYS = 1
# Validation is carved out of the first historical growing season; strictly
# preceding every 2025 outer test. For threshold selection, NO outer test labels.
INNER_VALID_START = pd.Timestamp("2024-09-01")
INNER_VALID_END = pd.Timestamp("2024-10-01")

# Allowed as-of-decision-time inputs; no rule-generated depletion/RAW/threshold,
# no current recommendation, no future target or future observed weather.
NUMERIC = [
    "smap_theta_rootzone_m3m3", "smap_age_hours", "et0_past24h_mm",
    "kc_assumed", "root_depth_m_assumed", "sensor_moisture_raw",
    "battery_voltage", "sensor_valid_indicator", "precipitation_past24h_mm",
    "soil_temperature_at_decision", "air_temperature_at_decision",
    "anom_rule_at_decision", "anom_iforest_at_decision", "smap_theta_1d_ago", "smap_theta_3d_ago",
    "doy_sin", "doy_cos", "days_since_crop_season_start",
]
CATEGORICAL = ["field_id", "crop", "soil_class_coarse"]
FEATURES = NUMERIC + CATEGORICAL
FORBIDDEN = ["depletion_mm", "raw_mm", "taw_mm", "depletion_over_raw",
             "theta_trigger_nominal", "theta_trigger_min_sensitivity",
             "theta_trigger_max_sensitivity", "nominal_action", "recommendation",
             "assumption_robustness", "rain_forecast_next24h_mm",
             "et0_forecast_next24h_mm", "validated_irrigation_label"]
assert not set(FEATURES) & set(FORBIDDEN)


def load_source(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path, low_memory=False)


def historical_snapshots(raw: pd.DataFrame) -> pd.DataFrame:
    """Recompute teacher decisions from supplied archive instead of fabricating labels."""
    teacher = make_decisions(raw)
    return teacher.sort_values(["field_id", "decision_timestamp"]).reset_index(drop=True)


def engineer_features(decisions: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    d = decisions.copy()
    d["decision_timestamp"] = pd.to_datetime(d["decision_timestamp"], errors="coerce")
    assert d["validated_irrigation_label"].eq(False).all(), "Unexpected validated labels: investigate separately"
    d = d.sort_values(["field_id", "decision_timestamp"]).copy()
    d["doy_sin"] = np.sin(2 * np.pi * d.decision_timestamp.dt.dayofyear / 365.25)
    d["doy_cos"] = np.cos(2 * np.pi * d.decision_timestamp.dt.dayofyear / 365.25)
    # Crop seasons are assumptions from irrigation_recommendations.py, not
    # known sowing dates. This feature is computed from the current timestamp.
    from irrigation_recommendations import CROPS
    d["days_since_crop_season_start"] = d.apply(
        lambda r: (r.decision_timestamp.normalize() - pd.Timestamp(f"{r.decision_timestamp.year}-{CROPS[r.crop]['stages'][0]}")).days
        if r.crop in CROPS else np.nan, axis=1)

    # Daily values are at 06:00. shift by row could bridge missing days, so
    # use calendar-time as-of lookups constrained to at-or-before t-24/t-72.
    for days in (1, 3):
        ref = d[["field_id", "decision_timestamp", "smap_theta_rootzone_m3m3"]].copy()
        ref["decision_timestamp"] = ref["decision_timestamp"] + pd.Timedelta(days=days)
        ref = ref.rename(columns={"smap_theta_rootzone_m3m3": f"smap_theta_{days}d_ago"})
        d = d.merge(ref, on=["field_id", "decision_timestamp"], how="left", validate="one_to_one")

    # Fetch optional observed sensor features at the timestamp or before it.
    # This is historical observation, NOT a future weather forecast.
    optional = {"air_temperature_sensor":"air_temperature_at_decision",
                "soil_temperature_sensor":"soil_temperature_at_decision",
                "any_quality_flag":"anom_rule_at_decision",
                "anom_iforest":"anom_iforest_at_decision"}
    r = raw.copy()
    r["timestamp"] = pd.to_datetime(r["timestamp"], errors="coerce")
    r = r.sort_values(["field_id", "timestamp"]).drop_duplicates(["field_id", "timestamp"])
    exact_cols = [c for c in optional if c in r.columns]
    if exact_cols:
        r2 = r[["field_id", "timestamp", *exact_cols]].rename(columns={"timestamp":"decision_timestamp", **optional})
        d = d.merge(r2, how="left", on=["field_id", "decision_timestamp"], validate="one_to_one")
    for c in ("air_temperature_at_decision", "soil_temperature_at_decision", "anom_rule_at_decision", "anom_iforest_at_decision"):
        if c not in d: d[c] = np.nan
    for c in ("anom_rule_at_decision", "anom_iforest_at_decision"):
        known = d[c].notna()
        d[c] = d[c].astype('string').str.lower().isin(['true','1','yes']).astype(float).where(known)

    # Past rain from *sensor* only, not ERA5 future outcomes. Require at least
    # 18 samples within the 24 hours before the decision, otherwise missing.
    if "precipitation_sensor" in r:
        rain = r[["field_id", "timestamp", "precipitation_sensor"]].copy()
        rain["precipitation_sensor"] = pd.to_numeric(rain.precipitation_sensor, errors="coerce")
        rain.loc[rain.precipitation_sensor.lt(0), "precipitation_sensor"] = np.nan
        # Reindex each field to an hourly grid; prevent 21-day gaps becoming 24 records.
        chunks = []
        for field, sub in rain.groupby("field_id"):
            ix = pd.date_range(sub.timestamp.min(), sub.timestamp.max(), freq='h')
            sub = sub.set_index("timestamp").reindex(ix)
            sub["field_id"] = field
            # strictly past, t-24..t-1 inclusive
            sub["precipitation_past24h_mm"] = sub.precipitation_sensor.shift(1).rolling(24, min_periods=18).sum()
            chunks.append(sub[["field_id", "precipitation_past24h_mm"]].rename_axis("decision_timestamp").reset_index())
        rr = pd.concat(chunks, ignore_index=True)
        d = d.merge(rr, on=["field_id", "decision_timestamp"], how="left", validate="one_to_one")
    else:
        d["precipitation_past24h_mm"] = np.nan

    for c in NUMERIC:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    for c in CATEGORICAL:
        d[c] = d[c].astype('string').fillna('unknown').astype('category')
    d[TARGET_NAME] = d["recommendation"].where(d["recommendation"].isin(CLASSES))
    return d


def make_model() -> LGBMClassifier:
    return LGBMClassifier(
        objective="multiclass", num_class=3, n_estimators=160,
        learning_rate=.035, num_leaves=9, min_child_samples=10,
        reg_lambda=5.0, colsample_bytree=.95, subsample=1.0,
        class_weight="balanced", random_state=SEED, n_jobs=2,
        verbosity=-1,
    )


def train_model(tr: pd.DataFrame) -> LGBMClassifier:
    if len(set(tr[TARGET_NAME])) < len(CLASSES):
        raise ValueError(f"Train contains fewer than 3 proxy classes: {tr[TARGET_NAME].value_counts().to_dict()}")
    mdl = make_model()
    mdl.fit(tr[FEATURES], tr[TARGET_NAME])
    return mdl


def predict_proba(model, dataset: pd.DataFrame) -> pd.DataFrame:
    vals = model.predict_proba(dataset[FEATURES]); out = pd.DataFrame(vals, columns=list(model.classes_), index=dataset.index)
    for c in CLASSES:
        if c not in out:
            out[c] = 0.0
    return out[CLASSES]


def assign_actions(probs: pd.DataFrame, threshold: float) -> np.ndarray:
    """Review alert overrides non-review argmax if its probability clears threshold."""
    default_non_review = probs[["no_action", "monitor"]].idxmax(axis=1).to_numpy(dtype=object)
    default_non_review[probs["review_irrigation"].to_numpy() >= threshold] = "review_irrigation"
    return default_non_review


def pick_threshold(train: pd.DataFrame) -> tuple[float, dict]:
    """Only historical 2024 validation, always before outer 2025 holdouts."""
    fit = train[train.decision_timestamp < INNER_VALID_START - pd.Timedelta(days=GAP_DAYS)].copy()
    validation = train[(train.decision_timestamp >= INNER_VALID_START) &
                       (train.decision_timestamp < INNER_VALID_END)].copy()
    audit = {"inner_train_n":len(fit), "inner_validation_n":len(validation),
             "inner_train_latest":str(fit.decision_timestamp.max()),
             "inner_validation_earliest":str(validation.decision_timestamp.min()),
             "inner_validation_end_exclusive":str(INNER_VALID_END),
             "criterion":"F2 positive=review_irrigation, threshold tuned only in 2024"}
    if len(validation) < 20 or len(fit) < 40 or fit[TARGET_NAME].nunique() < 3 or validation[TARGET_NAME].eq('review_irrigation').sum() == 0:
        audit.update({"note":"insufficient 2024 inner-split support; default 0.50", "threshold":.5})
        return .5, audit
    m = train_model(fit)
    scores = predict_proba(m, validation)['review_irrigation'].to_numpy()
    y = validation[TARGET_NAME].eq('review_irrigation').to_numpy().astype(int)
    candidates = np.round(np.linspace(.05, .90, 86), 3)
    values = [(float(fbeta_score(y, scores>=v, beta=2, zero_division=0)), float(v)) for v in candidates]
    best_score = max(v[0] for v in values)
    # Choose highest threshold among ties to limit gratuitous alerts.
    best = max(t for s,t in values if s >= best_score - 1e-12)
    audit.update({"threshold":best, "inner_f2":best_score,
                  "inner_pos_n":int(y.sum()), "inner_selected_alerts":int((scores>=best).sum())})
    return best, audit


def metrics(y: pd.Series, pred: np.ndarray, review_p: np.ndarray, prefix: dict) -> dict:
    positive = y.eq('review_irrigation').to_numpy()
    pred_positive = pred=='review_irrigation'
    d = {**prefix, 'n':int(len(y)), 'n_review_true_proxy':int(positive.sum()),
         'n_review_pred':int(pred_positive.sum()),
         'agreement_macro_F1':float(f1_score(y, pred, labels=CLASSES, average='macro', zero_division=0)),
         'agreement_weighted_F1':float(f1_score(y,pred,labels=CLASSES,average='weighted',zero_division=0)),
         'review_recall_proxy':float(recall_score(positive,pred_positive,zero_division=0)) if positive.any() else float('nan'),
         'review_precision_proxy':float(precision_score(positive,pred_positive,zero_division=0)) if pred_positive.any() else float('nan'),
         'review_F2_proxy':float(fbeta_score(positive,pred_positive,beta=2,zero_division=0)) if positive.any() else float('nan'),
         'review_false_negatives_proxy':int((positive&~pred_positive).sum()),
         'review_false_positives_proxy':int((~positive&pred_positive).sum()),
         'review_PR_AUC_proxy':float(average_precision_score(positive,review_p)) if positive.any() else float('nan'),
         'balanced_accuracy_proxy':float(balanced_accuracy_score(y,pred)),
         'explicit_truth_available':False,
    }
    return d


def run(raw: pd.DataFrame, outroot: Path, reuse_teacher: Path | None = None) -> dict:
    outroot.mkdir(parents=True, exist_ok=True)
    for sub in ('outputs','models','figures'):
        (outroot/sub).mkdir(exist_ok=True)
    if reuse_teacher and reuse_teacher.exists():
        teacher = pd.read_csv(reuse_teacher, parse_dates=['decision_timestamp'])
    else:
        teacher = historical_snapshots(raw)
        teacher.to_csv(outroot/'outputs'/'decision_prototype_daily.csv', index=False)
    d = engineer_features(teacher,raw)
    df = d.loc[d[TARGET_NAME].notna()].copy()
    df[TARGET_NAME] = pd.Categorical(df[TARGET_NAME], categories=CLASSES)
    assert df[['field_id','decision_timestamp']].duplicated().sum()==0
    assert len(df) and df.decision_timestamp.notna().all()
    assert not set(FEATURES)&set(FORBIDDEN)

    class_counts = {str(k):int(v) for k,v in df[TARGET_NAME].value_counts().items()}
    train_rows, pred_rows, thresholds = [], [], []
    for i,(start,end) in enumerate(OUTER_FOLDS, 1):
        ts = pd.Timestamp(start); te=pd.Timestamp(end)
        train = df.loc[df.decision_timestamp < ts-pd.Timedelta(days=GAP_DAYS)].copy()
        test = df.loc[(df.decision_timestamp >= ts)&(df.decision_timestamp < te)].copy()
        if not len(test): raise ValueError(f'No test rows for {start} - {end}')
        assert train.decision_timestamp.max() < test.decision_timestamp.min()
        assert not set(train.index) & set(test.index)
        assert (test.decision_timestamp<te).all()
        threshold, threshold_info = pick_threshold(train)
        model = train_model(train)
        probs = predict_proba(model, test)
        pred = assign_actions(probs,threshold)
        prefix = {'fold':i,'test_start':start,'test_end_exclusive':end,
                  'train_last_date':str(train.decision_timestamp.max()),
                  'n_train':len(train),'risk_threshold_from_2024_inner_validation':threshold}
        train_rows.append(metrics(test[TARGET_NAME], pred,probs.review_irrigation.to_numpy(),prefix))
        part = test[['field_id','decision_timestamp',TARGET_NAME,'recommendation','assumption_robustness',
                     'smap_theta_rootzone_m3m3','et0_past24h_mm','kc_assumed','battery_voltage']].copy()
        part['fold']=i
        part['ml_predicted_proxy_action']=pred
        for c in CLASSES:
            part[f'ml_probability_{c}']=probs[c].to_numpy()
        part['threshold_from_inner_2024']=threshold
        part['is_proxy_false_negative_review']=(test[TARGET_NAME].to_numpy()=='review_irrigation') & (pred!='review_irrigation')
        part['label_is_validated_irrigation_action']=False
        pred_rows.append(part)
        thresholds.append({'fold':i,**threshold_info})

    pred_df = pd.concat(pred_rows,ignore_index=True).sort_values(['decision_timestamp','field_id'])
    met_df = pd.DataFrame(train_rows)
    met_df.to_csv(outroot/'outputs'/'decision_policy_walkforward_metrics.csv',index=False)
    pred_df.to_csv(outroot/'outputs'/'decision_policy_walkforward_predictions.csv',index=False)
    pd.DataFrame(thresholds).to_csv(outroot/'outputs'/'decision_policy_inner_threshold_audit.csv',index=False)
    met_by_field=[]
    for field, part in pred_df.groupby('field_id'):
        actual=part[TARGET_NAME].astype(str)
        met_by_field.append(metrics(actual,part.ml_predicted_proxy_action.to_numpy(),
                                   part.ml_probability_review_irrigation.to_numpy(),{'field_id':field}))
    pd.DataFrame(met_by_field).to_csv(outroot/'outputs'/'decision_policy_per_field_metrics.csv',index=False)

    # Aggregated unique outer-test predictions; never mix train rows into scores.
    y = pred_df[TARGET_NAME].astype(str)
    pp = pred_df.ml_predicted_proxy_action.to_numpy()
    pooled = metrics(y,pp,pred_df.ml_probability_review_irrigation.to_numpy(),
                     {'split':'pooled_strictly_future_outer_tests'})
    conf=confusion_matrix(y,pp,labels=CLASSES)
    cm=pd.DataFrame(conf,index=[f'teacher_{k}' for k in CLASSES],columns=[f'ml_{k}' for k in CLASSES])
    cm.to_csv(outroot/'outputs'/'decision_policy_confusion_matrix.csv')

    # Train final model on all historical assumed in-season rule examples for
    # research/inference. This is not a backtest score and never touches test eval.
    final_model=train_model(df)
    final_threshold, final_inner = pick_threshold(df)
    joblib.dump({
        'model':final_model,'features':FEATURES,'categories':CLASSES,'risk_threshold':final_threshold,
        'label_definition':TARGET_NAME,'training_source':'assumed FAO-56 style teacher (unverified)',
        'calibration':'2024 train-only historical validation; no test tuning',
        'model_scope':'two historical fields; does not generalize to new field without validation',
        'not_for_automatic_irrigation':True,
    },outroot/'models'/'irrigation_policy_proxy_MULTICLASS_UNVERIFIED.joblib')
    # Persist as-of historical features separate from labels for reproducibility.
    df[['field_id','decision_timestamp',*FEATURES,TARGET_NAME,'recommendation','assumption_robustness']].to_csv(
        outroot/'outputs'/'decision_policy_proxy_training_table.csv',index=False)

    fig, ax=plt.subplots(figsize=(8,5)); ax.imshow(conf,cmap='Blues')
    for row in range(len(CLASSES)):
        for col in range(len(CLASSES)):
            ax.text(col,row,str(conf[row,col]),ha='center',va='center',color='black',fontsize=12)
    ax.set_xticks(range(3),CLASSES,rotation=20,ha='right')
    ax.set_yticks(range(3),CLASSES)
    ax.set_xlabel('ML predicted proxy action');ax.set_ylabel('Unverified FAO-56 proxy action')
    ax.set_title('Walk-forward agreement (NOT real irrigation accuracy)')
    fig.tight_layout();fig.savefig(outroot/'figures'/'decision_policy_confusion.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    xy=met_df[['fold','agreement_macro_F1','review_recall_proxy','review_precision_proxy']].set_index('fold')
    xy.plot(kind='bar',ax=ax,rot=0)
    ax.set_ylim(0,1.05);ax.set_xlabel('Walk-forward outer fold');ax.set_ylabel('Proxy-label agreement')
    ax.set_title('ML rule-distillation agreement (not field-validated metrics)')
    fig.tight_layout();fig.savefig(outroot/'figures'/'decision_policy_fold_agreement.png',dpi=160);plt.close(fig)
    teacher_counts={str(k):int(v) for k,v in teacher.recommendation.value_counts().items()}
    cross=df.groupby(['field_id',TARGET_NAME],observed=True).size().reset_index(name='n')
    cross.to_csv(outroot/'outputs'/'decision_policy_class_counts_by_field.csv',index=False)
    sensitive_positives=int(((df[TARGET_NAME]=='review_irrigation') &
                             df.assumption_robustness.eq('assumption_sensitive')).sum())
    summary={
        'status':'research_only; trained ML policy distilling unverified physics-based actions',
        'real_world_irrigation_precision_recall_available':False,
        'labels':'Teacher = previous FAO-56-style rule with ASSUMED loam soil FC/PWP and crop calendar',
        'target_name':TARGET_NAME,
        'teacher_counts_all_dates':teacher_counts,
        'training_counts_three_in_season_classes':class_counts,
        'proxy_positive_labels_sensitive_to_soil_assumptions':sensitive_positives,
        'all_proxy_positives':int((df[TARGET_NAME]=='review_irrigation').sum()),
        'n_labeled_training_rows':len(df),
        'fields':sorted(map(str,df.field_id.unique())),
        'test_windows':OUTER_FOLDS,
        'gap_days':GAP_DAYS,
        'threshold_selection':'Only historical 2024 validation; independent 2025 outer tests',
        'pooled_outer_test_agreement_WITH_TEACHER_NOT_REAL_WORLD':pooled,
        'final_2024_only_threshold':final_threshold,
        'constraints':[
           'Two fields, review positives only in field_3335: site transfer not demonstrated.',
           'All review candidates depend on uncertain soil hydraulic assumptions.',
           'No true irrigation event labels, no field sensor calibration, no real-time forecast issued times.',
           'Archive ends in October 2025. Predictions are historical, not currently actionable.',
           'The student classifier cannot validate or outperform the teacher measured against the teacher itself.',
           'insufficient_data and out-of-season cases are handled by the rule engine, not relabeled as no_action.',
           'No automatic valve control or agronomic accuracy claims.',
        ],
    }
    (outroot/'outputs'/'decision_policy_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str))
    print(json.dumps(summary,ensure_ascii=False,indent=2,default=str))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--teacher',type=Path,default=None,help='Optional cached decision_prototype_daily.csv from the SAME input')
    args=parser.parse_args()
    run(load_source(args.data),args.output_root,args.teacher)


if __name__=='__main__':
    main()
