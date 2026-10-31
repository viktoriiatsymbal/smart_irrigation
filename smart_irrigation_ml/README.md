# Smart Irrigation — ML engineer deliverable

This package trains and evaluates **soil moisture forecasting** and **daily binary classification** using the uploaded integrated irrigation dataset. It includes real trained models, out-of-fold predictions, a fully reproducible Python pipeline, a demonstration notebook, and audit files. **It is an offline research benchmark, not a production watering controller.**

## What was supplied

- `irrigation_ml_dataset.csv`: 27,346 records, 131 source columns, 2 fields, 2024-03 through 2025-10.
- `data_quality_report.csv`, `data_dictionary.csv`, and exploratory notebooks.
- The actual `hourly_model_table.csv`, `daily_decision_table.csv`, `decision_labels_daily.csv`, `qa_drift.csv`, `pipeline_meta.json`, and `ref_*.csv` described in the specification **were not uploaded**. They are not recreated or represented as genuine source files.
- The `analysis_advanced_real_info.ipynb` notebook, on execution, would make **simulated FAO-56 decision labels**, not observed farmer watering decisions. Its export block also removes `battery_voltage` and anomaly flags from the selected features, contrary to the present ML requirements. Please coordinate a clean export retaining trust indicators.

## Setup and rerun

```bash
cd smart_irrigation_ml
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
mkdir -p input
cp /path/to/irrigation_ml_dataset.csv input/
python ml_pipeline.py --data input/irrigation_ml_dataset.csv
jupyter notebook Smart_Irrigation_ML.ipynb
```

The input CSV is **not** packaged (40 MB, already uploaded separately). Results included in this archive were produced from the uploaded raw integrated dataset. By default, the pipeline searches for `input/hourly_model_table.parquet`, `input/hourly_model_table.csv`, then `input/irrigation_ml_dataset.csv`. You can also use `python ml_pipeline.py --data /absolute/path/to/hourly_model_table.csv`.

When the final DE export is available, put in `input/`:

- `hourly_model_table.csv` or `.parquet`, with `timestamp`, `field_id` and `sm_clean`. Future targets are reconstructed at *exact* t+H on a full hourly index.
- `decision_labels_daily.csv`, with `field_id`, `day` and `act_h24` values `irrigate`, `postpone` or `no_action`; or binary `needs_irrigation` / `label`. If there is a `system` column, this code uses `sprinkler`.
- Optional `daily_decision_table.csv` for current-day known reference features (`kc`, `zr_m`, `p_mad`, `taw_mm` only, and only if they were genuinely known at decision time).
- Optional `ref_fields_soil.csv` containing `field_id` with `fc`, `pwp`, `infil_mm_h` for known static properties.

**Important:** the DE hourly export may omit trust features; `battery_voltage`, `anom_iforest`, `anom_sm_any` and rule flags must be retained by DE for the model to use them. Missing indicators are not hallucinated; they default to zero and will have no predictive information.

## Forecasting: targets, leakage prevention, evaluation

- Targets: measured cleaned soil moisture 24h, 72h and 168h ahead. Reconstructing missing `sm_clean` from the raw data uses **provisional** rules: sensor readings outside [10,100] and repeated plateaus after >=72h at non-saturation readings are flagged. This is **not a reproduction** of the DE cleaning. If DE provides `sm_clean`, it is used as the input target and only these explicitly specified additional rules run.
- No interpolation: original telemetry gaps, including the 20-day interruption, remain `NaN`; no supervised loss on rows missing y(t+H). A full hourly grid avoids accidentally shifting targets by the wrong number of hours.
- Features: only available-at-issuance sensor history, lags, trailing-window aggregates, battery and quality indicators, time-of-day and seasonality. No future sensor, ERA5, SMAP or Open-Meteo values are used. Meteorological observations can be stale in practice and require an operational access check.
- Models: LightGBM to predict the level and LightGBM to predict change from current moisture. Baseline: persistence. Each has 3 walk-forward folds: `[2025-04-01,2025-06-15)`, `[2025-06-15,2025-08-15)`, `[2025-08-15,2025-10-13)`; purge gap equals H and evaluation excludes t+H labels outside the fold. Report MAE and RMSE, separately for each horizon.
- **Observed outcome:** persistence is better than both LightGBM variants in aggregate for all three horizons. Models are provided as trained experiments, **not claimed superior**. See `outputs/forecast_pooled_comparison.csv` and `outputs/forecast_walkforward_metrics.csv`.
- The `models/` directory contains post-evaluation models refitted using all available clean targets to illustrate inference. Performance numbers always come from the strictly held-out fold predictions in `outputs/`.

## Classification and labels

- If genuine `decision_labels_daily.csv` is present, the code trains a classifier on those provided labels. `postpone` is counted as no irrigation for the 24h binary objective. Note that the labels created by the supplied advanced notebook are *simulated decisions*, not field observations.
- Because **no label file was uploaded**, the included **illustration** instead predicts whether **tomorrow at 06:00 sensor moisture will be below 70%**. This is a *low-moisture risk proxy*, not “irrigation needed” and not a validated threshold. Its F1, Recall and PR-AUC cannot be quoted as agronomic irrigation performance.
- Trains balanced LightGBM binary classifier at the 06:00 snapshot, aligns inputs at time t with future target label. Decision threshold is tuned on a strictly later inner train window to optimize F2 (greater weight on Recall); test remains unseen. Outputs show false negatives and false positives for all three folds.
- One class/few examples -> fold skipped; never invents an F1. For actual irrigation deployment, obtain a farmer-verified label or a calibrated agronomic simulator with validated FC/PWP/root depth, irrigation logs, and weather forecasts.

## Quality, drift, and limitations

- `outputs/clean_target_audit.csv`: per-hour sensor validity and reconstructed target.
- `outputs/drift_train_vs_test.csv`: distribution shift audit before/after 2025-04-01. This is not the unprovided `qa_drift.csv`, so no claim about CUSUM/Kendall significance from our outputs.
- Only two fields; `crop` and year are confounded for at least one field; unseen-field and cross-year transfer are not established.
- Raw source is a retrospective merged table; time zone and source publication delay are unknown. **No retrospective ERA5 reanalysis or future precipitation is used as a predictor**.
- High battery/quality relevance is not proof of causation; change detection and cleaning require sensor validation.
- If the target has near-constant values, persistence naturally excels; report that rather than overstating model performance.

## Deliverable contents

- `ml_pipeline.py`: training, feature engineering, chronological validation, calibration, diagnostics, and output export.
- `Smart_Irrigation_ML.ipynb`: human-readable report with code, evaluated results, and plots.
- `models/*.joblib`: trained model plus exact feature order and target mode.
- `outputs/*.csv` / `*.json`: auditable metrics, predictions, train/test drift, target audit, label audit.
- `figures/*.png`: sensor quality, forecasting and classification illustrations.
- `training_log.txt`: original run output.

**Readiness:** regression benchmark reproduced; decision classification pipeline demonstrated using explicitly synthetic labels; data-engineering export integration optional; actual irrigation decision accuracy remains unmeasured. Do not automatically actuate valves based on these models.

## Addendum: provisional agronomic irrigation decision layer (v2)

The script `irrigation_recommendations.py` and notebook `Irrigation_Decision_Prototype.ipynb` implement an **additional, explicit, non-ML agronomic decision prototype** instead of mislabeling a moisture cutoff classifier as an irrigation model.

### Execute

```bash
python irrigation_recommendations.py --data input/irrigation_ml_dataset.csv
jupyter notebook Irrigation_Decision_Prototype.ipynb
python -m unittest discover -s tests -p 'test_recommendations.py' -v
```

Generated output:

- `outputs/decision_prototype_daily.csv`: one 06:00 historical snapshot per field/day and conditional recommendation, assumed physical parameters, remotely sensed moisture, thresholds, QA indicators and sensitivity flags.
- `outputs/decision_prototype_summary.json`: counts and restrictions.
- `outputs/decision_prototype_last_historical.csv`: last 2025 historical snapshots (NOT today's irrigation advice).

Optional genuine archived *issued* weather forecasts:

```bash
python irrigation_recommendations.py --data input/irrigation_ml_dataset.csv --forecast input/issued_forecasts.csv
```

`issued_forecasts.csv` requires `field_id, issue_time, decision_timestamp, rain_next24h_mm, et0_next24h_mm` (timestamps consistently localized). Only forecasts with `issue_time <= decision_timestamp` and no older than 24h are accepted. The CSV provenance must be independently verified; a timestamp field alone does not prove it is a true historical forecast. **Never generate this file from next-day ERA5 observations or retrospectively available datasets.** `postpone_review` is not available without a valid supplied forecast.

### Exact interpretation

- `no_action`: hypothetical daily depletion below RAW under chosen midrange soil parameters; does **not** verify absence of crop stress.
- `monitor`: estimated proximity to RAW or disagreement under reasonable alternative FC/PWP parameters.
- `review_irrigation`: provisional deficit meets FAO-56-style RAW threshold; **a person must review** local field soil and conditions.
- `postpone_review`: future rain forecast supplied as issued-at-time suggests irrigation can be delayed, requires manual verification.
- `no_action_out_of_season`: the notebook's **assumed** sowing/harvest calendar says the crop is not growing; NOT a statement about the actual 2026 field.
- `insufficient_data`: no defensible provisional calculation given missing/stale data.

**Physical basis:** FAO-56 Chapter 8: `TAW=1000*(FC-PWP)*Zr`, `RAW=p*TAW`, `Dr=clip(1000*(FC-theta_rootzone)*Zr,0,TAW)`, and `ETc=Kc*ET0`. https://www.fao.org/4/x0490e/x0490e0e.htm. Parameters for corn, sorghum, phenology, loam midrange FC=0.25/PWP=0.12, and their sensitivity limits originate from **ASSUMPTION tables** in `analysis_advanced_real_info.ipynb` cells 18/20/50, **not from laboratory field calibration**. SMAP L4 root-zone moisture (m³/m³, coarse grid) is used for this physics calculation; the uncalibrated sensor 0–100 measurement is *never* interpreted as volumetric moisture. The latest available SMAP timestamp must be in the past and <=6 hours old. The previous 24h ET0 influences the p adjustment only when at least 18 hourly readings are available; otherwise p_base is an explicit fallback. Future ERA5 reanalysis is **never** used as an issued forecast.

### Observed prototype distribution on attached data

- 1,160 historic daily 06:00 snapshots across both fields; 641 with adequate data in **assumed** growing seasons.
- 339 nominal `no_action`, 241 `monitor`, 61 `review_irrigation`, 22 `insufficient_data`, 497 assumed off-season.
- **All 61** `review_irrigation` cases were `assumption_sensitive` across the broad plausible soil water-retention interval; no case demonstrated a robust dry decision for every parameter scenario.
- All results are historical and end in October 2025. There are no observed irrigation actions, so **F1/Recall/Precision cannot honestly be computed** for agronomic recommendations. The separate LightGBM classifier remains an explicitly labeled demonstration of a low-moisture proxy, not a substitute for validated decisions.

**Priority for production:** verify root-zone soil hydraulics and probe calibration, actual crop dates and growth phase, irrigation logs and delivery rates, actual archived weather forecasts and their issuance times, and agronomist-reviewed decision labels. Then evaluate the ML policy on true event/outcome labels using walk-forward validation with asymmetric false-negative costs. No automatic valve control is enabled or recommended.


## V3 — Proxy-labelled classification + hybrid advisory (2026-10-31)

Run `train_decision_policy.py --data input/irrigation_ml_dataset.csv --teacher outputs/decision_prototype_daily.csv`, `export_proxy_datasets.py --data input/irrigation_ml_dataset.csv --teacher outputs/decision_prototype_daily.csv` and `predict_hybrid_policy.py --data input/irrigation_ml_dataset.csv`.

Open `ML_Decision_Policy_FAO56.ipynb` (already executed), and read **`ML_REPORT_UA.md`** for complete Ukrainian instructions and actual proxy-agreement metrics. Model artifact: `models/irrigation_policy_proxy_MULTICLASS_UNVERIFIED.joblib`.

All labels are preliminary physical-model **proxies**. The classifier agreed with the physics proxy on held-out 2025 samples with pooled macro-F1 0.687, proxy review recall 0.649 (13 false negatives vs teacher), and does not validate real irrigation need. All 61 proxy review candidates are soil-parameter-sensitive, with positives from one field only. The deterministic physical rule remains the main non-actuating advisor. `hybrid_advisory_historical.csv` is partially in-sample and is NOT the independent evaluation table.
