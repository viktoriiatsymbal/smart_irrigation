# Smart Irrigation — ML Engineer Deliverable ONLY

This package contains **only the ML Engineer work**: forecasting, retrospective FAO-56-style decisions, ML proxy classification, hybrid advisory, trained models, evaluation, tests and notebooks. No Data Engineering exploratory notebooks, source data dictionary, QA source report or integrated CSV is bundled.

## Contents

- `Smart_Irrigation_ML.ipynb` — soil moisture forecasting experiments and plots.
- `ML_Decision_Policy_FAO56.ipynb` — proxy-label decision classification, evaluation and discussion.
- `Irrigation_Decision_Prototype.ipynb` — transparent physics decision prototype.
- `ml_pipeline.py` — feature engineering, cleaned target reconstruction, time-aware evaluation for 24/72/168 hours.
- `irrigation_recommendations.py` — FAO-56-style approximate water-balance proxy decisions.
- `train_decision_policy.py` — 3-class LightGBM learning and walk-forward validation.
- `predict_hybrid_policy.py` — historical combined advisory; does **not** control irrigation.
- `export_proxy_datasets.py` — optional daily feature/proxy-label exports.
- `models/`, `outputs/`, `figures/`, `tests/` — trained artefacts, historical results, plots, integrity tests.
- `README.md` and `ML_REPORT_UA.md` — methodology, caveats and results.

## Quick start

1. Extract ZIP, then `cd smart_irrigation_ml`.
2. Install Python requirements:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   python -m pip install -r requirements.txt
   ```

3. To **view** the completed ML results, open the included notebooks and `ML_REPORT_UA.md` directly. No dataset is needed to view the saved results.
4. To **rerun training** against the original team data, put `irrigation_ml_dataset.csv` (provided by your Data Engineering team) into a new `input/` directory:

   ```bash
   mkdir -p input
   cp /path/to/irrigation_ml_dataset.csv input/
   python ml_pipeline.py --data input/irrigation_ml_dataset.csv
   python irrigation_recommendations.py --data input/irrigation_ml_dataset.csv
   python train_decision_policy.py --data input/irrigation_ml_dataset.csv
   python export_proxy_datasets.py --data input/irrigation_ml_dataset.csv --teacher outputs/decision_prototype_daily.csv
   python predict_hybrid_policy.py --data input/irrigation_ml_dataset.csv
   python -m unittest discover -s tests -p 'test_*.py' -v
   ```

## Scientific limitations

The target labels for irrigation decisions are **approximate physics-rule proxies**, not verified farmer actions. Classification scores measure agreement with these assumed labels, not real-world irrigation safety, benefit or accuracy. Some model outputs in `hybrid_advisory_historical.csv` are in-sample; use only walk-forward predictions for evaluation. Moisture forecasts did not beat the persistence baseline overall. The soil and planting assumptions are not field-calibrated; this is a historical academic prototype, not an automatic irrigation controller.
