# Smart Irrigation

A decision-support system for irrigation. It forecasts soil moisture 24 to 72 hours ahead from a probe, weather and a rain forecast, and tells a farmer what to do: irrigate today, plan to irrigate in the next days, postpone because rain is coming, or do nothing. When the probe is unusable it says why and, for outages up to 10 days, gives a labelled low-confidence estimate from the weather.

**Status:** research prototype on two fields (2024-03 to 2025-10). It gives advice and never controls valves. There are no real irrigation logs, so the "dry" trigger is derived from the probe itself and has not been validated against farm outcomes. See the limitations in [docs/PIPELINE_REPORT.md](docs/PIPELINE_REPORT.md).

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# 1. cleaned feature tables (about 12 s) -> data/processed/
cd analysis && ../.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace analysis_advanced.ipynb && cd ..

# 2. train, evaluate, write models (about 11 s) -> artifacts/
PYTHONPATH=src .venv/bin/python -m irrigation_ml.pipeline

# 3. a recommendation from the command line
PYTHONPATH=src .venv/bin/python -m irrigation_ml.recommend --at 2025-07-28T06:00 --c-fn 5

# 4. the app: Farmer and Developer views, switched in the sidebar
.venv/bin/streamlit run app/app.py

# tests (67 checks, about 70 s)
.venv/bin/python -m unittest tests.test_irrigation_ml tests.test_correctness
```

`data/processed/` and `artifacts/` are not committed; steps 1 and 2 regenerate them from the committed dataset.

## Layout

| Path | Content |
|---|---|
| `src/` | preprocessing of the raw sources and their integration into `data/irrigation_ml_dataset.*` |
| `analysis/analysis_advanced.ipynb` | data-engineering notebook: de-accumulation of ERA5, ET0/ETc, anomaly rules, failure mask and cleaned target, drift tests, reference tables, FAO-56 label simulation, export |
| `src/irrigation_ml/` | the ML package: features, forecaster, dry-risk classifier, no-probe fallback, decision policy, `recommend()`, backtest |
| `app/` | Streamlit app |
| `tests/` | unit tests and an independent correctness audit |
| `docs/PIPELINE_REPORT.md` | detailed technical reference for every stage, with results and known issues |

## Results in one table (walk-forward, 2025-04 to 2025-10)

| Component | Result |
|---|---|
| Forecast vs "no change" | MAE +17% / +14% / +7% better at 24 / 48 / 72 h (RMSE about equal: large rain jumps are not predicted) |
| Dry-risk classifier | PR-AUC 0.74 / 0.75 / 0.80 against base rates of 0.03 / 0.04 / 0.06 |
| No-probe fallback | AUC 0.92 to 0.96 for outages up to 10 days; no skill beyond |
| Decision policy | warns before 14 of 35 real dry-downs with no false alarms; 19 get no warning |

## Known open issue

When the system flags dry soil 48 to 72 hours ahead, the probe in fact crosses the trigger within about a day, so "irrigate in 2 days" is optimistic. See section 10.1 of the report.
