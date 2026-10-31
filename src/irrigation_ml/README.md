# irrigation_ml: forecasting + irrigation decision support

Replaces the earlier ML prototype (removed; see git history). It trains on the DE feature stores in `data/processed/`
(`analysis/analysis_advanced.ipynb` exports the trust indicators; see `trust_features_*` in `pipeline_meta.json`).
Full reference: `docs/PIPELINE_REPORT.md`.

```bash
.venv/bin/python -m unittest tests.test_irrigation_ml tests.test_correctness   # 67 checks incl. an independent audit
PYTHONPATH=src .venv/bin/python -m irrigation_ml.pipeline         # train + walk-forward + artifacts/  (~20 s)
PYTHONPATH=src .venv/bin/python -m irrigation_ml.forecast --ablate   # feature-group / drift-weight ablation
PYTHONPATH=src .venv/bin/python -m irrigation_ml.backtest         # decision backtest vs realised sensor outcome
PYTHONPATH=src .venv/bin/python -m irrigation_ml.recommend --at 2025-05-05T06:00 [--c-fn 5] [--system drip] [--json]
```

## Pieces
| module | role |
|---|---|
| `features.py` | causal features at issue time; groups: sensor+trust, weather, forecast-rain, retro (ERA5/SMAP) |
| `forecast.py` | LightGBM on the 24/48/72 h **change** with an L1 objective, recency weights (half-life 120 d), quantiles; walk-forward |
| `risk.py` | classifier: P(probe < field trigger within H h), trained on the realised outcome |
| `decision.py` | per-field sensor-calibrated trigger, conformal intervals, asymmetric-cost policy, irrigation amount (capped at 40 mm per application, deficit reported), no-probe decision rule; the forward-looking `irrigate_soon` plan (deadline one day before the model's dry window) |
| `offline.py` | no-probe fallback: P(dry) from the last trusted reading, its age and the weather since; trained on simulated outages; offered up to 10 days offline |
| `backtest.py` | scores decisions against what the probe actually did |
| `recommend.py` | `recommend(field, time)`: action + reasons + forecast band + amount |

## Measured results (walk-forward, 3 folds 2025-04..10, train purged by H, persistence always the baseline)
Forecast MAE skill vs persistence: **+21% (24 h), +17% (48 h), +9% (72 h)**; RMSE skill +7/+6/+1%. Never worse than
persistence in any fold. Dry (no-rain) periods gain most (+26/+23/+19% MAE); in rain-forecast periods the gain is small
(+10/+10/+2%). 10-90 intervals cover 72/65/65% raw; conformal offsets widen them to the 80% target (calibrated on the same OOF data).

Dry-risk classifier, rows where the probe is still above the trigger ("anticipation"; PR-AUC vs base rate):
24 h 0.55 (0.02), 48 h 0.63 (0.04), 72 h 0.66 (0.06). At 72 h: threshold 0.05 -> recall 53% / precision 75%;
0.25 -> 39% / 73%; 0.02 -> 88% / 26%. **Most sudden dry-downs are still missed**; choose the cost ratio (`--c-fn`) accordingly.

What changed vs the old package: L2 loss on a heavy-tailed target (rain jumps up to +57) biased forecasts and lost to
persistence at every horizon; L1 + small trees + recency weighting wins. Smaller models won on a plateau
(selected on the same folds, so the margin is slightly optimistic).

## Limits, read before trusting it
- **No ground-truth irrigation data.** "Dry" = probe below a per-field trigger derived from the probe's own in-season
  plateaus with FAO-56 RAW logic (p=0.55). It is an assumption; no agronomist has validated it.
- **`field_3335` probe barely moves** (narrow range); its trigger is flagged approximate -> confidence "low".
- Only 2 fields, <2 seasons; transfer to new fields/years is untested. Folds 2-3 have few labelled rows.
- **Rain forecast = ERA5 truth + simulated noise** (DE `noisy` mode), so forecast-rain features and the
  "postpone" branch are optimistic. Postpone additionally requires the model to predict recovery; in the backtest
  it never fired. Needs a real archived forecast feed (e.g. Open-Meteo historical forecasts) to validate.
- Reanalysis/satellite columns are *not* used in the production feature set (latency).
- Probe scale is 0-100 uncalibrated; irrigation amounts are approximate (`taw_mm` from assumed loam/root depth).
- Advisory only: no valve control.
