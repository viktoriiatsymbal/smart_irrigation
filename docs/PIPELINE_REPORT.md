# Smart Irrigation Pipeline: Technical Reference

State of the system as of 2026-10-31 (branch `feat/irrigation-decision-system`, merged into `main`).

This document explains every part of the current pipeline: where the data comes from, how it is cleaned, how the models are built and evaluated, how a recommendation is produced, how the apps work, and how it is all tested. It is written from the code as it stands, and every number was measured in this repository.

**Contents**

1. [The system in one page](#1-the-system-in-one-page)
2. [Data sources and integration (`src/`)](#2-data-sources-and-integration-src)
3. [Data engineering notebook](#3-data-engineering-notebook-analysisanalysis_advancedipynb)
4. [The ML package (`src/irrigation_ml/`)](#4-the-ml-package-srcirrigation_ml)
5. [Evaluation protocol and current results](#5-evaluation-protocol-and-current-results)
6. [The applications (`app/`)](#6-the-applications-app)
7. [Quality assurance](#7-quality-assurance)
8. [Files and artifacts](#8-files-and-artifacts)
9. [Design decisions and history](#9-design-decisions-and-history)
10. [Known limitations and open issues](#10-known-limitations-and-open-issues)
11. [Operating guide](#11-operating-guide)
12. [Parameter reference](#12-parameter-reference)
13. [Appendix: schemas](#13-appendix-schemas)

---

## 1. The system in one page

**Goal.** Tell a farmer, for each field, whether to irrigate today, plan to irrigate in the next days, postpone because rain is coming, or do nothing, using soil-moisture probe data, weather, and a rain forecast. The system forecasts what the probe will read over the next 24, 48 and 72 hours and turns that into an action with reasons.

**Fields and data.** Two fields (`field_3334`, `field_3335`, about 46.0°N 20.5°E), hourly data from 2024-03 to 2025-10-13. `field_3334` grew sorghum in 2024 and corn in 2025; `field_3335` grew corn in both years. The soil-moisture probe reports a 0 to 100 index that is **not** calibrated to volumetric water content.

### 1.1 End-to-end flow

```
 RAW SOURCES                       src/ preprocessing                 integrated dataset
 ───────────                       ──────────────────                 ──────────────────
 Zenodo probe workbooks  ───────►  preprocess_zenodo.py  ──┐
 Open-Meteo hourly       ───────►  (same script)           │
 ERA5-Land NetCDF        ───────►  preprocess_copernicus.py├──► integrate_data.py ──► data/irrigation_ml_dataset.parquet/.csv
 SMAP L4 + static + 1 km ───────►  preprocess_nsidc.py     ┘     (27,346 rows × 131 columns, hourly, 2 fields)

 DATA-ENGINEERING NOTEBOOK (analysis/analysis_advanced.ipynb)
 ─────────────────────────────────────────────────────────────────────
 de-accumulate ERA5 → ET0/ETc → anomaly rules + Isolation Forest → failure mask → sm_clean target
 → drift tests → soil/crop reference tables → FAO-56 label simulation → leakage control → export
                                      │
                                      ▼
 data/processed/  hourly_model_table.parquet (27,346 × 98)   daily_decision_table.parquet (2,276 × 89)
                  decision_labels_daily.parquet (2,328 × 19) qa_drift.csv  ref_*.csv  pipeline_meta.json

 ML PACKAGE (src/irrigation_ml/)           python -m irrigation_ml.pipeline
 ───────────────────────────────
 data.py     full hourly grid, exact t+H targets
 features.py 63 causal features (45 sensor+trust, 15 weather, 3 rain-forecast)
        │
        ├─► forecast.py  LightGBM, change in moisture at +24/48/72 h (median + 10/90 % quantiles)  ─► forecast_models.joblib
        ├─► risk.py      LightGBM classifier  P(probe < field trigger within H h)                  ─► risk_models.joblib
        ├─► offline.py   no-probe fallback, trained on simulated outages                           ─► offline_models.joblib
        └─► decision.py  per-field trigger, conformal intervals, asymmetric-cost policy            ─► calibration.json

 recommend.py   one call: features → forecast + risk → decide() → (if probe unusable: diagnose + fallback) → recommendation
 backtest.py    replays the policy on out-of-fold predictions against what the probe really did

 app/app.py     Streamlit app, sidebar toggle:  Farmer view  |  Developer view
```

### 1.2 What each stage answers

| Stage | Question it answers | Main output |
|---|---|---|
| Integration | What do the probes, weather and satellites say, on one hourly clock? | `irrigation_ml_dataset.*` |
| Data engineering | Which probe readings can be trusted, and what are the physical quantities (ET0, ETc, rain, root-zone size)? | `hourly_model_table.parquet`, with `sm_clean` as the cleaned target |
| Forecaster | Where will the probe be in 1, 2 and 3 days, and how sure are we? | median and 80% band at +24/48/72 h |
| Risk classifier | How likely is it that the probe drops below this field's irrigation trigger within H hours? | `p_dry[H]` |
| Fallback | If the probe is down, can we still say something? | `p_dry[H]` from the last trusted reading and the weather |
| Decision policy | Given risk, rain and trust, what should the farmer do, by when, and how much? | action, due date, amount, reasons |
| Backtest | Would these decisions have warned before real dry-downs? | recall, false alarms, timing |
| Apps | How does a farmer (and a developer) see all this? | Streamlit pages |

### 1.3 Current headline results (all walk-forward, 2025-04 to 2025-10)

| Component | Result | Caveat |
|---|---|---|
| Forecast vs "no change" (persistence) | MAE **+17% / +14% / +7%** better at 24 / 48 / 72 h | RMSE about equal to persistence; large rain jumps are not predicted |
| Dry-risk classifier | PR-AUC 0.74 / 0.75 / 0.80 on hard cases (base rates 0.025 / 0.042 / 0.062) | only 91 / 151 / 223 positive rows, strongly autocorrelated |
| No-probe fallback | AUC 0.95 to 0.96 for outages up to 3 days, 0.92 to 0.94 for 3 to 10 days; no skill beyond 10 days | evaluated on few events |
| Decision policy | warns before 14 of 35 real dry-downs with no false alarms; 19 of 35 get no warning | 35 events on 12 days; timing of "plans" is too optimistic |

---

## 2. Data sources and integration (`src/`)

Four preprocessing scripts convert raw files to parquet; `integrate_data.py` merges them. They are committed in git; their raw inputs (`zenodo_data/`, `copernicus_data/`, `nsidc_data/`) are gitignored and not in this checkout, so this stage was **not re-run** during this work. Its output, `data/irrigation_ml_dataset.parquet/.csv`, is the starting point for everything below.

### 2.1 Probe data and Open-Meteo (`preprocess_zenodo.py`)

- **Input.** Excel workbooks with one sheet per station. Each station id is mapped to a field (`field_3334`, `field_3335`) with fixed coordinates.
- **Columns kept** (9 sensor columns, suffix `_sensor` unless noted): air temperature, air humidity, air pressure, dew point, precipitation (rain gauge), soil moisture, soil temperature, battery voltage and solar-panel voltage (the last two have no suffix), plus `crop` and `station_id`.
- **Cleaning at this stage.** Duplicate timestamps are flagged then dropped (first kept). Range flags: soil temperature outside [-20, 70], air temperature outside [-50, 60], humidity outside [0, 100], soil moisture outside [0, 100], negative precipitation; their union is `any_quality_flag`.
- **Hourly table.** Readings are floored to the hour. Continuous variables take the **median** within the hour, precipitation takes the **sum**, flags take the max. Each station-year is reindexed to a complete hourly grid, so missing hours exist as rows with `NaN`.
- **Open-Meteo** (`openmeteo_*` columns, 20 of them) is read from its own Excel files, including `openmeteo_et0_fao_evapotranspiration` and precipitation. It covers only part of the period (61% of hours for `field_3334`, 35% for `field_3335`).

### 2.2 ERA5-Land (`preprocess_copernicus.py`)

- NetCDF files, the grid point nearest to each field (`era5_grid_latitude/longitude` record which one).
- Converted columns: temperatures from Kelvin to Celsius (`*_c`), total precipitation to mm (`era5_tp_mm`), pressure to hPa, 10 m wind speed from the u and v components, relative humidity from temperature and dew point (Magnus formula).
- 35 ERA5 columns reach the integrated dataset: soil water (`swvl1`-`4`), soil temperature (`stl1`-`4`), leaf area, evaporation terms, radiation and runoff.
- **Important:** several ERA5-Land variables are *accumulated* since 01:00 each day. They are stored raw here and fixed in the notebook (section 3.2).

### 2.3 SMAP L4 (`preprocess_nsidc.py`)

- **Dynamic:** 3-hourly files, nearest cell to each field, 17 variables (surface, root-zone and profile soil moisture, six soil-temperature layers, evapotranspiration, precipitation, infiltration and runoff fluxes, leaf area, greenness, surface temperature and pressure).
- **Static:** 22 per-cell constants (porosity, wilting point, sand and clay fractions from the microwave model, layer depths and so on).
- **1 km daily** product as a separate table.
- SMAP cells are about 9 km wide, so they describe the neighbourhood, not the field.

### 2.4 Integration (`integrate_data.py`)

| Step | Detail |
|---|---|
| Base | the hourly probe table |
| Open-Meteo, ERA5 | left merge on (`field_id`, `timestamp`) |
| SMAP L4 | `merge_asof` per field, backward, tolerance 3 h; `smap_age_hours` records how old the SMAP sample is |
| SMAP static | merge on `field_id` |
| SMAP 1 km | merge on (`field_id`, date) |
| Metadata | `has_sensor_target`, `has_era5`, `has_smap_l4`, `has_openmeteo` |
| Output | `irrigation_ml_dataset.parquet` and `.csv` (27,346 rows × 131 columns), `data_quality_report.csv`, `data_dictionary.csv` |

Column provenance (from the data dictionary): ERA5-Land 35, SMAP static 24, SMAP L4 21, integration metadata 21, Open-Meteo 20, probe 9, SMAP 1 km 1.

Quality report: probe soil-moisture availability is 93.6% (`field_3334`) and 67.4% (`field_3335`); ERA5 and SMAP L4 are about 100% covered.

---

## 3. Data engineering notebook (`analysis/analysis_advanced.ipynb`)

The notebook is written in Ukrainian and organised as numbered "blocks" and "problems" (53 cells). It reads `../data/irrigation_ml_dataset.csv`, writes to `../data/processed/` (both relative to `analysis/`), and **runs end to end in about 12 seconds**. Below, its logic is described by stage, not by cell order. Configuration lives in one cell:

| Constant | Value | Meaning |
|---|---|---|
| `H_LIST` | 24, 48, 72 | forecast horizons (hours) |
| `DEAD_SM_THRESHOLD` | **5** | reading below this = dead probe (was 10, see 9.4) |
| `BUFFER_H` | 48 | hours around a failure that are also distrusted |
| `POST_GAP_H` | 24 | hours after a data gap that are distrusted |
| `SENSOR_SHIFT_H` | 0 | time shift applied to probe columns (see 3.3) |
| `FORECAST_MODE` | `"noisy"` | rain forecast = ERA5 truth plus simulated error |
| `DECISION_HOUR` | 6 | hour of the daily decision snapshot |
| `TEST_START`, `TEST_END` | 2025-04-01, 2025-11-01 | test window |
| `GAP_H` | 72 | purge gap between train and test |
| `MIN_COVERAGE` | 0.4 | minimum non-missing share per field for a feature to be kept |
| `EVENT_DATE` | 2024-10-16 | step change found by CUSUM on `field_3334` |
| `RANDOM_SEED` | 42 | |

### 3.1 Inputs and hygiene

Sorted by (`field_id`, `timestamp`), duplicates dropped, a `day` column added, and the rule columns coerced to booleans. Asserts check that the earlier cells have run.

### 3.2 ERA5 de-accumulation, ET0 and ETc

ERA5-Land accumulations reset at **01:00** each day. The notebook detects accumulated columns by checking that sharp drops occur at 01:00 (at least 80% of resets at 01:00 and at least 50 resets), forces three confirmed ones (`era5_tp_mm`, `era5_pev`, `era5_ssrd`), and converts them to hourly increments:

- the increment is the difference to the previous hour, except at 01:00 where it is the value itself;
- differences are not taken across gaps (the result is `NaN`);
- unit conversions: precipitation stays in mm (`p_era5_mm_h`), radiation to W/m² (`era5_ssrd_wm2`, divide by 3600), runoff and evaporation from m to mm (×1000), potential evaporation `pet_era5_mm_h` (sign flipped because ERA5 evaporation is negative).

**ET0** (reference evapotranspiration, mm/h) is Open-Meteo's FAO-56 value where available, otherwise `k_pet × pet_era5_mm_h`. The scaling factor `k_pet` (0.592) is fitted by least squares on the overlap, **using only data before the test start**, to avoid leakage. **ETc** = Kc × ET0.

**Rain.** `p_era5_mm_h` is the single rain source for features and labels. The tipping-bucket gauge is unreliable (next paragraph) and is not used as a feature.

| Rain gauge vs ERA5 (daily) | `field_3334` | `field_3335` |
|---|---|---|
| Spearman correlation | 0.49 | 0.41 |
| Gauge total / ERA5 total | 0.74 | 0.46 |
| Rainy days (ERA5 ≥ 5 mm) missed by the gauge | 31% | 48% |

### 3.3 Time alignment

The notebook scans lags between probe air temperature and ERA5 2 m temperature, and between the solar-panel voltage and ERA5 radiation. Both pairs on both fields give a best lag of **-2 hours**. `SENSOR_SHIFT_H` was left at 0, so **probe and weather data are not shifted**. The effect on 24 to 72 hour forecasts should be small, but it is an open decision (section 10).

### 3.4 Anomaly detection

Per-field **rule flags** (`flag_field`), all computed per hour:

| Flag | Rule | Counts as probe failure? |
|---|---|---|
| `rng_<sensor>` | outside physical range: soil moisture [0, 100], soil temperature [-15, 50], air temperature [-35, 50], humidity [0, 100], pressure [900, 1080] | soil moisture: yes |
| `sm_flat_stuck` | identical reading for **at least 48 hours**, below 99.9 | **yes** |
| `sm_sat_plateau` | identical reading for at least 72 hours at 99.9 or above | no (informational: saturation) |
| `sm_spike` | **drops only**: reading more than 5 robust deviations below its 25-hour centred median (Hampel, minimum deviation 1.0) | yes |
| `sm_jump_up_no_rain` | rise of more than 5 in one hour while none of the three rain sources (gauge, ERA5, Open-Meteo) shows over 0.2 mm in the last 6 h | **no**: informational (a likely irrigation event) |
| `sm_drop_fast` | fall of more than 10 in one hour | yes |
| `st_flat` | soil temperature identical for at least 24 h | no (feeds soil-temperature cleaning) |
| `st_vs_era5`, `at_vs_era5`, `ap_vs_era5`, `ah_vs_era5` | residual against ERA5 beyond 5 robust deviations in a 7-day window | no (soil-temperature cleaning, trust feature) |
| `pr_negative`, `pr_dry_gauge` | negative gauge reading; gauge dry while ERA5 shows over 3 mm in 3 h | no |

**`anom_sm_any`** is the union of the failure flags only: out-of-range, stuck, downward spike, fast drop (and only where a reading exists). Rises are never in it.

**Isolation Forest** (`anom_iforest`): per field, on the features soil moisture, soil temperature, air temperature, air humidity and the hourly changes of soil moisture and soil temperature. It is trained on the **first 70% of each field's rows**, with the threshold set at the worst 1% of training scores (300 trees, seed 0), and applied to all rows. It is a statistical outlier flag, **not** part of the failure mask. It was never confirmed that its flags correspond to faults; its rate rises 3 to 8 times after the training window ends (drift in conditions), so some flags are probably false alarms.

**Battery link.** The notebook tabulates anomaly rate against battery voltage (bins 0 to 3.6, 3.6 to 3.8, 3.8 to 4.0, above 4.0). Degradation below about 3.8 to 4.0 V is the reason battery voltage is kept as a trust feature.

### 3.5 The failure mask and the cleaned target `sm_clean`

```
sm_dead      = probe reading < 5                       (real dead-probe readings sit at ~1.2)
sm_post_gap  = the 24 h after any gap longer than 3 h
sm_bad_core  = anom_sm_any OR sm_dead
sm_bad       = (sm_bad_core dilated by ±48 h, centred) OR sm_post_gap
sm_clean     = probe reading where NOT sm_bad, else NaN
is_sm_valid  = sm_clean is not NaN
```

Design principle (set by the project owner and now implemented): **only failures are rejected. A rise is rain or irrigation and is never rejected.** The mask is therefore: dead probe, stuck value, drops and impossible values. The ±48 h buffer exists because a failure drags unstable readings with it.

**Footprint of each rule** (hours masked *only* because of that rule, before the last two changes; `field_3334` / `field_3335`): dead probe 1,071 / 517; stuck value 792 / 688; spike (then two-sided) 108 / 1; jump-up (then in the mask) 2 / 97; fast drop 1 / 4. The stuck and dead rules dominate. Evidence for keeping the stuck rule is in 9.5.

**Result.** `sm_clean` is valid on 72.9% of hours for `field_3334` and 52.9% for `field_3335`. The target is **never interpolated**: gaps stay `NaN`, and anything that needs a future value skips rows where it is missing.

**Non-causal versus causal.** `sm_bad` and `is_sm_valid` use a *centred* window, so they use future hours. That is fine for deciding which targets are valid, but it would leak if used as a model input. For inputs the export adds causal versions (3.12): `trust_bad_past_48h` (the same failure core, but looking only backwards) and `sm_obs_causal` (the probe reading where that causal flag is off).

### 3.6 Soil temperature

`st_flat` runs of 24 hours or more are judged against ERA5 soil temperature: if ERA5 moved at least 2 °C while the probe stood still, it is a stuck probe (`st_stuck`); otherwise it is plausible (cold or frozen soil in winter). `st_clean` is the probe soil temperature with `st_stuck`, range violations and `st_vs_era5` removed (valid for 74.8% / 59.8% of hours).

### 3.7 Sensor drift

Weekly medians of the difference "probe minus reference" are regressed on time with an annual sine and cosine (to absorb seasonality), with HAC (Newey-West, 4 lags) standard errors. The slope is reported per year with a 95% interval, and **drift is confirmed** when the interval excludes zero **and** the slope exceeds a tolerance (0.5 °C for temperatures, 1 hPa for pressure, 2 percentage points for humidity, 0.05 in rank space for soil moisture). A **control** pair, ERA5 against SMAP, is included: if the references drift against each other, drift against them is not blamed on the probe.

| Field | Variable | Slope per year | Confirmed |
|---|---|---|---|
| `field_3334` | soil temperature vs ERA5 | +1.45 °C | yes |
| `field_3334` | humidity vs ERA5 | -3.6 points | yes |
| both | control: ERA5 vs SMAP | about +0.08 to +0.10 (rank) | yes (so part of the apparent drift is in the references) |
| both | soil moisture vs ERA5 / SMAP | not significant | no |

A step change on `field_3334` around 2024-10-16 was also tested (Mann-Whitney). Results go to `qa_drift.csv` (14 rows).

### 3.8 Reference tables

Hand-entered, FAO-56-based, **not field-calibrated**:

- **`ref_soil.csv`**: field capacity (FC), permanent wilting point (PWP), porosity, saturated conductivity and basic infiltration rate for 12 USDA textures (FAO-56 Table 19, a few approximated). `taw_mm_per_m` = (FC - PWP) × 1000.
- **`ref_fields_soil.csv`**: field texture from the SMAP cell's sand and clay fractions via the USDA triangle. Both fields come out as **loam** (FC 0.25, PWP 0.12, infiltration 13 mm/h). This describes a roughly 9 km cell, not the field.
- **`ref_crop.csv`**: Kc (initial, mid, end), root depth, depletion fraction p (0.55 for corn and sorghum) and a seasonal ETc range per crop (8 crops).
- **`ref_irrigation.csv`**: system efficiency (drip 0.90, sprinkler 0.75, surface 0.60), wetted fraction and typical rate.
- **`ref_soil_sensitivity.csv`**: how the irrigation trigger moves under plausible FC and PWP values (the trigger in m³/m³ ranges from 0.173 to 0.228).
- **Phenology calendar** (month-day: sowing, development, mid, late, end): corn 04-20, 05-20, 07-01, 08-20, 09-30; sorghum 05-10, 06-05, 07-10, 08-25, 10-05. Outside the season Kc is `NaN`, so ETc and the labels are `NaN`.

Derived per hour: `kc` (piecewise-linear between the calendar dates), `etc_mm_h = kc × et0_mm_h`, root depth `zr_m` (grows from 0.3 m to the crop's maximum by the start of mid-season), `season_frac` (0 to 1 over the season) and `taw_mm = (FC - PWP) × zr_m × 1000`. The crop *name* is deliberately not meant as a model input in the DE design, because on `field_3334` it equals the year (see 10.4).

### 3.9 Targets and the simulated rain forecast

- `y_sm_h{24,48,72}`: `sm_clean` exactly H hours later (valid only if both now and then are valid), plus `y_dsm_h*` (the change) and `y_sat_h*` (saturated at the target time). The ML package rebuilds its own exact targets from `sm_clean` (4.1), so these are kept mainly for the DE diagnostics.
- `sm_saturated`: `sm_clean` at or above 99.9, with `NaN` where the probe is invalid. Saturation censors the probe at 100.
- **Target dynamics** (persistence RMSE against the standard deviation of the target) are logged to show how little the series moves: persistence RMSE at 24 / 48 / 72 h is 5.7 / 8.0 / 9.6 on `field_3334` and 2.9 / 4.5 / 6.0 on `field_3335`, against target standard deviations of about 19 and 9.4.
- **`rain_fc_next{24,48,72}h`**: the rain in the next H hours. In `noisy` mode this is the **ERA5 truth with simulated forecast error**: log-normal noise whose spread grows with lead time (sigma = 0.35 + 0.15 × (lead/24 - 1)), a miss probability (10% + 5% per extra day) for rain events of at least 2 mm, and a 3% chance of a false alarm of a few mm on dry hours. The true values are kept in `rain_true_next*h` for evaluation and are **blacklisted as features**. **This is not a real forecast**; see section 10.

### 3.10 The FAO-56 label simulation

Because there are no farmer irrigation logs, the notebook simulates what an "ideal manager" would do and uses that as the decision label. For each field, day and irrigation system (sprinkler, drip):

1. Daily ETc and effective rain are summed from hourly values (a day needs all 24 hours, otherwise it is `NaN`). Effective rain is 0.8 × rain when the day's rain exceeds 2 mm, otherwise 0.
2. Total available water `TAW = (FC - PWP) × zr × 1000` (mm), and the readily available water `RAW = clip(p + 0.04 × (5 - ETc_day), 0.1, 0.8) × p_factor × TAW`, with `p_factor` 1.0 for sprinkler and 0.8 for drip.
3. The simulated manager starts each in-season stretch at field capacity, adds ETc minus effective rain to the depletion `D` each day, and irrigates back to field capacity whenever `D` reaches RAW. These synthetic events are `irr_synth_mm`.
4. For each horizon of 1, 2 and 3 days, the label is: `irrigate` if the depletion projected with the (noisy) rain forecast reaches RAW; `postpone` if it would not with the forecast rain but would without any rain; otherwise `no_action`. `gross_need_mm_h*` is the projected depletion divided by system efficiency.

The daily table takes the 06:00 snapshot of the hourly features and joins these labels. About 50 `irrigate` days per system are labelled (roughly 8% of labelled days); there is one `postpone`.

**These labels are a simulation, not observations.** They depend only on ERA5 rain, ET and assumed soil, **never on the probe**. On `field_3335` the probe does not distinguish the "irrigate" days from the rest (89.6 against 89.2), and on days with a good probe the labels' "irrigate" flags were 6% precise against a 4.5% base rate. The production decision therefore does **not** use them (section 4.5 explains the replacement). They remain in `data/processed/` as a benchmark.

### 3.11 Leakage control and split design

- A **blacklist** removes everything that must not be a feature: identifiers, the raw probe columns (replaced by cleaned ones), raw accumulated ERA5 columns, Kelvin and pascal duplicates, simulator outputs (`D_morn_mm`, `raw_mm`, `irr_synth_mm`), all `y_*`, `act_*`, `gross_need_*`, `rain_true_*`, `smap_static_*` and `era5_grid_*` (they identify the field), the QC flags, and, in the DE feature list, battery and solar voltage (as a "failure-mode proxy"). Features also need at least 40% coverage in the worst field. 61 hourly and 61 daily features survive.
- **Walk-forward folds:** `[2025-04-01, 2025-06-15)`, `[2025-06-15, 2025-08-15)`, `[2025-08-15, 2025-11-01)`, each training on everything before it minus a gap of 72 hours. The ML package uses the same folds (`config.py`).
- A **leave-one-field-out** split is defined but not used by the ML package (two fields are too few for transfer to mean much).
- **Sensitivity of the trigger** (block 13): under plausible FC and PWP values, the FAO trigger moves between 0.173 and 0.228 m³/m³, which is why a trigger derived from the probe's own behaviour was preferred (4.5).

### 3.12 Export

Written by the final cell, after assertions (dead-probe readings never remain valid, battery and solar never in the DE feature list, targets are `NaN` wherever `sm_clean` is, and so on).

| File | Content |
|---|---|
| `hourly_model_table.parquet` | 27,346 rows × 98 columns: keys, 61 features, **trust columns**, targets and `is_sm_valid` |
| `daily_decision_table.parquet` | 2,276 rows × 89 columns: 06:00 snapshot per field, day and system, features, trust columns, labels and needs |
| `decision_labels_daily.parquet` | 2,328 rows × 19 columns: labels only, including `*_oracle` and `*_noisy` variants |
| `qa_drift.csv`, `ref_*.csv`, `pipeline_meta.json` | drift report, reference tables, configuration, feature lists, blacklist, run log and assumptions |

**Trust columns added to the hourly table** (they were missing in the first version of the export, which broke a project requirement): `battery_voltage`, `solar_panel_voltage`, `anom_iforest`, `anom_sm_any`, `sm_dead`, `sm_post_gap`, all rule flags, `trust_bad_past_48h` and `sm_obs_causal`. The daily table gets the same plus `batt_min_24h` and trailing 24-hour shares of the anomaly flags and of usable readings. They are listed in `pipeline_meta.json` as `trust_features_hourly` / `trust_features_daily`.

---

## 4. The ML package (`src/irrigation_ml/`)

About 920 lines in nine modules. `config.py` holds the shared constants: horizons `(24, 48, 72)`, the three walk-forward folds, quantiles `(0.1, 0.9)`, saturation 99.9, and the paths.

### 4.1 Loading and targets (`data.py`)

`load_hourly()` reads the hourly parquet and **reindexes each field to a complete hourly grid** (27,868 rows; 522 gap rows are added: 24 for `field_3334`, 498 for `field_3335`). The new rows hold `NaN`. This matters because shifting by H *rows* is only correct if every row is one hour. Reindexing never invents target values (a test checks this). `add_targets()` creates `y24`, `y48`, `y72` as `sm_clean` shifted by exactly H hours.

### 4.2 Features (`features.py`)

All features are computed **at issue time t** from the past, per field. The central input is `sm_obs_causal` (the probe reading, `NaN` where the causal failure flag is on), called `sm0` below. **63 features** feed the forecaster: 45 sensor and trust features, 15 weather features and 3 rain-forecast features. The risk classifier uses the same 63 plus `gap_to_trigger` (64). A fourth group (8 reanalysis and satellite columns) exists for ablations and is **not used** in production.

| Group | Count | Features |
|---|---|---|
| Probe state | 20 | `sm0`; changes over 1, 3, 6, 12, 24, 48, 72 and 168 h (`dsm_*`); min, max and standard deviation over 24 h and 72 h; `sat_now`, `sat_share_72` (share of the last 72 h at saturation); `hours_since_obs`; `obs_share_24` (share of the last 24 h with a trusted reading); soil temperature `st_clean` |
| Calendar and crop | 9 | `hour_sin/cos`, `doy_sin/cos`, `crop_code`, `zr_m`, `kc`, `taw_mm`, `season_frac` |
| Trust | 16 | current-hour flags `trust_*` (battery, solar, anomaly flags, dead, post-gap, causal bad-past-48 h, range, stuck, spike, jump-up, fast drop, soil-temperature flat, dry gauge), `batt_min_24` (lowest battery in 24 h), `anom_iforest_24` (share of the last 24 h flagged by the Isolation Forest) |
| Weather (`wx`) | 15 | rain sums over 1, 6, 24, 72 and 168 h, `hours_since_rain`, air temperature, humidity, pressure, radiation, wind, `et0_mm_h`, `etc_mm_h`, ET0 sums over 24 h and 72 h |
| Rain forecast (`fc`) | 3 | `rain_fc_next24h`, `rain_fc_next48h`, `rain_fc_next72h` (**simulated**, see 3.9) |
| Reanalysis (`retro`, unused) | 8 | ERA5 and SMAP soil water, ERA5 soil and air temperature: not available live (ERA5-Land has about 5 days of latency, SMAP L4 about 2.5) |

Two naming quirks: trust features read `trust_<name>`, so the causal flag appears as `trust_trust_bad_past_48h`; and `crop_code` (sorghum 0, corn 1) is a model input even though the DE design advises against the crop name (10.4).

Rain history uses **ERA5** as a stand-in for a live weather feed, because the rain gauge is unreliable.

### 4.3 The forecaster (`forecast.py`)

**What it predicts.** For each horizon H, the *change* `y(t+H) - sm0`, not the level. Three models per horizon: the median (L1 loss) and the 10th and 90th percentiles (quantile loss). The point forecast is `sm0 + median change`, clipped to [0, 100].

**Why the change and the L1 loss.** The probe sits near saturation (median about 92); a typical 24-hour change is under 1 point, but rain jumps reach +57. An L2 loss chases those jumps, predicts positive changes on average and **lost to persistence at every horizon**. The L1 loss predicts the median change and wins.

**Model.** LightGBM, deliberately small: 100 trees, learning rate 0.03, 4 leaves, minimum 500 samples per leaf, row subsample 0.8, column subsample 0.7, L2 regularisation 5, seed 0. Larger models overfit and lost at 72 hours; the setting sits on a plateau (selected on the same walk-forward folds, so the margin is slightly optimistic).

**Drift adaptation.** Training rows are weighted by recency with a **half-life of 120 days** (`0.5 ** (age_days / 120)` relative to the end of the training window). In the ablation this helped every horizon.

**Missing data.** Rows without a target or without `sm0` are skipped; no interpolation. LightGBM takes `NaN` features natively.

**Walk-forward.** For each horizon and fold: train on rows with `timestamp < fold_start - H hours` (the *purge*: the last training target ends before the test window), test on rows inside the fold. Persistence (`sm0`) is the baseline in every table. Test rows are skipped if there are fewer than 50 test rows or 500 training rows.

**Interval calibration (conformal).** The raw 10-90 intervals cover only 71% / 67% / 62% of outcomes against a nominal 80%. Conformalised quantile regression computes, per horizon, the 80th percentile of `max(q10 - y, y - q90)` over the out-of-fold predictions and widens both ends by it. Offsets: **0.32 / 1.10 / 2.04** points. They are calibrated on the same out-of-fold data used for evaluation.

**Final models.** `train_final` refits on all labelled history, with the same weighting (reference = last timestamp), and saves the models with their feature order.

**Ablation (`python -m irrigation_ml.forecast --ablate`).** Compares groups (probe only, +weather, +rain forecast, +reanalysis) and half-lives (none, 90, 180 days). The conclusions: recency weighting helps; reanalysis columns do not help enough to justify their latency. The saved `forecast_ablation.csv` is from an **earlier run** (L1 loss but larger trees, before the data fixes) and should be used for relative comparison only.

### 4.4 The trigger: what counts as "dry" (`decision.py: sensor_thresholds`)

The probe is not calibrated, and the FAO labels do not follow it, so the irrigation trigger is derived **from each field's own probe behaviour**, mirroring FAO-56's depletion logic:

```
fc_s    = 75th percentile of the field's in-season trusted readings     (its "wet plateau")
pwp_s   =  2nd percentile                                               (its "driest" level)
trigger = fc_s - p_mad × (fc_s - pwp_s)                                 p_mad = 0.55 for corn and sorghum
reliable = (fc_s - pwp_s) >= 15                                         (the probe moves enough to define a trigger)
```

"In season" means `kc` is defined. Current triggers (all data): **54.4 for `field_3334`, 70.9 for `field_3335`**; both reliable. For the **walk-forward** evaluation the trigger is computed only from data before 2025-04-01 (so the test period cannot define "dry"): 54.7 and 87.3; the latter is flagged unreliable because the probe range before the test was only about 8 points.

This is an **assumption**: no agronomist has validated it, and there are no irrigation records to check it against.

### 4.5 The dry-risk classifier (`risk.py`)

**Label.** For each hour t and horizon H: `stress_H = 1` if the lowest trusted reading in `(t, t+H]` is below the field trigger, `0` if not, `NaN` if fewer than `H/4` trusted readings exist in the window. This is the **realised sensor outcome**, not a simulation. (Implemented as a rolling minimum evaluated at t+H and shifted back.)

**Model.** LightGBM classifier, 150 trees, 6 leaves, minimum 100 per leaf, same sub-sampling and regularisation, with the same recency weights (half-life 120 days). Inputs: the 63 forecaster features plus `gap_to_trigger = sm0 - trigger`. One model per horizon.

**Evaluation.** Same walk-forward and purge. Because "already below the trigger" is trivially predictable, the key subset is **anticipation rows**: probe currently *above* the trigger (`gap_to_trigger > 0`). Metric: PR-AUC (area under precision-recall) against the base rate.

| Horizon | Anticipation rows | Positives | Base rate | PR-AUC |
|---|---|---|---|---|
| 24 h | 3,626 | 91 | 0.025 | 0.74 |
| 48 h | 3,602 | 151 | 0.042 | 0.75 |
| 72 h | 3,578 | 223 | 0.062 | 0.80 |

Recall and precision at 72 h depend on the alert threshold: at 0.25, 40% / 92%; at 0.17, 42% / 90%; at 0.05, 66% / 79%; at 0.02, 92% / 39%. The classifier is sharp (bimodal), and its low-probability region (0.02 to 0.05) is under-confident. Rows are hourly and autocorrelated, so the number of **independent** events is much smaller than the row count.

### 4.6 The no-probe fallback (`offline.py`)

**Purpose.** On `field_3334` about a quarter of days have no usable probe reading, in runs of weeks (communication gaps, a probe that reports near-zero at low battery, stuck values). Without a fallback the farmer gets nothing for those days.

**Why not the DE weather labels.** They have no relationship with what a working probe later shows (6% precision against a 4.5% base rate; AUC 0.41 for the modelled depletion).

**Idea.** Predict the same dry-risk, but from the *last trusted reading, its age and the weather since*, as if the probe had gone silent. To train it, outages are **simulated** on the history: for every hour t an age `a` is drawn log-uniformly between 6 and 600 hours; the model sees only the reading from `t - a` plus weather up to t; the label is what the (hidden) probe did after t.

**23 features:** `age_h`, `last_sm` (reading at t - a), `gap_last` (that reading minus the trigger), `last_trend_24` (change in the 24 hours before it), `rain_since` and `et0_since` (sums over the outage), rain sums over 24, 72, 168 h, ET0 sums over 24 and 72 h, `hours_since_rain`, mean air temperature over 24 h, `crop_code`, season sine and cosine, `kc`, `zr_m`, `taw_mm`, `season_frac`, and the three rain forecasts. The training frame (`build`) and the single-row inference (`infer_features`) use the same arithmetic, and a test proves they agree.

**Model.** Same classifier settings as 4.5, one per horizon, same walk-forward and recency weights.

**Skill (AUC, walk-forward).** "Hard" means the hidden probe is still above the trigger.

| Outage age | AUC all cases (24 / 48 / 72 h) | Naive "last reading below trigger" (24 h) | AUC hard cases (24 / 48 / 72 h) |
|---|---|---|---|
| up to 3 days | 0.96 / 0.95 / 0.95 | 0.92 | 0.93 / 0.88 / 0.90 |
| 3 to 10 days | 0.94 / 0.92 / 0.92 | 0.79 | 0.80 / 0.80 / 0.79 |
| 10 to 25 days | 0.77 / 0.78 / 0.79 | 0.57 | 0.36 / 0.43 / 0.57 |

**Use.** The estimate is offered for outages **up to 10 days** (240 h). Beyond that, hard-case skill is no better than chance, so the system says no estimate is possible. The estimate is always low confidence, never gives an amount (the soil deficit is unknown) and never recommends postponing. Hard-case event counts are small (12 to 48 per bucket at 24 h).

### 4.7 The decision policy (`decision.py`)

#### Policy parameters (`Policy` dataclass)

| Parameter | Default | Meaning |
|---|---|---|
| `c_fn` | 3.0 (the apps and CLI pass **5.0**; the farmer slider uses 10 / 5 / 2) | cost of missing a needed irrigation |
| `c_fp` | 1.0 | cost of an unnecessary irrigation |
| `tau` | `c_fp / (c_fp + c_fn)` | risk threshold: 0.25 at c_fn 3, **0.167 at 5**, 0.091 at 10, 0.333 at 2 |
| `rain_effective_mm` | 5.0 | forecast rain that justifies postponing |
| `battery_min_v` | 3.8 | below this the probe is not trusted |
| `min_obs_share` | 0.5 | minimum share of trusted readings in the last 24 h |
| `max_event_mm` | 40.0 | largest single application (gross) |
| `offline_max_age_h` | 240 | longest outage for which a no-probe estimate is offered |

The threshold `tau` is the standard decision-theory result: act when the probability of the costly event exceeds `c_fp / (c_fp + c_fn)`.

#### `decide()`: the order of checks

Inputs: current probe reading, `p_dry[24/48/72]` from the classifier, the field trigger, the rain forecasts, trust information (lowest battery in 24 h, share of trusted readings, an "unusual readings" flag), the forecast medians, the field's total available water, and the irrigation system.

1. **No usable reading → `check_sensor`.** The reading is missing, fewer than half of the last 24 hours are trusted, or the lowest battery in 24 h is below 3.8 V. (The caller may then try the fallback, 4.8.) *An anomaly-detector flag alone does not block advice*; it adds a note and lowers confidence to "low".
2. **Risk is made non-decreasing with horizon.** The three classifiers are fitted separately, but "dry within 24 h" is contained in "dry within 72 h", so each risk is raised to the maximum of the earlier ones. A "dry right now" entry (`p[0]`) is 1 if the reading is already below the trigger.
3. **No risk → `no_action`** if the maximum risk through 72 h is below `tau`.
4. **Rain recovers → `postpone`.** If a forecast rain of at least 5 mm falls within some horizon `rain_h`, the forecast **median** at `rain_h` is back above the trigger, and the risk did not start before `rain_h`. *Rain alone is never enough* (ERA5-type rain often misses the field; an early version that postponed on rain alone was wrong three times out of three).
5. **Imminent risk → `irrigate`.** Dry now, or `p[24]` at least `tau`. The amount is computed (below).
6. **Later risk → `irrigate_soon`** (the forward-looking plan). Risk first reaches `tau` at 48 or 72 hours. `dry_expected_h` is that horizon; `irrigate_in_h = dry_expected_h - 24` is the advised deadline (one step earlier). The amount uses the forecast median moisture at the dry window.

**Amount.** `deficit = clip((fc_s - moisture) / (fc_s - pwp_s), 0, 1) × TAW / efficiency` (efficiency 0.75 sprinkler, 0.90 drip; TAW from the notebook, about 170 mm for loam and a mature root zone), then `amount = min(deficit, 40 mm)`. The deficit is reported separately and `capped` says whether the cap applied. A single application near 20 to 50 mm is realistic; the old uncapped figure was a median of 159 mm. **The 40 mm cap and the 0.55 depletion fraction are assumptions.**

**Confidence.** "low" if there are notes (narrow probe range, unusual readings); otherwise "high" when the maximum risk is below `tau/2` or above `2 × tau`, else "medium".

**Known weakness of step 6.** In the backtest the real crossing came within 24 hours in 94% of cases flagged at 48 hours and 57% flagged at 72 hours, so the advised deadline is still late (10.1). The docstring in the code says so.

#### `decide_offline()`

Used when the probe is unusable but the outage is up to 10 days. Risks are made non-decreasing; `p[24]` at least `tau` → `irrigate` (due now); risk reaches `tau` by 48 or 72 h → `irrigate_soon`; otherwise `no_action`. Confidence is always low, there is no amount, there is never a postpone, and a note says that rain, even if forecast, cannot be confirmed to reach the soil.

#### Other helpers

`calibrate_intervals` (conformal offsets, 4.3); `prob_below` (interpolates a CDF through the three quantiles; kept for tests and experiments, no longer used by `decide()`); `gross_need_mm`; `fmt_duration` ("less than an hour", "5 hours", "8.3 days").

### 4.8 Orchestration (`recommend.py`)

`recommend(at, policy, system, prepared)` returns one recommendation per field at issue time `at` (default: the last timestamp). For each field:

1. Take the feature row at or before `at`; load the trigger from `calibration.json`.
2. If there is a reading: compute the **forecast** for each horizon (median and quantiles, each added to `sm0` and clipped), widen the interval by the conformal offset and clip to [0, 100]; compute the **risk** with the classifier.
3. Read the **rain forecast** from the row, build the **trust** information (lowest battery in 24 h, trusted share, `anom_iforest_24 > 0.25` as "unusual"), and read TAW.
4. Call `decide()`.
5. **If the result is `check_sensor`:** drop any forecast (never show a forecast built on an unusable reading), call `diagnose_sensor()`, and, if the last trusted reading is at most 10 days old, build the fallback features (`infer_features`), get risks from the offline models and call `decide_offline()`. If no estimate is possible, the reason is stated.
6. Add `sensor_issue` (a plain-language reason), `irrigate_by` (the due date, for plans) and a note if the latest data is more than 6 hours old (which also lowers confidence to low).

**`diagnose_sensor()`** looks at the last 24 hours and names the cause in this priority: low battery (with the voltage); no data for at least half the hours; near-zero readings (dead probe) for at least 30%; a stuck value for at least 30%; otherwise "most readings were rejected". It also gives the time since the last trusted reading, where "trusted" means a causal valid reading **and** battery at least 3.8 V (or unknown). If a trusted reading exists right now but the previous 24 h were flagged, it says the sensor has just recovered.

**Output modes** (`mode`): `sensor` (normal), `weather_only` (fallback estimate) and `none` (no estimate).

**Output fields:** `field_id`, `issued_at`, `crop`, `soil_moisture`, `trigger`, `forecast` ({H: (low, median, high)}), `rain_forecast_mm`, `action`, `irrigate_in_h`, `dry_expected_h`, `irrigate_by`, `confidence`, `reasons`, `p_dry`, `amount_mm`, `deficit_mm`, `capped`, `mode`, `sensor_issue`.

In production, the table passed in would be rebuilt from live probe telemetry and a real weather-forecast feed; everything after that is unchanged. At present it replays the historical file.

### 4.9 Backtest (`backtest.py`)

Replays the policy at **00, 06, 12 and 18 h** on **out-of-fold** forecasts and risks (so the models never saw those rows), with conformally widened intervals and the walk-forward trigger, against what the probe actually did:

- `stress72`: the lowest trusted reading within 72 hours is below the trigger.
- `first_cross_h`: hours until the probe first crosses the trigger within 72 hours.
- `naive`: the probe is already below the trigger now.

Scores are recall, precision and false alarms for several flag definitions. **Anticipation cases** (probe above the trigger now) are the ones that matter, because "already dry" is trivial. At the default cost (`c_fn` 5): 736 cases, 552 anticipation cases, **35 real dry-downs on 12 distinct days**. Advice before those 35 events: irrigate today 5, plan to irrigate 9, no action 19, check the sensor 2. All 14 warnings were followed by a real dry-down (no false alarms). Raising the cost of a miss to `c_fn` 10 catches 18 of 35 (2 false alarms), and to 20 catches 28 of 35 (6 false alarms). "Postpone" never fired, so that branch is untested on real data.

### 4.10 Training and artifacts (`pipeline.py`)

`python -m irrigation_ml.pipeline` (about 11 seconds) does, in order:

1. Build features. Forecaster: walk-forward (out-of-fold predictions and metrics saved), then final models.
2. Risk classifier: walk-forward with the trigger from pre-test data, then final models with the trigger from all data.
3. Fallback: walk-forward (with its metrics) and final models.
4. Write `calibration.json`: conformal offsets, final thresholds, walk-forward thresholds, data end.

---

## 5. Evaluation protocol and current results

### 5.1 Protocol

- **Time-based only.** No random splits. Three expanding-window folds in 2025.
- **Purge.** Training targets must end before the test window starts (`timestamp < fold_start - H`).
- **Baselines.** Persistence for the forecaster; "last reading below trigger" for the fallback; "probe below trigger now" for the policy.
- **Metrics.** MAE, RMSE and skill (`1 - model/persistence`) for forecasts; coverage for intervals; PR-AUC, ROC-AUC, recall and precision at stated thresholds for classifiers; recall and false alarms against realised outcomes for the policy.
- **Segments.** Forecast skill is reported for all rows, for rows with a change above 3, and for dry (forecast rain under 0.1 mm) and wet (above 2 mm) periods.
- **Selection caveat.** Model settings were chosen on these same folds, and the conformal offsets are calibrated on the same predictions, so reported margins are slightly optimistic.

### 5.2 Forecast results

| Horizon | n | MAE persistence | MAE model | MAE skill | RMSE skill | Raw 10-90 coverage |
|---|---|---|---|---|---|---|
| 24 h | 4,685 | 2.07 | 1.72 | **+17.0%** | +1.4% | 71% |
| 48 h | 4,565 | 3.93 | 3.39 | **+13.7%** | +0.4% | 67% |
| 72 h | 4,445 | 5.59 | 5.17 | **+7.4%** | -1.5% | 62% |

By segment (MAE skill at 24 / 48 / 72 h): dry periods +25% / +22% / +18%; periods with forecast rain above 2 mm +5% / +5% / +1%; rows with a change above 3 points +11% / +11% / +9%.

**The RMSE caveat.** In squared-error terms the model is no better than persistence. Large rain-driven jumps dominate RMSE, and neither method predicts them. The clearest case: 21 to 22 August 2025, a very dry soil (about 12) rose to 95 within a day, an error of about 83 for both, although 5 mm of rain was forecast. This is a real model weakness (10.2).

### 5.3 Risk and fallback results

See the tables in 4.5 and 4.6.

### 5.4 Policy results and the timing finding

See 4.9. Timing, from 60 hourly situations on only 6 distinct days where the plan branch applies:

| Model flagged | Probe crossed within 24 h | Within 48 h | Median time to crossing |
|---|---|---|---|
| 48 h | 94% | 100% | 13 h |
| 72 h | 57% | 100% | 18 h |

A "plan" therefore means "dry soil within about a day"; the model's 2 to 3 day horizon is optimistic, and the advised deadline (flag minus 24 h) is later than the real crossing in 91% (48 h flags) and 100% (72 h flags) of cases.

### 5.5 Behaviour over the whole record (daily 06:00, both fields)

| Field | Days | Normal (probe) | Weather-only estimate | No estimate |
|---|---|---|---|---|
| `field_3334` | 584 | 435 (74%) | 87 | 62 |
| `field_3335` | 576 | 295 (51%) | 91 | 190 |

---

## 6. The applications (`app/`)

One Streamlit program with a toggle (`streamlit run app/app.py`). English only, no emojis. The record ends 2025-10-13, so the app runs in **demo mode**, replaying historical dates.

### 6.1 Structure

| File | Role |
|---|---|
| `app.py` | page setup, the **Farmer / Developer** toggle in the sidebar; the farmer page sits in a narrow centre column; the farmer's sidebar choices survive a visit to the developer view |
| `common.py` | cached loaders: `prepared()` (features, cached resource), `recommendations()` (cached data), `raw_sensor()` (raw probe readings for the chart) |
| `farmer_view.py` | the farmer page |
| `dev_view.py` | the developer console |

### 6.2 Farmer view

Sidebar: field, date with **Previous day / Next day** buttons (disabled at the ends of the data), sprinkler or drip, and a caution slider (cautious `c_fn` 10, balanced 5, save water 2).

The page, top to bottom:

1. A **sensor banner** when there is a problem, with the specific cause ("The sensor is reporting near-zero values, which means a probe fault. The last trusted reading was 8.3 days ago.") or a note for unusual readings.
2. A large **action card**, with the action, a one-line explanation and a confidence label. Colours: **yellow with dark text** = irrigate today; **blue** = irrigate tomorrow or in N days (with "Irrigate by Wednesday 30 Jul"); green = no irrigation needed; purple = postpone; red = check the sensor. For a fallback estimate the card says "Estimate without the sensor, confidence: low".
3. Three numbers: soil moisture now (hidden when the reading is unusable), the level below which to irrigate, and the suggested amount in mm (with m³ per hectare underneath; a note explains a capped amount: "The soil is about 171 mm short. Apply 40 mm now and repeat on the following days.").
4. A **chart**: the last 14 days of trusted readings (grey line), untrusted readings (orange dots, not used for advice), the forecast (blue dashed line with the 80% band) when there is one, and the irrigation trigger (red dotted line), with a "how to read this" note listing only what is shown. If the sensor sent nothing for 14 days, the page says so.
5. A **3-day plan table** (today and the next three days): suggested action per day, expected moisture with its range, risk of drying out and cumulative forecast rain. Without the probe it is labelled weather-based and omits the moisture column.
6. A **sensor health** panel (battery, age of the last reading, 24-hour coverage) and a disclaimer: advice, not control.

### 6.3 Developer view

Eight tabs: **Overview** (headline metrics, triggers, offsets), **Data & trust** (coverage and anomaly rates, battery against anomaly rate, the masked series, the drift table highlighted with a theme-safe translucent red), **Forecast** (metrics against persistence, ablation, out-of-fold plots, error by size of change), **Dry-risk classifier** (PR curve, threshold table, calibration, and the fallback metrics), **Decision backtest** (cost slider, scored policies, action counts, case table), **Explain** (feature importance by gain, share of trust features), **Case inspector** (a recommendation for any date and hour, with the full output and the feature row) and **Pipeline** (artifact and input listings, DE trust features, buttons to run tests and to retrain).

### 6.4 Caching pitfall

Streamlit caches `recommendations()` on `(date, cost, system)` only. When model code or artifacts change while the server runs, old results (for example without a newly added field) can be served and cause a `KeyError`. Restart Streamlit or press `C` and clear the cache. A cache key that includes file modification times was proposed but not adopted.

---

## 7. Quality assurance

**67 automated checks, all passing** (`python -m unittest tests.test_irrigation_ml tests.test_correctness`, about 70 seconds).

### 7.1 `tests/test_irrigation_ml.py` (23, fast)

Decision rules: untrusted probe never advises; wet soil gives no action; dry now irrigates with an amount; the amount is capped and the deficit reported; an anomaly flag alone lowers confidence but still advises; rain alone never postpones; a recovering forecast postpones; imminent risk is "irrigate today", not a plan; risk first at 48 or 72 h gives the right plan and deadline; the planned amount uses the forecast at the dry window; a higher cost lowers the threshold. Fallback rules: no estimate beyond 10 days or with an unknown age, always low confidence, never an amount, never postpone. Wording: durations are human ("less than an hour", never "0.0 days"). Regression: the `KeyError` case of 2025-08-10. A **future-tampering test** that every numeric column after a cutoff can change without altering earlier features.

### 7.2 `tests/test_correctness.py` (44, independent audit)

It recomputes things a different way from the production code and compares:

| Area | What is checked |
|---|---|
| Data | complete unique hourly grid; no invented target values; `sm_clean` equals the raw reading where kept and respects the dead threshold from the metadata; targets are exactly t+H (merge-based, on 1,200 random rows); the causal mask equals a brute-force 48 h window |
| Failure mask | the mask equals exactly dead, stuck, drop and out-of-range, recomputed from raw readings; every spike flag lies below its neighbourhood; the 6 July 2024 rain onset is kept; drops, stuck and dead hours stay excluded as targets and inputs |
| Dead-probe rule | a smooth dip to 9.9 on 20 August 2025 keeps its 96 surrounding hours and the app in sensor mode; about 2,400 real failure hours stay excluded |
| Features | lags, rolling sums and min/max match brute force on 600+ rows; no target-like columns are inputs; hours since observation are consistent |
| Forecast evaluation | out-of-fold rows carry true targets; persistence equals the causal reading; rows lie in their fold; reported metrics equal an independent recomputation; **spy**: every model's training set ends before the test window |
| Intervals | conformal coverage at least 79%, never narrower than raw; quantile crossing under 1% |
| Risk | labels match a brute-force window; the trigger ignores post-cutoff data; spy on training sets; probabilities valid and better than chance (PR-AUC) and a constant (Brier) |
| Fallback | training frame equals brute force for given ages; inference row equals training row; ignores the future; spy on training sets; beats the naive rule up to 10 days |
| Decision properties | random inputs: a higher cost never drops a warning; no usable reading never advises; amounts non-negative, correct and within the cap; risk non-decreasing with horizon; due dates and windows consistent over a daily sweep of the whole record (which must contain plans) |
| Diagnosis | the reason names the real cause (communication gap, dead probe, low battery with a reading present), no "0.0 days" anywhere |
| Backtest | realised outcomes and crossing times match brute force on 60 cases; cases lie in the test period |
| Recommendation | advice at t unchanged when later data change (except the rain-forecast feed); final models load with the right feature order; outputs in range over about 50 dates |

### 7.3 The audit itself was tested

Deliberate bugs were injected one at a time to confirm a test fails: target shifted by an hour, purge removed, label window shifted, trigger using post-cutoff data, look-ahead feature, monotonicity removed, an untrusted probe still advising, amount ignoring efficiency, amount cap removed, anomaly flag blocking again, 10-day limit removed, offline estimate giving an amount, an off-by-one window, offline purge removed, outage age looking forward, a wrong diagnosis branch, battery rule dropped from the age calculation, safety margin removed, plan reverted, planned amount ignoring the forecast, rain recovery dropped, due date off by a day, crossing time from the wrong origin. All were caught; two were first missed because a test was weak and were strengthened. Lessons recorded: when swapping text of equal length, Python may reuse stale bytecode (run mutations with `PYTHONDONTWRITEBYTECODE=1`); the failure-mask and dead-rule tests were also shown to fail on the previous data.

### 7.4 UI checks

Streamlit's headless `AppTest` rendered the app for both fields every second day across the whole record (about 850 screens, with several caution levels) with no exceptions, covering every state: probe working, unusual readings, weather-only estimate, no estimate, plans. The apps also start as real servers and answer health checks with HTTP 200. **They have not been viewed in a browser**; layout, colours and chart appearance are unchecked.

### 7.5 What the tests do not cover

- Whether the dry trigger, the 0.55 depletion fraction, the 40 mm cap or the cost ratio are agronomically right (needs real irrigation outcomes).
- The DE notebook's internal logic beyond reproducing `sm_clean` and the mask checks (its drift tests, Isolation Forest and FAO simulation were not independently audited).
- **Boolean flags that look ahead** (10.3): the tampering tests alter numeric columns only.
- Behaviour with live data.
- Rendering in a browser.

---

## 8. Files and artifacts

### 8.1 Repository layout

```
smart_irrigation/
├── README.md                     overview and quick start
├── requirements.txt
├── docs/PIPELINE_REPORT.md       this document
├── src/                          preprocess_*.py, integrate_data.py (raw sources -> integrated dataset)
├── src/irrigation_ml/            the ML package (+ its README.md)
├── data/
│   ├── irrigation_ml_dataset.*   integrated dataset (committed)
│   └── processed/                [gitignored] DE export (parquet), qa_drift, ref_*, pipeline_meta.json
├── analysis/                     analysis_advanced.ipynb (the DE notebook), download.py
├── app/                          Streamlit app (Farmer / Developer)
├── tests/                        two test modules
└── artifacts/                    [gitignored] models and evaluation outputs, regenerated by the pipeline
```

`data/processed/` and `artifacts/` are not committed: both are regenerated in seconds (11.2 and 11.3) from the committed dataset. The earlier prototype (`Smart_Irrigation_ML_ONLY`, later pushed as `smart_irrigation_ml/`) and the older exploratory notebooks were removed as superseded; they remain in the git history.

### 8.2 Artifacts (`artifacts/`)

| File | Size | Produced by | Used by |
|---|---|---|---|
| `forecast_models.joblib` | 568 KB | `forecast.train_final` | `recommend` (median and quantile models, feature order) |
| `risk_models.joblib` | 325 KB | `risk.train_final` | `recommend` |
| `offline_models.joblib` | 363 KB | `offline.train_final` | `recommend` (fallback) |
| `calibration.json` | 0.7 KB | `pipeline` | `recommend`: conformal offsets, final and walk-forward triggers, data end |
| `forecast_oof.parquet` | 467 KB | `forecast.walk_forward` | metrics, backtest, developer view |
| `forecast_metrics.csv` | 2 KB | `forecast.summarize` | developer view |
| `risk_oof.parquet` | 74 KB | `risk.walk_forward` | backtest, developer view |
| `offline_oof.parquet`, `offline_metrics.csv` | 133 KB, 2 KB | `offline` | developer view, tests |
| `forecast_ablation.csv` | 11 KB | `forecast --ablate` (stale, see 4.3) | developer view |

All of `artifacts/` is regenerated by `python -m irrigation_ml.pipeline`.

---

## 9. Design decisions and history

The system was reached through a sequence of problems; each decision is recorded with its evidence.

1. **The DE export dropped the trust indicators.** `battery_voltage` and the anomaly flags were blacklisted, contradicting the requirement that they inform the model. Fixed by adding trust columns to both tables, with **causal** versions for inputs (`trust_bad_past_48h`, `sm_obs_causal`), because the notebook's own validity mask uses a centred window and would leak.
2. **The first forecaster lost to persistence.** Cause: L2 loss on a heavy-tailed target. Fix: predict the *change*, with L1 loss, small trees, and recency weighting; walk-forward gains became real. A grid showed a plateau of heavily regularised models.
3. **The FAO labels are circular and ignore the probe.** They depend only on ERA5 rain, ET and assumed soil. Replaced, for the decision, by a classifier trained on the realised probe outcome, with a per-field trigger derived from the probe.
4. **The decision was too rain-trusting.** An early "postpone" fired on forecast rain alone three times and the probe stayed dry each time. Postpone now needs the moisture forecast itself to recover.
5. **Risk could fall with horizon.** Separate classifiers gave "24 h 60%, 72 h 30%" in 8% of rows. Fixed by enforcing a non-decreasing risk inside `decide()`.
6. **Irrigation amounts were unrealistic** (median 159 mm). Capped at 40 mm per application with the deficit reported separately.
7. **Probe outages gave no advice** (about a quarter of days on `field_3334`). Added the no-probe fallback (4.6) after showing that the DE weather labels carry no signal.
8. **Dead-probe threshold 10 → 5.** A legitimately dry reading of 9.9 for one hour discarded about 45 plausible hours. 23 of the 25 episodes below 10 are real failures sitting at about 1.2; two are smooth dips with a healthy battery. After the change: 126 hours regained, none lost, no values changed.
9. **Failure mask: rises are never rejected.** The two-sided outlier test and the jump-up rule were rejecting rain onsets (6 July 2024: 90.6, 93.7, 100 with 11.8 mm of rain). Spike now flags drops only; jump-up is informational. 162 hours regained, none lost. The days after 7 July 2024 are a genuine communication gap, not a validation issue.
10. **The stuck-value rule was kept** after examining all 14 episodes: 5 are at dead-probe level; 9 are exactly identical readings for 50 to 244 hours, mostly in cold winter soil, but one is a summer hang (`field_3335`, 13 to 15 August 2024, 85.9 at 29 °C). A probe frozen at "wet" is the dangerous direction because it hides a drying field.
11. **A forward-looking plan became first-class** (`irrigate_soon`), with a due date and a day-by-day table, replacing a "no action, re-check tomorrow" note. Its timing later proved optimistic (5.4), and the original claim that a one-day margin "caught every plan" was wrong and was corrected.
12. **Anomaly-detector flags no longer block advice**; they lower confidence. The Isolation Forest is a statistical outlier flag, not a failure detector.

---

## 10. Known limitations and open issues

### 10.1 Plan timing is optimistic (open decision)

When the model flags dry soil at 48 or 72 hours, the probe crosses within about a day. The warnings are right (all 14 in the backtest), but "irrigate in 2 days" is not supported by the data. Options: (a) say "irrigate tomorrow" for every plan and keep the model's window as information (simplest, matches the evidence); (b) add finer horizons (6, 12, 18, 24 h) for a real time-to-dry estimate and retest; (c) keep the wording with a "may come sooner" label. Recommended: (a) now, (b) as the real fix. This needs the project owner's decision.

### 10.2 Large rain jumps are not anticipated

From a very dry soil the forecaster does not predict the big rise after rain, so RMSE skill is about zero and the system can advise irrigating just before rain arrives. A dedicated rain-response model (rain amount and timing, antecedent moisture) is the suggested fix.

### 10.3 Some trust flags use hindsight (found while writing this report)

Verified on the data:

- `sm_flat_stuck` (and `st_flat`) are based on the **length of the whole run** of identical readings, so the flag is already on at the **first** hour of a run that will last 48 hours or more. For example, on `field_3335` the reading 96.5 begins at 2025-02-14 21:00 and is flagged from that hour. About **658 of 1,343** stuck-flagged hours (2.4% of all hours) lie within the first 47 hours of a run and could not have been known at the time. `st_flat` flags 3,366 hours.
- The downward-spike test uses a **centred** 25-hour window, so it looks up to 12 hours ahead (16 hours flagged).
- These flags feed `anom_sm_any`, therefore `trust_bad_past_48h` and `sm_obs_causal`, which are advertised as causal. The causal versions look only backwards **in time** but inherit the hindsight of their inputs.
- **Consequence:** in historical evaluation the system "knows" in advance that a stuck run is coming and treats those hours as unusable, so they drop out of scoring; live, the probe would be trusted until 48 hours of flatness are seen. Effect on the reported metrics is probably small (mostly cold-season hours) but is not quantified.
- The tampering tests change **numeric** columns only, so they could not detect this. A fix: compute the stuck and flat flags causally (flag an hour only once it has repeated for the last 48 or 24 hours) while keeping the retrospective mask for deciding which *targets* are valid.

### 10.4 Crop is confounded with year, and `crop_code` is a model input

On `field_3334`, sorghum is 2024 and corn is 2025. The DE design says not to feed the crop name; the ML package nevertheless uses `crop_code`. Because all test folds are in 2025 and corn, the effect is probably limited, but the model can use it as a year indicator in training. Physical crop descriptors (`kc`, `zr_m`, `taw_mm`, `season_frac`) are also included.

### 10.5 Other limits

| Area | Limit |
|---|---|
| No ground truth | There are no irrigation logs. "Dry" is a trigger derived from the probe; the depletion fraction 0.55, the 40 mm cap and the cost ratio are assumptions. None has been checked against farm outcomes. |
| Rain forecast is simulated | `noisy` ERA5. The forecast-rain features and the postpone branch are optimistic until a real archived forecast feed (for example Open-Meteo historical forecasts) replaces them. |
| Small data | Two fields, under two seasons, 35 independent dry-down events on 12 days in the backtest; transfer to new fields or years is untested. |
| Selection effects | Settings and conformal offsets were fitted on the evaluation folds; the Isolation Forest is trained on the first 70% of each field. |
| Probe | The 0 to 100 index is uncalibrated; `field_3335` barely moves at times (its pre-test range was 8 points); soil texture comes from a 9 km cell and TAW from assumed loam and root depth. |
| Time alignment | A consistent -2 hour lag between probe and ERA5 was found but not applied (3.3). |
| Rain gauge | Unreliable (misses 31% and 48% of rainy days), not used as a feature. |
| Latency | Reanalysis and satellite columns are not used in production features; the live system needs a real weather feed for rain history, ET0 and the forecast. |
| Drift | Confirmed for soil temperature and humidity on `field_3334`; part of it is in the reference products (the control also drifts). Recency weighting is the only adaptation; there is no automatic retraining. |
| Sensor outages | Beyond 10 days offline the system gives no estimate. |
| Fallback | Evaluated on few events; the trigger is the same unvalidated assumption. |
| Intervals | Under-cover in rain events and large changes (coverage 11% to 24% on rows with a change above 3 points). |
| Housekeeping | `forecast_ablation.csv` is stale (an earlier run; regenerate with `forecast --ablate`); the DE notebook prints and plots are in Ukrainian. |

---

## 11. Operating guide

### 11.1 Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

### 11.2 Regenerate the DE tables

Run `analysis/analysis_advanced.ipynb` from the `analysis/` directory (about 12 s). It reads `../data/irrigation_ml_dataset.csv` and writes `../data/processed/`:

```bash
cd analysis && ../.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace analysis_advanced.ipynb --ExecutePreprocessor.timeout=1800
```

Re-running reproduces the committed tables exactly (verified: identical values in all three parquet tables and in `pipeline_meta.json`).

### 11.3 Train and evaluate (about 11 s)

```bash
PYTHONPATH=src .venv/bin/python -m irrigation_ml.pipeline
PYTHONPATH=src .venv/bin/python -m irrigation_ml.forecast --ablate   # optional
PYTHONPATH=src .venv/bin/python -m irrigation_ml.backtest            # optional
```

### 11.4 Get a recommendation

```bash
PYTHONPATH=src .venv/bin/python -m irrigation_ml.recommend --at 2025-07-28T06:00 --c-fn 5 --system sprinkler [--json]
```

### 11.5 Run the apps and the tests

```bash
.venv/bin/streamlit run app/app.py
.venv/bin/python -m unittest tests.test_irrigation_ml tests.test_correctness
```

### 11.6 After changing anything

1. If the DE rules change, regenerate the tables, then retrain; metrics and triggers will move.
2. Restart Streamlit (stale cache, 6.4).
3. Re-run the tests; for new logic, add a test and inject a bug to prove it can fail (use `PYTHONDONTWRITEBYTECODE=1`).
4. Update the status and pipeline reports, since numbers are quoted in them.

### 11.7 Moving towards live use

Replace, and keep everything downstream: (1) live ingestion of probe telemetry into the hourly table, with the failure flags computed **causally** (10.3); (2) a real weather source for rain history, ET0 and the forecast `rain_fc_next*h`, with their issue times; (3) retraining on a schedule with an alarm when skill against persistence drops; (4) delivery of the daily advice by message as well as the app. Do not actuate valves: every output is advice with reasons.

---

## 12. Parameter reference

| Parameter | Value | Where | Notes |
|---|---|---|---|
| Horizons | 24, 48, 72 h | `config.py` | |
| Walk-forward folds | 2025-04-01 to 06-15, 06-15 to 08-15, 08-15 to 11-01 | `config.py` | purge = H |
| Quantiles | 0.1, 0.9 | `config.py` | |
| Saturation | 99.9 | `config.py` | |
| Forecaster | 100 trees, lr 0.03, 4 leaves, min leaf 500, subsample 0.8, colsample 0.7, L2 reg 5, L1 loss | `forecast.py` | |
| Recency half-life | 120 days | `pipeline.py`, `risk.py`, `offline.py` | |
| Conformal level | 0.8 | `decision.py` | offsets 0.32 / 1.10 / 2.04 |
| Risk and fallback classifiers | 150 trees, lr 0.03, 6 leaves, min leaf 100, subsample 0.8, colsample 0.7, L2 reg 5 | `risk.py`, `offline.py` | |
| Trigger anchors | 75th and 2nd percentile; p 0.55; minimum range 15 | `decision.py` | |
| Walk-forward trigger cutoff | 2025-04-01 | `pipeline.py` | |
| Fallback outage ages | 6 to 600 h, log-uniform | `offline.py` | offered up to 240 h |
| Cost, threshold | `c_fn` 5 (app default), `c_fp` 1, `tau` = 1/(1+c_fn) | `decision.py` | farmer slider 10 / 5 / 2 |
| Postpone rain | 5 mm | `decision.py` | |
| Battery floor | 3.8 V | `decision.py` | |
| Minimum trusted share | 0.5 in 24 h | `decision.py` | |
| Unusual-readings flag | Isolation Forest share in last 24 h above 0.25 | `recommend.py`, `backtest.py` | |
| Amount cap | 40 mm gross | `decision.py` | |
| Efficiency | sprinkler 0.75, drip 0.90 | `decision.py` | |
| Backtest decision hours | 0, 6, 12, 18 | `backtest.py` | |
| DE: dead threshold | 5 | notebook | was 10 |
| DE: buffer, post-gap | 48 h, 24 h | notebook | |
| DE: stuck run | 48 h at below 99.9 | notebook | |
| DE: fast drop | more than 10 per hour | notebook | |
| DE: Isolation Forest | 300 trees, train on first 70%, threshold at worst 1% | notebook | |
| DE: effective rain | 0.8 × rain when daily rain above 2 mm | notebook | labels only |
| DE: rain-forecast noise | sigma 0.35 + 0.15 × (lead/24 - 1), miss 10% + 5%/day, false alarm 3% | notebook | |

---

## 13. Appendix: schemas

### 13.1 `hourly_model_table.parquet` (27,346 × 98)

- **Keys:** `field_id`, `timestamp`, `crop`.
- **Probe and weather:** air temperature, humidity, pressure, dew point; ERA5 soil water and temperature, leaf area, evaporation, wind, humidity; SMAP soil moisture, soil temperature layers, fluxes, leaf area, greenness; `p_era5_mm_h`, `pet_era5_mm_h`, `et0_mm_h`, `kc`, `etc_mm_h`, `era5_*_mm_h`, `era5_ssrd_wm2`, `rain_mm_h`, `st_clean`.
- **Target and masks:** `sm_clean` (cleaned probe reading), `is_sm_valid`, `sm_saturated`, `y_sm_h*`, `y_dsm_h*`, `y_sat_h*`.
- **Crop and soil descriptors:** `zr_m`, `season_frac`, `taw_mm`.
- **Simulated rain forecast:** `rain_fc_next24h/48h/72h`.
- **Trust columns:** `battery_voltage`, `solar_panel_voltage`, `anom_iforest`, `anom_sm_any`, `sm_dead`, `sm_post_gap`, `rng_*`, `sm_flat_stuck`, `sm_sat_plateau`, `sm_spike`, `sm_jump_up_no_rain`, `sm_drop_fast`, `st_flat`, `st_vs_era5`, `at_vs_era5`, `ap_vs_era5`, `ah_vs_era5`, `pr_negative`, `pr_dry_gauge`, `trust_bad_past_48h`, `sm_obs_causal`.

### 13.2 `daily_decision_table.parquet` (2,276 × 89) and `decision_labels_daily.parquet` (2,328 × 19)

One row per field, day and system (`sprinkler`, `drip`) at the 06:00 snapshot: features, trust columns (plus `batt_min_24h`, `anom_iforest_share_24h`, `anom_sm_any_share_24h`, `sm_obs_share_24h`), and the simulated labels `act_h24/48/72` (`irrigate`, `postpone`, `no_action`; `_oracle` and `_noisy` variants) with `gross_need_mm_h*`, `D_morn_mm`, `raw_mm`, `taw_mm`, `irr_synth_mm`.

### 13.3 `calibration.json`

```json
{
  "interval_offsets": {"24": 0.32, "48": 1.10, "72": 2.04},
  "thresholds": {"field_3334": {"fc_s": ..., "pwp_s": ..., "trigger": 54.4, "p_mad": 0.55, "reliable": true},
                 "field_3335": {"...": "...", "trigger": 70.9}},
  "thresholds_walkforward": {"field_3334": {"trigger": 54.7}, "field_3335": {"trigger": 87.3, "reliable": false}},
  "data_end": "2025-10-13 00:00:00"
}
```

### 13.4 A recommendation (output of `recommend()`)

```python
{
 "field_id": "field_3334", "issued_at": "2025-07-28 06:00:00", "crop": "corn",
 "soil_moisture": 64.0, "trigger": 54.4,
 "forecast": {24: (59.2, 60.4, 63.9), 48: (55.0, 58.6, 64.9), 72: (50.2, 56.1, 65.9)},   # (low, median, high), conformal
 "rain_forecast_mm": {24: 0.4, 48: 1.7, 72: 2.7},
 "action": "irrigate_soon",              # irrigate | irrigate_soon | postpone | no_action | check_sensor
 "irrigate_in_h": 48, "dry_expected_h": 72, "irrigate_by": "2025-07-30",
 "confidence": "high", "reasons": ["forecast: soil moisture is expected to reach ..."],
 "p_dry": {24: 0.01, 48: 0.11, 72: 0.96, 0: 0.0},
 "amount_mm": 40.0, "deficit_mm": 116.6, "capped": True,
 "mode": "sensor",                       # sensor | weather_only | none
 "sensor_issue": None
}
```
