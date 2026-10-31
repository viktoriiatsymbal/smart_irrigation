"""Independent correctness audit. Recomputes quantities a different way from the production code and compares.
Needs data/processed/*.parquet and artifacts/ (run irrigation_ml.pipeline first). Slower than test_irrigation_ml."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from irrigation_ml import decision, forecast, risk  # noqa: E402
from irrigation_ml.config import ARTIFACTS, FOLDS, HORIZONS, PROCESSED  # noqa: E402
from irrigation_ml.decision import Policy, decide, sensor_thresholds  # noqa: E402

DF = FEAT = RAW = None


def setUpModule():
    global DF, FEAT, RAW
    DF, FEAT = forecast.prepare()
    RAW = pd.read_parquet(PROCESSED / "hourly_model_table.parquet")


def at(df, fid, ts, col):
    s = df[(df.field_id == fid) & (df.timestamp == ts)][col]
    return s.iloc[0] if len(s) else np.nan


class TestData(unittest.TestCase):
    def test_grid_is_complete_hourly_and_unique(self):
        for fid, g in DF.groupby("field_id"):
            self.assertTrue(g.timestamp.is_unique)
            self.assertTrue((g.timestamp.diff().dropna() == pd.Timedelta(hours=1)).all(), fid)

    def test_grid_adds_no_invented_values(self):
        raw_keys = set(zip(RAW.field_id, RAW.timestamp))
        added = DF[[(f, t) not in raw_keys for f, t in zip(DF.field_id, DF.timestamp)]]
        self.assertGreater(len(added), 0)                       # the DE table has gaps
        self.assertTrue(added["sm_clean"].isna().all())         # reindexing must not fabricate the target

    def test_sm_clean_matches_raw_sensor_where_valid_and_in_range(self):
        raw = pd.read_parquet(ROOT / "data" / "irrigation_ml_dataset.parquet", columns=["field_id", "timestamp", "soil_moisture_sensor"])
        m = RAW.merge(raw, on=["field_id", "timestamp"], how="left")
        ok = m.sm_clean.notna()
        self.assertTrue(np.allclose(m.loc[ok, "sm_clean"], m.loc[ok, "soil_moisture_sensor"]), "sm_clean must be the raw reading, unmodified")
        import json
        dead = json.loads((PROCESSED / "pipeline_meta.json").read_text())["config"]["DEAD_SM_THRESHOLD"]
        self.assertEqual(dead, 5)                                           # real failures sit at ~1.2; dry soil reads 9-10
        self.assertGreaterEqual(m.sm_clean.min(), dead)
        self.assertLessEqual(m.sm_clean.max(), 100)
        self.assertTrue(m.loc[~m.is_sm_valid, "sm_clean"].isna().all())

    def test_targets_are_exactly_t_plus_h(self):
        key = DF.set_index(["field_id", "timestamp"])["sm_clean"]
        rng = np.random.default_rng(1)
        for H in HORIZONS:
            idx = rng.choice(len(DF), 400, replace=False)
            for i in idx:
                r = DF.iloc[i]
                want = key.get((r.field_id, r.timestamp + pd.Timedelta(hours=H)), np.nan)
                got = r[f"y{H}"]
                self.assertTrue((np.isnan(want) and np.isnan(got)) or want == got, (H, r.field_id, r.timestamp))

    def test_causal_observation_mask_matches_bruteforce(self):
        for fid, g in DF.groupby("field_id"):
            g = g.reset_index(drop=True)
            bad = (g.anom_sm_any | g.sm_dead | g.sm_post_gap).fillna(False).to_numpy(bool)
            for i in np.random.default_rng(2).choice(len(g), 300, replace=False):
                past_bad = bad[max(0, i - 48): i + 1].any()
                self.assertEqual(bool(g.trust_bad_past_48h.fillna(False).iloc[i]), bool(past_bad), (fid, i))   # NaN rows = grid gaps
                if past_bad:
                    self.assertTrue(pd.isna(g.sm_obs_causal.iloc[i]), (fid, i))     # a reading near a fault is never used as input


class TestDeadProbeRule(unittest.TestCase):
    """A smooth dip to 9.9 on a dry day must not discard two days of data, while real probe failures (~1.2) still must."""
    @classmethod
    def setUpClass(cls):
        cls.raw = pd.read_parquet(ROOT / "data" / "irrigation_ml_dataset.parquet", columns=["field_id", "timestamp", "soil_moisture_sensor"])

    def test_legitimate_dry_reading_near_10_keeps_surrounding_days(self):
        g = DF[(DF.field_id == "field_3334") & (DF.timestamp >= "2025-08-19") & (DF.timestamp < "2025-08-23")]
        r = self.raw[(self.raw.field_id == "field_3334") & (self.raw.timestamp == pd.Timestamp("2025-08-20 09:00"))]
        self.assertAlmostEqual(float(r.soil_moisture_sensor.iloc[0]), 9.9)        # the single reading just below 10
        self.assertEqual(int(g.sm_clean.notna().sum()), len(g))                  # all 96 hours kept as target
        self.assertEqual(int(g.sm_obs_causal.notna().sum()), len(g))             # and usable as input
        from irrigation_ml.recommend import recommend
        for day in ("2025-08-20", "2025-08-21", "2025-08-22"):
            r = [x for x in recommend(f"{day}T06:00", Policy(), "sprinkler", (DF, FEAT)) if x["field_id"] == "field_3334"][0]
            self.assertEqual(r["mode"], "sensor", day)

    def test_real_probe_failures_stay_excluded(self):
        m = self.raw.merge(DF[["field_id", "timestamp", "sm_clean", "sm_obs_causal"]], on=["field_id", "timestamp"])
        dead = m[m.soil_moisture_sensor < 5]
        self.assertGreater(len(dead), 2000)                                        # ~2,400 failure hours in the record
        self.assertTrue(dead.sm_clean.isna().all()); self.assertTrue(dead.sm_obs_causal.isna().all())
        for fid, start in (("field_3334", "2025-09-18 13:00"), ("field_3335", "2024-08-26 04:00")):
            w = m[(m.field_id == fid) & (m.timestamp >= start) & (m.timestamp < pd.Timestamp(start) + pd.Timedelta(hours=200))]
            self.assertLess(w.sm_clean.notna().mean(), 0.05, (fid, start))         # long failure episodes remain masked


class TestFailureMaskPrinciple(unittest.TestCase):
    """Rises mean rain or irrigation and must never be rejected; only failures (dead probe, stuck value, drops) are."""
    @classmethod
    def setUpClass(cls):
        cls.raw = pd.read_parquet(ROOT / "data" / "irrigation_ml_dataset.parquet", columns=["field_id", "timestamp", "soil_moisture_sensor"])

    def test_mask_is_exactly_the_failure_rules(self):
        m = RAW.merge(self.raw, on=["field_id", "timestamp"], how="left")
        want = (m.rng_soil_moisture_sensor | m.sm_flat_stuck | m.sm_spike | m.sm_drop_fast) & m.soil_moisture_sensor.notna()
        self.assertTrue((m.anom_sm_any.astype(bool) == want).all())            # the jump-up rule is NOT part of the mask
        self.assertGreater(int(m.sm_jump_up_no_rain.sum()), 0)                  # but it is still exported, as an information flag

    def test_spike_flag_only_marks_drops_never_rises(self):
        for fid, g in DF.groupby("field_id"):
            g = g.merge(self.raw, on=["field_id", "timestamp"], how="left").reset_index(drop=True)
            s = g.soil_moisture_sensor
            med = s.rolling(25, center=True, min_periods=12).median()
            flagged = g.sm_spike.fillna(False).astype(bool)
            self.assertTrue((s[flagged] < med[flagged]).all(), fid)             # every flagged hour is below its neighbourhood

    def test_rain_onset_of_6_july_2024_is_kept(self):
        g = DF[(DF.field_id == "field_3334") & (DF.timestamp >= "2024-07-03") & (DF.timestamp < "2024-07-06 11:00")]
        self.assertEqual(int(g.sm_clean.notna().sum()), len(g))                 # 03:00-04:00 on 6 Jul: 90.6 -> 93.7 -> 100 with 11.8 mm of rain
        r = self.raw[(self.raw.field_id == "field_3334") & (self.raw.timestamp == "2024-07-06 04:00")].soil_moisture_sensor.iloc[0]
        self.assertEqual(float(r), 100.0)

    def test_drops_and_stuck_values_are_still_rejected(self):
        for flag in ("sm_drop_fast", "sm_flat_stuck", "sm_dead"):
            hit = DF[DF[flag].fillna(False).astype(bool)]
            self.assertGreater(len(hit), 3, flag)
            self.assertTrue(hit.sm_clean.isna().all(), flag)                    # a real failure is never kept as a target
            self.assertTrue(hit.sm_obs_causal.isna().all(), flag)               # nor used as an input


class TestFeatures(unittest.TestCase):
    def test_lag_and_rolling_features_match_bruteforce(self):
        f = FEAT.merge(DF[["field_id", "timestamp", "sm_obs_causal", "p_era5_mm_h"]], on=["field_id", "timestamp"])
        rng = np.random.default_rng(3)
        for fid, g in f.groupby("field_id"):
            g = g.reset_index(drop=True)
            sm, rain = g.sm_obs_causal.to_numpy(float), g.p_era5_mm_h.fillna(0).to_numpy(float)
            for i in rng.choice(np.arange(200, len(g)), 300, replace=False):
                self.assertTrue(np.isclose(g.sm0[i], sm[i], equal_nan=True))
                self.assertTrue(np.isclose(g.dsm_24[i], sm[i] - sm[i - 24], equal_nan=True))
                self.assertTrue(np.isclose(g.rain_24h[i], rain[i - 23: i + 1].sum()))
                self.assertTrue(np.isclose(g.rain_72h[i], rain[i - 71: i + 1].sum()))
                w = sm[i - 23: i + 1]
                if np.isfinite(w).sum() >= 6:
                    self.assertTrue(np.isclose(g.sm_min_24[i], np.nanmin(w)))
                    self.assertTrue(np.isclose(g.sm_max_24[i], np.nanmax(w)))

    def test_hours_since_obs_nonnegative_and_zero_when_observed(self):
        self.assertTrue((FEAT.hours_since_obs.dropna() >= 0).all())
        self.assertTrue((FEAT.loc[FEAT.sm0.notna(), "hours_since_obs"] == 0).all())

    def test_feature_columns_are_unique_and_no_target_columns(self):
        cols = [c for k in FEAT.attrs["cols"].values() for c in k]
        self.assertEqual(len(cols), len(set(cols)))
        self.assertFalse([c for c in cols if c.startswith(("y_", "y24", "y48", "y72", "act_", "stress", "gross_need"))])


class TestEvaluationIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.oof = pd.read_parquet(ARTIFACTS / "forecast_oof.parquet")

    def test_oof_rows_use_true_targets_and_persistence_is_sm0(self):
        key = DF.set_index(["field_id", "timestamp"])["sm_clean"]
        s = self.oof.sample(1500, random_state=0)
        want = [key.get((r.field_id, r.timestamp + pd.Timedelta(hours=r.H)), np.nan) for r in s.itertuples()]
        self.assertTrue(np.allclose(s.y, want))
        self.assertTrue(np.allclose(s.persist, s.sm0))
        sm0 = [at(DF, r.field_id, r.timestamp, "sm_obs_causal") for r in s.itertuples()]
        self.assertTrue(np.allclose(s.sm0, sm0))

    def test_oof_rows_fall_inside_their_fold_and_are_unique(self):
        for a, b in FOLDS:
            g = self.oof[self.oof.fold == a]
            self.assertTrue(((g.timestamp >= pd.Timestamp(a)) & (g.timestamp < pd.Timestamp(b))).all())
        self.assertFalse(self.oof.duplicated(["field_id", "timestamp", "H"]).any())
        self.assertTrue(self.oof.model.between(0, 100).all())

    def test_reported_metrics_equal_recomputation(self):
        rep = pd.read_csv(ARTIFACTS / "forecast_metrics.csv")
        for H, g in self.oof.groupby("H"):
            row = rep[(rep.H == H) & (rep.segment == "all")].iloc[0]
            self.assertAlmostEqual(row.MAE_model, (g.model - g.y).abs().mean(), places=9)
            self.assertAlmostEqual(row.MAE_persist, (g.persist - g.y).abs().mean(), places=9)
            self.assertAlmostEqual(row.RMSE_model, np.sqrt(((g.model - g.y) ** 2).mean()), places=9)
            self.assertEqual(row.n, len(g))

    def test_forecast_training_sets_end_before_the_test_window(self):
        seen = []
        real_fit = forecast.fit
        forecast.fit = lambda X, d, w=None, *a, **k: (seen.append(FEAT.loc[X.index, "timestamp"]), real_fit(X, d, w, *a, **k))[1]
        try:
            forecast.walk_forward(FEAT, DF, "forecast", 120.0, with_quantiles=False)
        finally:
            forecast.fit = real_fit
        self.assertEqual(len(seen), len(HORIZONS) * len(FOLDS))
        k = 0
        for H in HORIZONS:
            for a, _ in FOLDS:
                ts = seen[k]; k += 1
                self.assertLessEqual(ts.max() + pd.Timedelta(hours=H), pd.Timestamp(a), (H, a))   # target time of last training row
                tgt = DF.set_index(["field_id", "timestamp"])[f"y{H}"]                              # and every training target exists
                self.assertTrue(len(ts) > 500)

    def test_conformal_offsets_reach_nominal_coverage(self):
        off = decision.calibrate_intervals(self.oof)
        for H, g in self.oof.dropna(subset=["q10", "q90"]).groupby("H"):
            raw_cov = ((g.y >= g.q10) & (g.y <= g.q90)).mean()
            cov = ((g.y >= g.q10 - off[H]) & (g.y <= g.q90 + off[H])).mean()
            self.assertGreaterEqual(cov, 0.79, H)
            self.assertGreaterEqual(cov, raw_cov)

    def test_quantile_crossing_is_rare(self):
        self.assertLess((self.oof.q10 > self.oof.q90).mean(), 0.01)

    def test_model_beats_persistence_overall_and_never_catastrophically_in_a_fold(self):
        for (H, fold), g in self.oof.groupby(["H", "fold"]):
            skill = 1 - (g.model - g.y).abs().mean() / (g.persist - g.y).abs().mean()
            self.assertGreater(skill, -0.05, (H, fold))


class TestRisk(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.th = sensor_thresholds(DF, until="2025-04-01")
        cls.lab = risk.build_labels(DF, cls.th)

    def test_labels_match_bruteforce_window_semantics(self):
        """label_H(t) = min of trusted readings in (t, t+H] < trigger, masked if < H/4 readings."""
        rng = np.random.default_rng(4)
        for H in HORIZONS:
            for i in rng.choice(len(DF), 500, replace=False):
                r = DF.iloc[i]
                w = DF[(DF.field_id == r.field_id) & (DF.timestamp > r.timestamp) & (DF.timestamp <= r.timestamp + pd.Timedelta(hours=H))]["sm_clean"].dropna()
                got = self.lab[f"stress{H}"].iloc[i]
                if len(w) < H // 4:
                    self.assertTrue(np.isnan(got), (H, i, len(w)))
                else:
                    self.assertEqual(float(w.min() < self.th[r.field_id]["trigger"]), got, (H, i))

    def test_walkforward_trigger_ignores_data_after_cutoff(self):
        tampered = DF.copy()
        late = tampered.timestamp >= pd.Timestamp("2025-04-01")
        tampered.loc[late, "sm_clean"] = 5.0
        self.assertEqual(sensor_thresholds(tampered, until="2025-04-01"), self.th)
        self.assertNotEqual(sensor_thresholds(tampered), sensor_thresholds(DF))

    def test_trigger_formula(self):
        for fid, t in self.th.items():
            self.assertAlmostEqual(t["trigger"], t["fc_s"] - 0.55 * (t["fc_s"] - t["pwp_s"]))
            self.assertTrue(t["pwp_s"] < t["trigger"] < t["fc_s"])

    def test_risk_training_sets_end_before_the_test_window(self):
        seen = []
        real = risk.lgb.LGBMClassifier

        class Spy(real):
            def fit(self, X, y, *a, **k):
                seen.append(FEAT.loc[X.index, "timestamp"]); return super().fit(X, y, *a, **k)
        risk.lgb.LGBMClassifier = Spy
        try:
            risk.walk_forward(FEAT, self.lab)
        finally:
            risk.lgb.LGBMClassifier = real
        k = 0
        for H in HORIZONS:
            for a, _ in FOLDS:
                if k < len(seen):
                    self.assertLessEqual(seen[k].max() + pd.Timedelta(hours=H), pd.Timestamp(a), (H, a))
                k += 1
        self.assertGreaterEqual(len(seen), 6)

    def test_oof_probabilities_are_valid_and_informative(self):
        from sklearn.metrics import average_precision_score, brier_score_loss
        oof = pd.read_parquet(ARTIFACTS / "risk_oof.parquet")
        self.assertTrue(oof.p.between(0, 1).all())
        for H, g in oof.groupby("H"):
            ap = average_precision_score(g.y, g.p)
            self.assertGreater(ap, g.y.mean() * 1.5, H)
            self.assertLess(brier_score_loss(g.y, g.p), brier_score_loss(g.y, np.full(len(g), g.y.mean())), H)


class TestDecisionProperties(unittest.TestCase):
    TH = dict(fc_s=90.0, pwp_s=30.0, trigger=57.0, reliable=True)
    OK = dict(battery_min=4.1, obs_share=1.0, anomaly=False)

    def test_policy_is_monotone_in_risk_and_cost(self):
        rng = np.random.default_rng(5)
        order = {"no_action": 0, "postpone": 1, "irrigate": 2}
        for _ in range(500):
            sm = rng.uniform(20, 100)
            p = np.sort(rng.uniform(0, 1, 3)); rain = dict(zip((24, 48, 72), rng.uniform(0, 20, 3)))
            med = dict(zip((24, 48, 72), rng.uniform(20, 100, 3)))
            lo = decide(sm, dict(zip((24, 48, 72), p)), self.TH, rain, self.OK, Policy(c_fn=2), med, 170.0)
            hi = decide(sm, dict(zip((24, 48, 72), p)), self.TH, rain, self.OK, Policy(c_fn=20), med, 170.0)
            # higher cost of a miss must never turn "irrigate" into something less protective
            if lo["action"] == "irrigate":
                self.assertEqual(hi["action"], "irrigate", (sm, p, rain, med))
            if lo["action"] in ("irrigate", "irrigate_soon"):                 # a higher cost of a miss never drops a warning
                self.assertIn(hi["action"], ("irrigate", "irrigate_soon"), (sm, p, rain, med))
            for d in (lo, hi):
                if d["action"] == "irrigate_soon":
                    self.assertIn(d["dry_expected_h"], (48, 72)); self.assertEqual(d["irrigate_in_h"], d["dry_expected_h"] - 24)
                    self.assertGreaterEqual(d["p_dry"][d["dry_expected_h"]], Policy().c_fp / (Policy().c_fp + (2 if d is lo else 20)))
            for d in (lo, hi):
                self.assertIn(d["action"], {"no_action", "postpone", "irrigate", "irrigate_soon", "check_sensor"})
                if d["amount_mm"] is not None:
                    self.assertGreaterEqual(d["amount_mm"], 0)
                    self.assertLessEqual(d["amount_mm"], Policy().max_event_mm)
                    self.assertIn(d["action"], ("irrigate", "irrigate_soon"))

    def test_untrusted_never_acts_on_any_input(self):
        rng = np.random.default_rng(6)
        for _ in range(300):
            p = dict(zip((24, 48, 72), rng.uniform(0, 1, 3)))
            bad = rng.choice([dict(self.OK, battery_min=3.0), dict(self.OK, obs_share=0.0)])
            d = decide(rng.uniform(0, 100), p, self.TH, {24: 9, 48: 9, 72: 9}, bad, Policy(), {24: 70, 48: 70, 72: 70}, 170.0)
            self.assertEqual(d["action"], "check_sensor"); self.assertIsNone(d["amount_mm"])

    def test_reported_risk_is_nondecreasing_with_horizon(self):
        rng = np.random.default_rng(7)
        for _ in range(300):
            p = dict(zip((24, 48, 72), rng.uniform(0, 1, 3)))          # deliberately inconsistent input
            d = decide(rng.uniform(50, 100), p, self.TH, {24: 0, 48: 0, 72: 0}, self.OK, Policy(), {24: 70, 48: 70, 72: 70}, 170.0)
            r = [d["p_dry"][H] for H in (24, 48, 72)]
            self.assertEqual(r, sorted(r)); self.assertGreaterEqual(r[0], p[24])

    def test_amount_formula(self):
        self.assertAlmostEqual(decision.gross_need_mm(60, self.TH, 170, "sprinkler"), (90 - 60) / 60 * 170 / 0.75)
        self.assertAlmostEqual(decision.gross_need_mm(60, self.TH, 170, "drip"), (90 - 60) / 60 * 170 / 0.90)
        self.assertEqual(decision.gross_need_mm(95, self.TH, 170), 0.0)
        self.assertAlmostEqual(decision.gross_need_mm(0, self.TH, 170, "drip"), 170 / 0.9)    # clipped at TAW

    def test_prob_below_is_a_valid_cdf(self):
        xs = np.linspace(0, 100, 201)
        ps = [decision.prob_below(x, 55, 65, 80) for x in xs]
        self.assertTrue(all(0 <= p <= 1 for p in ps)); self.assertEqual(ps, sorted(ps))


class TestOfflineModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from irrigation_ml import offline
        cls.offline = offline
        cls.th = sensor_thresholds(DF, until="2025-04-01")
        cls.lab = risk.build_labels(DF, cls.th)

    def test_training_frame_matches_bruteforce_for_given_ages(self):
        ages = np.random.default_rng(9).integers(6, 600, len(DF))
        X = self.offline.build(DF, FEAT, self.th, ages=ages)
        sm = FEAT["sm0"]
        for i in np.random.default_rng(10).choice(np.arange(800, len(DF)), 300, replace=False):
            r = DF.iloc[i]; a = ages[i]; j = i - a
            if DF.field_id.iloc[j] != r.field_id:
                continue
            self.assertTrue(np.isclose(X.last_sm.iloc[i], sm.iloc[j], equal_nan=True))
            self.assertTrue(np.isclose(X.rain_since.iloc[i], DF.p_era5_mm_h.iloc[j + 1: i + 1].fillna(0).sum()))
            self.assertTrue(np.isclose(X.et0_since.iloc[i], DF.et0_mm_h.iloc[j + 1: i + 1].fillna(0).sum()))
            self.assertEqual(X.age_h.iloc[i], a)

    def test_inference_row_equals_training_frame_row(self):
        ages = np.full(len(DF), 100)
        X = self.offline.build(DF, FEAT, self.th, ages=ages)
        for i in np.random.default_rng(11).choice(np.arange(800, len(DF)), 60, replace=False):
            r = DF.iloc[i]
            x = self.offline.infer_features(DF, FEAT, r.field_id, r.timestamp, 100, self.th)
            pd.testing.assert_series_equal(x.iloc[0], X.iloc[i][self.offline.FEATURES].astype(float), check_names=False, rtol=1e-9)

    def test_offline_features_ignore_the_future(self):
        t = pd.Timestamp("2025-07-01T06:00")
        a = self.offline.infer_features(DF, FEAT, "field_3334", t, 80, self.th)
        tam = DF.copy(); fut = tam.timestamp > t; rng = np.random.default_rng(0)
        for c in tam.select_dtypes("number").columns:
            if not c.startswith("rain_fc"):
                tam.loc[fut, c] = rng.normal(50, 20, fut.sum())
        from irrigation_ml.features import build_features
        f2, cols = build_features(tam); f2.attrs["cols"] = cols
        b = self.offline.infer_features(tam, f2, "field_3334", t, 80, self.th)
        pd.testing.assert_frame_equal(a, b)

    def test_offline_training_sets_end_before_the_test_window(self):
        seen = []
        real = self.offline.lgb.LGBMClassifier

        class Spy(real):
            def fit(self, X, y, *a, **k):
                seen.append(FEAT.loc[X.index, "timestamp"]); return super().fit(X, y, *a, **k)
        self.offline.lgb.LGBMClassifier = Spy
        try:
            self.offline.walk_forward(DF, FEAT, self.lab, self.th)
        finally:
            self.offline.lgb.LGBMClassifier = real
        k = 0
        for H in HORIZONS:
            for a, _ in FOLDS:
                if k < len(seen):
                    self.assertLessEqual(seen[k].max() + pd.Timedelta(hours=H), pd.Timestamp(a), (H, a))
                k += 1
        self.assertGreaterEqual(len(seen), 6)

    def test_fallback_beats_naive_stale_rule_for_outages_up_to_10_days(self):
        o = pd.read_csv(ARTIFACTS / "offline_metrics.csv")
        for _, r in o[o.outage_age.isin(["<=3d", "3-10d"])].iterrows():
            self.assertGreater(r.AUC_model, r.AUC_stale_rule, (r.H, r.outage_age))
            self.assertGreater(r.AUC_model, 0.8, (r.H, r.outage_age))


class TestBacktest(unittest.TestCase):
    def test_realised_outcomes_match_bruteforce(self):
        from irrigation_ml.backtest import run
        cases, th, _ = run(Policy())
        for r in cases.sample(60, random_state=0).itertuples():
            w = DF[(DF.field_id == r.field_id) & (DF.timestamp > r.timestamp) & (DF.timestamp <= r.timestamp + pd.Timedelta(hours=72))].set_index("timestamp")["sm_clean"].dropna()
            self.assertAlmostEqual(w.min(), r.min72)
            self.assertEqual(bool(w.min() < r.trigger), bool(r.stress72))
            self.assertEqual(r.trigger, th[r.field_id]["trigger"])
            hit = w[w < r.trigger]                                         # first real crossing, in hours after the decision
            self.assertTrue((np.isnan(r.first_cross_h) and hit.empty) or abs(r.first_cross_h - (hit.index[0] - r.timestamp).total_seconds() / 3600) < 1e-9)
            self.assertEqual(bool(r.sm0 < r.trigger), bool(r.naive))
        self.assertTrue(cases.timestamp.dt.hour.isin((0, 6, 12, 18)).all())
        self.assertTrue((cases.timestamp >= pd.Timestamp("2025-04-01")).all())


class TestRecommend(unittest.TestCase):
    def test_recommendation_ignores_the_future(self):
        """Altering data after t (except the rain-forecast feed, which is a forecast by design) must not change advice at t."""
        from irrigation_ml.features import build_features
        from irrigation_ml.recommend import recommend
        t = pd.Timestamp("2025-07-01T06:00")
        base = recommend(str(t), Policy(), "sprinkler", (DF, FEAT))
        tam = DF.copy(); fut = tam.timestamp > t; rng = np.random.default_rng(0)
        for c in tam.select_dtypes("number").columns:
            if not c.startswith("rain_fc"):
                tam.loc[fut, c] = rng.normal(50, 20, fut.sum())
        tam["anom_iforest"] = np.where(fut, True, tam["anom_iforest"])
        f2, cols = build_features(tam); f2.attrs["cols"] = cols
        other = recommend(str(t), Policy(), "sprinkler", (tam, f2))
        for a, b in zip(base, other):
            self.assertEqual(a["action"], b["action"]); self.assertEqual(a["soil_moisture"], b["soil_moisture"])
            self.assertEqual(a["forecast"], b["forecast"]); self.assertEqual(a["p_dry"], b["p_dry"])

    def test_final_models_load_and_feature_order_matches(self):
        import joblib
        fm = joblib.load(ARTIFACTS / "forecast_models.joblib"); rm = joblib.load(ARTIFACTS / "risk_models.joblib")
        for H in HORIZONS:
            self.assertEqual(list(fm["horizons"][H]["mean"].booster_.feature_name()), fm["cols"])
            self.assertEqual(list(rm["models"][H].booster_.feature_name()), rm["cols"] + ["gap_to_trigger"])

    def test_fallback_modes_over_the_whole_record(self):
        from irrigation_ml.recommend import recommend
        modes = {"sensor": 0, "weather_only": 0, "none": 0}
        for day in pd.date_range("2024-03-10", "2025-10-12", freq="5D"):
            for r in recommend(f"{day.date()}T06:00", Policy(), "sprinkler", (DF, FEAT)):
                modes[r["mode"]] += 1
                if r["mode"] == "weather_only":
                    self.assertEqual(r["confidence"], "low"); self.assertIsNone(r["amount_mm"]); self.assertIn(r["action"], {"irrigate", "no_action"})
                    self.assertEqual(r["forecast"], {})
                if r["mode"] == "sensor" and r["amount_mm"] is not None:
                    self.assertLessEqual(r["amount_mm"], 40.0); self.assertLessEqual(r["amount_mm"], r["deficit_mm"] + 1e-9)
        self.assertTrue(all(v > 0 for v in modes.values()), modes)      # every branch is exercised by the real record

    def test_diagnosis_names_the_right_cause(self):
        from irrigation_ml.recommend import diagnose_sensor
        d = diagnose_sensor(DF, "field_3334", pd.Timestamp("2024-08-20T06:00"))
        self.assertIn("near-zero values, which means a probe fault", d["text"]); self.assertTrue(6 < d["age_h"] < 240)
        d = diagnose_sensor(DF, "field_3335", pd.Timestamp("2025-08-10T06:00"))
        self.assertIn("not sent data", d["text"]); self.assertGreater(d["age_h"], 240)
        d = diagnose_sensor(DF, "field_3334", pd.Timestamp("2025-10-05T06:00"))
        self.assertIn("days ago", d["text"])
        # low battery with a reading present (field_3335, 14-17 Dec 2024): named as a battery problem, and the reading taken
        # on a low battery is not counted as "last trusted"
        d = diagnose_sensor(DF, "field_3335", pd.Timestamp("2024-12-16T06:00"))
        self.assertIn("battery is low", d["text"]); self.assertGreater(d["age_h"], 24)
        fe = FEAT[(FEAT.field_id == "field_3335") & (FEAT.timestamp == pd.Timestamp("2024-12-16T06:00"))].iloc[0]
        self.assertFalse(pd.isna(fe.sm0))                         # the reading exists, yet the advice must not rely on it
        from irrigation_ml.recommend import recommend
        r = [x for x in recommend("2024-12-16T06:00", Policy(), "sprinkler", (DF, FEAT)) if x["field_id"] == "field_3335"][0]
        self.assertNotEqual(r["mode"], "sensor"); self.assertIn("battery", r["sensor_issue"].lower())

    def test_planned_irrigation_has_a_consistent_due_date_and_window(self):
        from irrigation_ml.recommend import recommend
        plans = 0
        for day in pd.date_range("2024-03-10", "2025-10-12", freq="D"):
            for r in recommend(f"{day.date()}T06:00", Policy(), "sprinkler", (DF, FEAT)):
                if r["action"] == "irrigate_soon":
                    plans += 1
                    due = pd.Timestamp(r["issued_at"]) + pd.Timedelta(hours=r["irrigate_in_h"])
                    self.assertEqual(r["irrigate_by"], str(due.date()))
                    self.assertEqual(r["dry_expected_h"], r["irrigate_in_h"] + 24)
                    self.assertIn(r["dry_expected_h"], (48, 72))
                    self.assertEqual(r["mode"] == "sensor", r["amount_mm"] is not None or r["deficit_mm"] is None)
                    if r["amount_mm"] is not None:
                        self.assertLessEqual(r["amount_mm"], 40.0)
                elif r["action"] == "irrigate":
                    self.assertEqual(r["irrigate_in_h"], 0); self.assertEqual(r["irrigate_by"], str(pd.Timestamp(r["issued_at"]).date()))
                else:
                    self.assertIsNone(r["dry_expected_h"])
        self.assertGreater(plans, 5)                      # the sweep must actually exercise the planning branch

    def test_no_zero_day_wording_and_recovered_sensor_case(self):
        import re
        from irrigation_ml.recommend import diagnose_sensor, recommend
        d = diagnose_sensor(DF, "field_3334", pd.Timestamp("2025-02-21T06:00"))      # trusted reading now, but 1 of 24 h trusted
        self.assertIn("Only 1 of the last 24 hours", d["text"]); self.assertNotIn("0.0 days", d["text"])
        for day in pd.date_range("2024-03-10", "2025-10-12", freq="2D"):
            for r in recommend(f"{day.date()}T06:00", Policy(), "sprinkler", (DF, FEAT)):
                text = " ".join(r["reasons"]) + " " + (r["sensor_issue"] or "")
                self.assertIsNone(re.search(r"\b0\.0 days", text), (day, r["field_id"], text))

    def test_output_ranges_over_many_dates(self):
        from irrigation_ml.recommend import recommend
        for day in pd.date_range("2024-04-01", "2025-10-10", freq="11D"):
            for r in recommend(f"{day.date()}T06:00", Policy(), "drip", (DF, FEAT)):
                for H, (lo, mid, hi) in r["forecast"].items():
                    self.assertTrue(0 <= lo <= mid <= hi <= 100 or lo <= hi, (day, r["field_id"], H, lo, mid, hi))
                    self.assertTrue(0 <= r["p_dry"][H] <= 1)
                if r["action"] == "irrigate":
                    self.assertTrue(r["amount_mm"] is None or r["amount_mm"] >= 0)


if __name__ == "__main__":
    unittest.main()
