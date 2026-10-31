import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from irrigation_ml.decision import Policy, decide, decide_offline, fmt_duration, gross_need_mm, prob_below  # noqa: E402

TH = dict(fc_s=90.0, pwp_s=30.0, trigger=57.0, reliable=True)
OK = dict(battery_min=4.1, obs_share=1.0, anomaly=False)
NORAIN = {24: 0.0, 48: 0.0, 72: 0.0}
LOW = {24: 0.01, 48: 0.01, 72: 0.01}


class TestDecision(unittest.TestCase):
    def test_prob_below_monotone_and_anchored(self):
        self.assertAlmostEqual(prob_below(70, 60, 70, 80), 0.5)
        xs = [prob_below(t, 60, 70, 80) for t in range(40, 100, 5)]
        self.assertEqual(xs, sorted(xs))

    def test_unusable_probe_never_recommends_water(self):
        for bad in (dict(OK, battery_min=3.5), dict(OK, obs_share=0.1)):
            self.assertEqual(decide(50.0, LOW, TH, NORAIN, bad)["action"], "check_sensor")
        self.assertEqual(decide(np.nan, LOW, TH, NORAIN, OK)["action"], "check_sensor")

    def test_anomaly_flag_alone_gives_low_confidence_advice_not_a_block(self):
        d = decide(45.0, LOW, TH, NORAIN, dict(OK, anomaly=True), taw_mm=170.0)
        self.assertEqual(d["action"], "irrigate"); self.assertEqual(d["confidence"], "low")
        self.assertTrue(any("unusual" in r for r in d["reasons"]))

    def test_dry_soil_first_expected_at_48h_plans_irrigation_for_tomorrow(self):
        d = decide(75.0, {24: 0.01, 48: 0.5, 72: 0.6}, TH, NORAIN, OK, median_fc={24: 74.0, 48: 60.0, 72: 56.0}, taw_mm=170.0)
        self.assertEqual(d["action"], "irrigate_soon"); self.assertEqual(d["dry_expected_h"], 48); self.assertEqual(d["irrigate_in_h"], 24)

    def test_dry_soil_first_expected_at_72h_plans_irrigation_in_two_days(self):
        d = decide(75.0, {24: 0.01, 48: 0.02, 72: 0.6}, TH, NORAIN, OK, median_fc={24: 74.0, 48: 70.0, 72: 56.0}, taw_mm=170.0)
        self.assertEqual(d["action"], "irrigate_soon"); self.assertEqual(d["dry_expected_h"], 72); self.assertEqual(d["irrigate_in_h"], 48)
        self.assertIn("2 day", " ".join(d["reasons"]))

    def test_planned_amount_uses_forecast_moisture_at_the_dry_window_and_is_capped(self):
        d = decide(75.0, {24: 0.01, 48: 0.02, 72: 0.6}, TH, NORAIN, OK, median_fc={24: 74.0, 48: 70.0, 72: 56.0}, taw_mm=170.0)
        self.assertAlmostEqual(d["deficit_mm"], gross_need_mm(56.0, TH, 170.0))
        self.assertEqual(d["amount_mm"], min(d["deficit_mm"], 40.0))

    def test_imminent_risk_is_irrigate_today_not_a_plan(self):
        d = decide(75.0, {24: 0.6, 48: 0.7, 72: 0.8}, TH, NORAIN, OK, taw_mm=170.0)
        self.assertEqual(d["action"], "irrigate"); self.assertEqual(d["irrigate_in_h"], 0); self.assertIsNone(d["dry_expected_h"])

    def test_plan_is_not_made_when_forecast_recovers_after_rain(self):
        rain = {24: 0.0, 48: 10.0, 72: 10.0}
        d = decide(75.0, {24: 0.01, 48: 0.5, 72: 0.5}, TH, rain, OK, median_fc={24: 74.0, 48: 70.0, 72: 70.0}, taw_mm=170.0)
        self.assertEqual(d["action"], "postpone")

    def test_wet_soil_no_action(self):
        self.assertEqual(decide(80.0, LOW, TH, NORAIN, OK)["action"], "no_action")

    def test_dry_now_irrigates_with_amount(self):
        d = decide(45.0, LOW, TH, NORAIN, OK, taw_mm=170.0)
        self.assertEqual(d["action"], "irrigate")
        self.assertAlmostEqual(d["deficit_mm"], gross_need_mm(45.0, TH, 170.0))
        self.assertAlmostEqual(d["amount_mm"], min(d["deficit_mm"], Policy().max_event_mm))
        self.assertGreater(d["amount_mm"], 0)

    def test_amount_is_capped_and_deficit_reported(self):
        d = decide(30.0, LOW, TH, NORAIN, OK, taw_mm=170.0)             # deficit far above one application
        self.assertEqual(d["amount_mm"], 40.0); self.assertGreater(d["deficit_mm"], 40.0); self.assertTrue(d["capped"])
        d = decide(30.0, LOW, TH, NORAIN, OK, Policy(max_event_mm=500.0), taw_mm=170.0)
        self.assertFalse(d["capped"]); self.assertAlmostEqual(d["amount_mm"], d["deficit_mm"])

    def test_rain_alone_does_not_postpone(self):
        rain = {24: 10.0, 48: 10.0, 72: 10.0}
        med_dry = {24: 45.0, 48: 45.0, 72: 45.0}           # forecast says moisture stays below trigger
        self.assertEqual(decide(45.0, LOW, TH, rain, OK, median_fc=med_dry)["action"], "irrigate")

    def test_postpone_when_forecast_recovers(self):
        rain = {24: 10.0, 48: 10.0, 72: 10.0}
        risk_later = {24: 0.01, 48: 0.5, 72: 0.5}
        med_ok = {24: 70.0, 48: 70.0, 72: 70.0}
        self.assertEqual(decide(75.0, risk_later, TH, rain, OK, median_fc=med_ok)["action"], "postpone")

    def test_asymmetric_cost_lowers_threshold(self):
        self.assertLess(Policy(c_fn=10).tau, Policy(c_fn=1).tau)
        p = {24: 0.2, 48: 0.2, 72: 0.2}
        self.assertEqual(decide(75.0, p, TH, NORAIN, OK, Policy(c_fn=1))["action"], "no_action")
        self.assertEqual(decide(75.0, p, TH, NORAIN, OK, Policy(c_fn=10))["action"], "irrigate")

    def test_unreliable_trigger_lowers_confidence(self):
        d = decide(80.0, LOW, dict(TH, reliable=False), NORAIN, OK)
        self.assertEqual(d["confidence"], "low")


class TestWording(unittest.TestCase):
    def test_durations_are_human(self):
        self.assertEqual(fmt_duration(0.0), "less than an hour"); self.assertEqual(fmt_duration(0.9), "less than an hour")
        self.assertEqual(fmt_duration(1.0), "1 hour"); self.assertEqual(fmt_duration(5.2), "5 hours")
        self.assertEqual(fmt_duration(47.0), "47 hours"); self.assertEqual(fmt_duration(48.0), "2.0 days"); self.assertEqual(fmt_duration(200.0), "8.3 days")

    def test_offline_reason_never_says_zero_days(self):
        for age in (0.0, 3.0, 30.0, 100.0, 239.0):
            d = decide_offline({24: .5, 48: .5, 72: .5}, age, NORAIN, TH)
            self.assertNotRegex(" ".join(d["reasons"]), r"\b0\.0 days")


class TestOfflineDecision(unittest.TestCase):
    TH = TH
    HI = {24: 0.6, 48: 0.7, 72: 0.8}

    def test_too_old_or_unknown_outage_gives_no_estimate(self):
        for age in (241.0, 1000.0, None, float("nan")):
            d = decide_offline(self.HI, age, NORAIN, self.TH)
            self.assertEqual(d["action"], "check_sensor"); self.assertEqual(d["mode"], "none")

    def test_estimate_is_always_low_confidence_without_amount_or_postpone(self):
        rng = np.random.default_rng(8)
        for _ in range(300):
            p = dict(zip((24, 48, 72), rng.uniform(0, 1, 3)))
            d = decide_offline(p, rng.uniform(6, 240), {24: 20, 48: 20, 72: 20}, self.TH, Policy(c_fn=float(rng.choice([1, 5, 20]))))
            self.assertEqual(d["mode"], "weather_only"); self.assertEqual(d["confidence"], "low")
            self.assertIn(d["action"], {"irrigate", "irrigate_soon", "no_action"}); self.assertIsNone(d["amount_mm"])
            if d["action"] == "irrigate_soon":
                self.assertEqual(d["dry_expected_h"] - d["irrigate_in_h"], 24); self.assertIn(d["dry_expected_h"], (48, 72))
            r = [d["p_dry"][H] for H in (24, 48, 72)]; self.assertEqual(r, sorted(r))

    def test_threshold_follows_cost(self):
        p = {24: 0.2, 48: 0.2, 72: 0.2}
        self.assertEqual(decide_offline(p, 48, NORAIN, self.TH, Policy(c_fn=1))["action"], "no_action")
        self.assertEqual(decide_offline(p, 48, NORAIN, self.TH, Policy(c_fn=10))["action"], "irrigate")


class TestNoLeakage(unittest.TestCase):
    def test_features_at_t_ignore_the_future(self):
        from irrigation_ml.data import add_targets, load_hourly
        from irrigation_ml.features import build_features
        df = add_targets(load_hourly()).sort_values(["field_id", "timestamp"]).reset_index(drop=True)
        cutoff = pd.Timestamp("2025-03-01")
        future = df["timestamp"] > cutoff
        tampered = df.copy()
        rng = np.random.default_rng(0)
        for c in tampered.select_dtypes("number").columns:
            tampered.loc[future, c] = rng.normal(50, 20, future.sum())
        tampered["anom_iforest"] = np.where(future, True, tampered["anom_iforest"])
        a, cols = build_features(df)
        b, _ = build_features(tampered)
        past = a["timestamp"] <= cutoff
        causal = [c for k in ("sensor", "wx") for c in cols[k]]   # 'fc'/'retro' groups are inputs from forecast/reanalysis feeds
        pd.testing.assert_frame_equal(a.loc[past, causal].reset_index(drop=True), b.loc[past, causal].reset_index(drop=True))

class TestRecommendEdgeCases(unittest.TestCase):
    def test_untrusted_probe_has_no_forecast_and_ui_fields_are_consistent(self):
        # 2025-08-10: field_3334 has a reading but fails the trust check (regression for KeyError: 24 in the UI)
        from irrigation_ml.recommend import recommend, render
        for r in recommend("2025-08-10T06:00"):
            if r["mode"] != "sensor":
                self.assertEqual(r["forecast"], {})                      # no forecast from an unusable reading
            self.assertLessEqual(set(r["forecast"]), set(r["p_dry"]))
            self.assertTrue(r["mode"] == "sensor" or r["sensor_issue"])  # the card must be able to say why
        render(recommend("2025-08-10T06:00"))

    def test_every_date_has_consistent_forecast_and_risk(self):
        from irrigation_ml.recommend import recommend
        from irrigation_ml.forecast import prepare
        prepared = prepare()
        for day in pd.date_range("2024-03-10", "2025-10-13", freq="9D"):
            for r in recommend(f"{day.date()}T06:00", prepared=prepared):
                # UI contract: every forecast horizon shown must have a risk value ('p_dry' also holds a 0 = "dry now" key)
                self.assertLessEqual(set(r["forecast"]), set(r["p_dry"]), f"{day} {r['field_id']}")
                if r["mode"] != "sensor":
                    self.assertEqual(r["forecast"], {})
                    self.assertTrue(r["sensor_issue"], f"{day} {r['field_id']}")
                if r["mode"] == "none":
                    self.assertEqual(r["action"], "check_sensor")
                if r["amount_mm"] is not None:
                    self.assertLessEqual(r["amount_mm"], Policy().max_event_mm)


if __name__ == "__main__":
    unittest.main()
