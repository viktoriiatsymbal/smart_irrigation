import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from irrigation_recommendations import crop_state, threshold_range, SOIL_PARAMS, make_decisions, load_forecasts


def fake_field(theta=.17, smap_lag_hours=.5):
    times = pd.date_range('2025-07-01 00:00:00', periods=54, freq='h')
    return pd.DataFrame({
        'timestamp': times, 'field_id': 'field_3335', 'crop': 'corn',
        'smap_sm_rootzone': theta,
        'smap_timestamp': times - pd.Timedelta(hours=smap_lag_hours),
        'smap_static_mwrtm_sand': .34, 'smap_static_mwrtm_clay': .24,
        'soil_moisture_sensor': 75., 'battery_voltage': 4.1,
        'openmeteo_et0_fao_evapotranspiration': np.nan,
    })


class TestIrrigationRecommendations(unittest.TestCase):
    def test_fao_nominal_threshold(self):
        s = SOIL_PARAMS['loam']
        self.assertAlmostEqual(s['fc'] - .55 * (s['fc'] - s['pwp']), .1785)
        low, high = threshold_range(.55, s)
        self.assertLess(low, .1785)
        self.assertGreater(high, .1785)

    def test_crop_calendar(self):
        self.assertFalse(crop_state(pd.Timestamp('2025-12-15'), 'corn')[0])
        is_growing, kc, depth = crop_state(pd.Timestamp('2025-07-15'), 'corn')
        self.assertTrue(is_growing)
        self.assertAlmostEqual(kc, 1.2)
        self.assertAlmostEqual(depth, 1.2)

    def test_provisional_dry_not_automatic(self):
        result = make_decisions(fake_field())
        self.assertGreater(len(result), 0)
        self.assertTrue((result.recommendation == 'review_irrigation').all())
        self.assertTrue((result.assumption_robustness == 'assumption_sensitive').all())
        self.assertFalse(result.validated_irrigation_label.any())
        self.assertFalse(result.forecast_used.any())

    def test_stale_smap_rejected(self):
        result = make_decisions(fake_field(smap_lag_hours=12))
        self.assertTrue((result.recommendation == 'insufficient_data').all())

    def test_future_issued_forecast_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'weather.csv'
            pd.DataFrame([{'field_id':'field_3335','decision_timestamp':'2025-07-01 06:00:00',
                           'issue_time':'2025-07-01 08:00:00','rain_next24h_mm':30.,
                           'et0_next24h_mm':3.}]).to_csv(path,index=False)
            f = load_forecasts(path)
            self.assertEqual(len(f),0)

    def test_postpone_requires_preissued_forecast(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'weather.csv'
            pd.DataFrame([{'field_id':'field_3335','decision_timestamp':'2025-07-01 06:00:00',
                           'issue_time':'2025-07-01 05:00:00','rain_next24h_mm':40.,
                           'et0_next24h_mm':2.}]).to_csv(path,index=False)
            out = make_decisions(fake_field(),load_forecasts(path))
            first = out[out.decision_timestamp == pd.Timestamp('2025-07-01 06:00:00')].iloc[0]
            self.assertTrue(first.forecast_used)
            self.assertEqual(first.recommendation, 'postpone_review')
            second = out[out.decision_timestamp == pd.Timestamp('2025-07-02 06:00:00')].iloc[0]
            self.assertEqual(second.recommendation, 'review_irrigation')


if __name__ == '__main__':
    unittest.main()
