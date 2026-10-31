import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml_pipeline import load_hourly, make_features, build_fold


class MLIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'example.csv'
        ts = pd.date_range('2025-01-01', periods=250, freq='h')
        ts = ts.delete(slice(40, 70))
        self.source = pd.DataFrame({
            'field_id': 'field_3334', 'timestamp': ts,
            'soil_moisture_sensor': np.arange(len(ts)) * 0.015 + 70,
            'battery_voltage': 4.1,
        })
        self.source.to_csv(self.path, index=False)

    def tearDown(self):
        self.temp.cleanup()

    def test_gaps_are_not_interpolated_or_shifted(self):
        frame, qa = load_hourly(self.path)
        self.assertEqual(len(frame), 250)
        gap = frame.loc[frame.timestamp.eq(pd.Timestamp('2025-01-02 18:00'))]
        self.assertTrue(gap.sm_clean.isna().all())
        f, feats = make_features(frame)
        origin = f.loc[f.timestamp.eq(pd.Timestamp('2025-01-01 20:00'))].iloc[0]
        self.assertTrue(np.isnan(origin.target_24h))
        origin2 = f.loc[f.timestamp.eq(pd.Timestamp('2025-01-04 00:00'))].iloc[0]
        future = f.loc[f.timestamp.eq(pd.Timestamp('2025-01-05 00:00'))].iloc[0]
        self.assertAlmostEqual(origin2.target_24h, future.sm_clean)
        self.assertTrue({'battery_voltage', 'battery_low', 'sensor_invalid_now'} <= set(feats))
        self.assertFalse(any(c.startswith('target_') for c in feats))

    def test_walk_forward_purge(self):
        frame, _ = load_hourly(self.path)
        f, _ = make_features(frame)
        tr, te = build_fold(f, 'target_24h', '2025-01-07', '2025-01-11', 24)
        self.assertTrue((f.loc[tr, 'timestamp'] + pd.Timedelta(hours=24) < pd.Timestamp('2025-01-07')).all())
        self.assertTrue((f.loc[te, 'timestamp'] >= pd.Timestamp('2025-01-07')).all())
        self.assertTrue((f.loc[te, 'timestamp'] + pd.Timedelta(hours=24) < pd.Timestamp('2025-01-11')).all())


if __name__ == '__main__':
    unittest.main()
