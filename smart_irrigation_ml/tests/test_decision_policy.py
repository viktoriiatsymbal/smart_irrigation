"""Prevent accidental ground-truth claims and temporal leakage in proxy policy."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from train_decision_policy import (FEATURES, FORBIDDEN, CLASSES, TARGET_NAME,
    INNER_VALID_START, OUTER_FOLDS, GAP_DAYS, engineer_features, assign_actions, metrics)

class PolicyTests(unittest.TestCase):
    def test_class_separation(self):
        self.assertEqual(CLASSES,['no_action','monitor','review_irrigation'])
        self.assertFalse(set(FEATURES)&set(FORBIDDEN))
        self.assertTrue('smap_theta_rootzone_m3m3' in FEATURES)
        self.assertTrue('battery_voltage' in FEATURES)

    def test_temporal_splits_chronological(self):
        for start,end in OUTER_FOLDS:
            self.assertTrue(INNER_VALID_START < pd.Timestamp(start)-pd.Timedelta(days=GAP_DAYS))
            self.assertTrue(pd.Timestamp(start)<pd.Timestamp(end))

    def test_review_threshold(self):
        proba = pd.DataFrame({'no_action':[.6,.1,.3], 'monitor':[.3,.2,.3],
                              'review_irrigation':[.1,.7,.4]})
        self.assertEqual(list(assign_actions(proba,.35)),['no_action','review_irrigation','review_irrigation'])

    def test_per_field_no_review_examples_are_not_fabricated_recall(self):
        y = pd.Series(['no_action','monitor','no_action'])
        pred=np.array(['monitor','monitor','no_action'])
        sc=np.array([.1,.1,.1])
        m=metrics(y,pred,sc,{'field_id':'field_without_proxy_alerts'})
        self.assertTrue(np.isnan(m['review_recall_proxy']))
        self.assertEqual(m['review_false_negatives_proxy'],0)
        self.assertFalse(m['explicit_truth_available'])

    def test_real_calendar_lags_not_previous_observed_row(self):
        time=pd.to_datetime(['2024-07-01 06:00','2024-07-03 06:00','2024-07-04 06:00'])
        data=pd.DataFrame({
            'field_id':['field_3335']*3,'decision_timestamp':time,
            'crop':['sorghum']*3,'soil_class_coarse':['loam']*3,
            'smap_theta_rootzone_m3m3':[.2,.3,.4], 'smap_age_hours':[1.5]*3,
            'et0_past24h_mm':[4.]*3,'kc_assumed':[.7]*3,'root_depth_m_assumed':[1.]*3,
            'sensor_moisture_raw':[50.]*3,'battery_voltage':[4.1]*3,
            'sensor_valid_indicator':[True]*3, 'validated_irrigation_label':[False]*3,
            'recommendation':['no_action','monitor','review_irrigation']
        })
        engineered=engineer_features(data,pd.DataFrame(columns=['field_id','timestamp']))
        self.assertTrue(np.isnan(engineered.loc[1,'smap_theta_1d_ago']))
        self.assertAlmostEqual(engineered.loc[2,'smap_theta_1d_ago'],.3)
        self.assertAlmostEqual(engineered.loc[1,'smap_theta_3d_ago'] if not np.isnan(engineered.loc[1,'smap_theta_3d_ago']) else 0,0)
        self.assertEqual(engineered[TARGET_NAME].notna().sum(),3)

if __name__=='__main__':
    unittest.main()
