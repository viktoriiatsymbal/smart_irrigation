"""Export honest PROXY daily feature/label files; no fabricated true irrigation actions."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from train_decision_policy import load_source, historical_snapshots, engineer_features, FEATURES, TARGET_NAME, CLASSES


def export(raw: pd.DataFrame, out: Path, reuse_teacher: Path | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    teacher = pd.read_csv(reuse_teacher, parse_dates=['decision_timestamp']) if reuse_teacher and reuse_teacher.exists() else historical_snapshots(raw)
    f = engineer_features(teacher,raw)
    f['date'] = f.decision_timestamp.dt.strftime('%Y-%m-%d')
    feature_table=f[['field_id','date','decision_timestamp',*FEATURES]].copy()
    labels=f[['field_id','date','decision_timestamp',TARGET_NAME,'recommendation','assumption_robustness',
              'validated_irrigation_label']].copy()
    labels['review_irrigation_binary_PROXY'] = np.where(labels[TARGET_NAME].notna(),
                                             (labels[TARGET_NAME]=='review_irrigation').astype(int), np.nan)
    labels=labels.rename(columns={TARGET_NAME:'decision_label_PROXY_NOT_VALIDATED',
                                  'recommendation':'physics_rule_action'})
    feature_table.to_csv(out/'daily_decision_table_PROXY.csv',index=False)
    labels.to_csv(out/'decision_labels_daily_PROXY_NOT_VALIDATED.csv',index=False)
    print('Daily features:',len(feature_table),' | 3-class proxy labels:',labels.decision_label_PROXY_NOT_VALIDATED.notna().sum())
    print('Unavailable/out-of-season labels left as NaN:',labels.decision_label_PROXY_NOT_VALIDATED.isna().sum())

if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--data',type=Path,required=True)
    a.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'outputs')
    a.add_argument('--teacher',type=Path,default=None)
    cfg=a.parse_args()
    export(load_source(cfg.data),cfg.output,cfg.teacher)
