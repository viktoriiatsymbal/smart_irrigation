"""Historical hybrid advisory: rule engine is authoritative; ML is diagnostic.

Uses past-only observed weather; no implied future-forecast availability.
NO automatic irrigation action, and NO validated field-truth labels.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from irrigation_recommendations import make_decisions
from train_decision_policy import engineer_features, predict_proba, assign_actions, load_source, FEATURES


def predict_historical_advisory(raw: pd.DataFrame, model_bundle: dict) -> pd.DataFrame:
    teacher = make_decisions(raw)
    d = engineer_features(teacher, raw)
    mdl = model_bundle['model']
    threshold = float(model_bundle['risk_threshold'])
    if list(model_bundle['features']) != FEATURES:
        raise ValueError('Incompatible model feature schema')
    inseason = d.recommendation.isin(['no_action','monitor','review_irrigation'])
    out = d[['field_id','decision_timestamp','crop','smap_theta_rootzone_m3m3',
             'recommendation','confidence','reason','assumption_robustness',
             'validated_irrigation_label']].copy()
    out = out.rename(columns={'recommendation':'physics_rule_action',
                              'confidence':'physics_rule_confidence'})
    out['ml_proxy_action']=pd.Series(index=d.index,dtype='object')
    for label in ('no_action','monitor','review_irrigation'):
        out[f'ml_proxy_probability_{label}']=np.nan
    out['review_probability_threshold_2024_only']=threshold
    if inseason.any():
        probs = predict_proba(mdl,d.loc[inseason])
        out.loc[inseason,'ml_proxy_action']=assign_actions(probs, threshold)
        for label in ('no_action','monitor','review_irrigation'):
            out.loc[inseason,f'ml_proxy_probability_{label}']=probs[label].to_numpy()
    out['ml_agrees_with_rule']=np.where(inseason,out.ml_proxy_action.eq(out.physics_rule_action), pd.NA)
    # Conservative, non-actuating decision: always fall back to the auditable
    # deterministic physical rule. ML disagreement requires manual inspection.
    out['hybrid_advisory_status']=np.select([
        d.recommendation.eq('insufficient_data'),
        d.recommendation.eq('no_action_out_of_season'),
        inseason & (out.ml_proxy_action!=out.physics_rule_action),
        inseason & d.assumption_robustness.eq('assumption_sensitive'),
        inseason,
    ],['insufficient_data','out_of_assumed_season','manual_review_model_disagrees',
       'manual_review_soil_uncertainty','rule_and_model_agree_unvalidated'],default='insufficient_data')
    out['not_an_automatic_irrigation_command']=True
    out['ml_is_in_sample_for_some_rows_not_evaluation']=True
    out['historical_only_no_live_forecast']=True
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True,type=Path)
    p.add_argument('--model',type=Path,default=Path(__file__).resolve().parent/'models'/'irrigation_policy_proxy_MULTICLASS_UNVERIFIED.joblib')
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'outputs'/'hybrid_advisory_historical.csv')
    a=p.parse_args()
    out=predict_historical_advisory(load_source(a.data),joblib.load(a.model))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    out.to_csv(a.output,index=False)
    print('Historical advisory only. ML trained on physical-rule proxies; not validated field labels.')
    print(out.hybrid_advisory_status.value_counts(dropna=False).to_string())
    print(out.sort_values('decision_timestamp').groupby('field_id', observed=True).tail(1)[[
      'field_id','decision_timestamp','physics_rule_action','ml_proxy_action','hybrid_advisory_status'
    ]].to_string(index=False))

if __name__=='__main__':
    main()
