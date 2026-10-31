"""Shared, cached loaders for the Streamlit apps. Run from the repo root: streamlit run app/farmer_app.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st  # noqa: E402

from irrigation_ml.config import ARTIFACTS  # noqa: E402


def artifacts_ready() -> bool:
    return all((ARTIFACTS / f).exists() for f in ("forecast_models.joblib", "risk_models.joblib", "calibration.json"))


@st.cache_resource(show_spinner="Loading data and models...")
def prepared():
    from irrigation_ml.forecast import prepare
    return prepare()


@st.cache_data(show_spinner="Computing recommendation...")
def recommendations(at: str, c_fn: float, system: str):
    from irrigation_ml.decision import Policy
    from irrigation_ml.recommend import recommend
    return recommend(at, Policy(c_fn=c_fn), system, prepared())


@st.cache_data(show_spinner=False)
def raw_sensor():
    """Raw probe readings, including the ones the pipeline rejects (for showing untrusted readings on the farmer chart)."""
    import pandas as pd
    return pd.read_parquet(ROOT / "data" / "irrigation_ml_dataset.parquet", columns=["field_id", "timestamp", "soil_moisture_sensor"])
