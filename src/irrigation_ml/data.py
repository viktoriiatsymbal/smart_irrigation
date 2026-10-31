import pandas as pd

from .config import PROCESSED, HORIZONS


def load_hourly() -> pd.DataFrame:
    """Hourly table reindexed to a complete hourly grid per field, so shifts are exactly t+H hours."""
    h = pd.read_parquet(PROCESSED / "hourly_model_table.parquet")
    parts = []
    for fid, g in h.groupby("field_id"):
        g = g.set_index("timestamp").sort_index()
        g = g[~g.index.duplicated()]
        full = pd.date_range(g.index.min(), g.index.max(), freq="h", name="timestamp")
        g = g.reindex(full)
        g["field_id"] = fid
        g["crop"] = g["crop"].ffill().bfill()
        parts.append(g.reset_index())
    return pd.concat(parts, ignore_index=True)


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """y_h = sm_clean at exactly t+H (NaN where the future reading was masked as untrustworthy)."""
    df = df.copy()
    for H in HORIZONS:
        df[f"y{H}"] = df.groupby("field_id")["sm_clean"].shift(-H)
    return df
