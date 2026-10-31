import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import IsolationForest

pd.set_option("display.width", 200); pd.set_option("display.max_columns", 40)

DATA = "irrigation_ml_dataset.csv"        # parquet теж ок, але потрібен pyarrow
df = (pd.read_csv(DATA, parse_dates=["timestamp"])
        .sort_values(["field_id", "timestamp"])
        .drop_duplicates(["field_id", "timestamp"])
        .reset_index(drop=True))

SM, ST, AT, AH, AP, PR = ("soil_moisture_sensor", "soil_temperature_sensor",
                          "air_temperature_sensor", "air_humidity_sensor",
                          "air_pressure_sensor", "precipitation_sensor")