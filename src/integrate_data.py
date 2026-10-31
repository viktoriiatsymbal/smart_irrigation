from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

ZENODO_DIR = BASE_DIR / "processed" / "zenodo"
COPERNICUS_DIR = BASE_DIR / "processed" / "copernicus"
NSIDC_DIR = BASE_DIR / "processed" / "nsidc"
FINAL_DIR = BASE_DIR / "final"

FINAL_DIR.mkdir(parents=True, exist_ok=True)

print("Loading processed sources...")

sensors = pd.read_parquet(ZENODO_DIR / "sensors_hourly.parquet")
openmeteo = pd.read_parquet(ZENODO_DIR / "openmeteo_hourly.parquet")
era5 = pd.read_parquet(COPERNICUS_DIR / "era5_hourly.parquet")
smap = pd.read_parquet(NSIDC_DIR / "smap_l4_3hourly.parquet")
smap_static = pd.read_parquet(NSIDC_DIR / "smap_static.parquet")
smap_1km = pd.read_parquet(NSIDC_DIR / "smap_1km_daily.parquet")


def normalize_datetime_ns(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").astype("datetime64[ns]")
    return df


sensors = normalize_datetime_ns(sensors, ["timestamp"])
openmeteo = normalize_datetime_ns(openmeteo, ["timestamp"])
era5 = normalize_datetime_ns(era5, ["timestamp"])
smap = normalize_datetime_ns(smap, ["timestamp"])
smap_1km = normalize_datetime_ns(smap_1km, ["date"])

final = sensors.copy()

print("Merging OpenMeteo...")

openmeteo_merge = openmeteo.drop(columns=["latitude", "longitude"], errors="ignore")
final = final.merge(openmeteo_merge, on=["field_id", "timestamp"], how="left")

print("Merging ERA5...")

final = final.merge(era5, on=["field_id", "timestamp"], how="left")

print("Merging SMAP L4...")

merged_fields = []

for field_id in final["field_id"].unique():
    left = final[final["field_id"] == field_id].sort_values("timestamp").copy()
    right = smap[smap["field_id"] == field_id].sort_values("timestamp").copy()

    if right.empty:
        merged_fields.append(left)
        continue

    right = right.rename(columns={"timestamp": "smap_timestamp"})

    merged = pd.merge_asof(
        left,
        right,
        left_on="timestamp",
        right_on="smap_timestamp",
        direction="backward",
        tolerance=pd.Timedelta("3h"),
        suffixes=("", "_smap_duplicate"),
    )

    if "field_id_smap_duplicate" in merged.columns:
        merged = merged.drop(columns=["field_id_smap_duplicate"])

    merged["smap_age_hours"] = (
        merged["timestamp"] - merged["smap_timestamp"]
    ).dt.total_seconds() / 3600

    merged_fields.append(merged)

final = pd.concat(merged_fields, ignore_index=True)

print("Merging SMAP static...")

static = smap_static.drop(columns=["source_file"], errors="ignore")
final = final.merge(static, on="field_id", how="left")

print("Merging SMAP 1km...")

final["date"] = final["timestamp"].dt.normalize()

smap_1km_merge = smap_1km.drop(columns=["source_file"], errors="ignore")

final = final.merge(
    smap_1km_merge,
    on=["field_id", "date"],
    how="left",
)

final["has_sensor_target"] = final["soil_moisture_sensor"].notna()

era5_cols = [c for c in final.columns if c.startswith("era5_")]
final["has_era5"] = final[era5_cols].notna().any(axis=1)

smap_cols = [
    c
    for c in final.columns
    if c.startswith("smap_")
    and not c.startswith("smap_static_")
    and not c.startswith("smap_1km_")
]

if smap_cols:
    final["has_smap_l4"] = final[smap_cols].notna().any(axis=1)

openmeteo_cols = [c for c in final.columns if c.startswith("openmeteo_")]

if openmeteo_cols:
    final["has_openmeteo"] = final[openmeteo_cols].notna().any(axis=1)

final = final.sort_values(["field_id", "timestamp"]).reset_index(drop=True)

report_rows = []

for field_id, df in final.groupby("field_id"):
    row = {
        "field_id": field_id,
        "start": df["timestamp"].min(),
        "end": df["timestamp"].max(),
        "rows": len(df),
        "target_available_pct": 100 * df["soil_moisture_sensor"].notna().mean(),
        "missing_sensor_record_pct": 100 * df["missing_sensor_record"].fillna(True).mean(),
        "quality_flag_count": int(df["any_quality_flag"].fillna(False).sum()),
        "era5_coverage_pct": 100 * df["has_era5"].mean(),
    }

    if "has_smap_l4" in df.columns:
        row["smap_l4_coverage_pct"] = 100 * df["has_smap_l4"].mean()

    if "has_openmeteo" in df.columns:
        row["openmeteo_coverage_pct"] = 100 * df["has_openmeteo"].mean()

    report_rows.append(row)

quality_report = pd.DataFrame(report_rows)

parquet_path = FINAL_DIR / "irrigation_ml_dataset.parquet"
final.to_parquet(parquet_path, index=False)

csv_df = final.copy()
csv_df["timestamp"] = csv_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

if "smap_timestamp" in csv_df.columns:
    csv_df["smap_timestamp"] = pd.to_datetime(
        csv_df["smap_timestamp"], errors="coerce"
    ).dt.strftime("%Y-%m-%d %H:%M:%S")

csv_path = FINAL_DIR / "irrigation_ml_dataset.csv"
csv_df.to_csv(csv_path, index=False)

quality_report.to_csv(FINAL_DIR / "data_quality_report.csv", index=False)

dictionary = pd.DataFrame({
    "column": final.columns,
    "source": [
        (
            "Zenodo sensor"
            if c.endswith("_sensor") or c in ["crop", "station_id"]
            else "OpenMeteo"
            if c.startswith("openmeteo_")
            else "ERA5-Land"
            if c.startswith("era5_")
            else "SMAP static"
            if c.startswith("smap_static_")
            else "SMAP 1km"
            if c.startswith("smap_1km_")
            else "SMAP L4"
            if c.startswith("smap_")
            else "Integration metadata"
        )
        for c in final.columns
    ],
})

dictionary.to_csv(FINAL_DIR / "data_dictionary.csv", index=False)

print("\n================================")
print("INTEGRATION DONE")
print("================================")

print("\nFinal shape:")
print(final.shape)

print("\nFields:")
print(final["field_id"].value_counts())

print("\nPeriod:")
print(final["timestamp"].min(), "->", final["timestamp"].max())

print("\nQuality report:")
print(quality_report)

print("\nCreated:")
print(parquet_path)
print(csv_path)
print(FINAL_DIR / "data_quality_report.csv")
print(FINAL_DIR / "data_dictionary.csv")
