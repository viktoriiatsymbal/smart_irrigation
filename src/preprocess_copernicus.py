from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

BASE_DIR = Path(__file__).resolve().parent

ERA5_DIR = BASE_DIR / "copernicus_data" / "era5_land"
OUT_DIR = BASE_DIR / "processed" / "copernicus"

OUT_DIR.mkdir(parents=True, exist_ok=True)

FIELDS = {
    "field_3334": {"latitude": 45.99, "longitude": 20.49},
    "field_3335": {"latitude": 45.98, "longitude": 20.47},
}


def relative_humidity_from_dewpoint(temperature_c, dewpoint_c):
    a = 17.625
    b = 243.04
    return 100 * np.exp(a * dewpoint_c / (b + dewpoint_c)) / np.exp(
        a * temperature_c / (b + temperature_c)
    )


def read_era5_file(file):
    ds = xr.open_dataset(file, engine="h5netcdf")

    if "valid_time" in ds.coords:
        time_name = "valid_time"
    elif "time" in ds.coords:
        time_name = "time"
    else:
        raise ValueError(f"No time coordinate in {file.name}")

    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"

    frames = []

    for field_id, meta in FIELDS.items():
        point = ds.sel(
            {
                lat_name: meta["latitude"],
                lon_name: meta["longitude"],
            },
            method="nearest",
        )

        df = point.to_dataframe().reset_index()
        df["timestamp"] = pd.to_datetime(df[time_name], errors="coerce")
        df["field_id"] = field_id

        if lat_name in df.columns:
            df["era5_grid_latitude"] = df[lat_name]

        if lon_name in df.columns:
            df["era5_grid_longitude"] = df[lon_name]

        rename = {var: f"era5_{var}" for var in ds.data_vars if var in df.columns}
        df = df.rename(columns=rename)

        wanted = [
            "timestamp",
            "field_id",
            "era5_grid_latitude",
            "era5_grid_longitude",
        ] + list(rename.values())

        wanted = [c for c in wanted if c in df.columns]
        frames.append(df[wanted])

    ds.close()

    return pd.concat(frames, ignore_index=True)


def preprocess_era5():
    files = sorted(ERA5_DIR.glob("*.nc"))

    if not files:
        raise FileNotFoundError("No ERA5 .nc files found.")

    print(f"Found {len(files)} ERA5 files")

    parts = []

    for i, file in enumerate(files, start=1):
        print(f"{i}/{len(files)} {file.name}")
        parts.append(read_era5_file(file))

    combined = pd.concat(parts, ignore_index=True, sort=False)

    def first_valid(series):
        non_null = series.dropna()
        if len(non_null) == 0:
            return np.nan
        return non_null.iloc[0]

    value_cols = [
        col for col in combined.columns if col not in ["timestamp", "field_id"]
    ]

    final = combined.groupby(["timestamp", "field_id"], as_index=False).first()
    final = final.sort_values(["field_id", "timestamp"]).reset_index(drop=True)

    kelvin_vars = [
        "era5_t2m",
        "era5_d2m",
        "era5_stl1",
        "era5_stl2",
        "era5_stl3",
        "era5_stl4",
    ]

    for var in kelvin_vars:
        if var in final.columns:
            final[var + "_c"] = final[var] - 273.15

    if "era5_tp" in final.columns:
        final["era5_tp_mm"] = final["era5_tp"] * 1000

    if "era5_sp" in final.columns:
        final["era5_sp_hpa"] = final["era5_sp"] / 100

    if "era5_u10" in final.columns and "era5_v10" in final.columns:
        final["era5_wind_speed_10m"] = np.sqrt(
            final["era5_u10"] ** 2 + final["era5_v10"] ** 2
        )

    if "era5_t2m_c" in final.columns and "era5_d2m_c" in final.columns:
        final["era5_relative_humidity"] = relative_humidity_from_dewpoint(
            final["era5_t2m_c"], final["era5_d2m_c"]
        )

    return final


if __name__ == "__main__":
    era5 = preprocess_era5()

    output = OUT_DIR / "era5_hourly.parquet"
    era5.to_parquet(output, index=False)

    print("\nFinal ERA5 shape:")
    print(era5.shape)

    print("\nPeriod:")
    print(era5["timestamp"].min(), "->", era5["timestamp"].max())

    print("\nRows per field:")
    print(era5["field_id"].value_counts())

    print("\nSaved:")
    print(output)
