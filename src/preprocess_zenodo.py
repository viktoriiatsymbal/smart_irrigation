from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

ZENODO_DIR = BASE_DIR / "zenodo_data"
OUT_DIR = BASE_DIR / "processed" / "zenodo"

OUT_DIR.mkdir(parents=True, exist_ok=True)

FIELD_META = {
    "3233333332303334": {"field_id": "field_3334", "latitude": 45.99, "longitude": 20.49},
    "3233333332303335": {"field_id": "field_3335", "latitude": 45.98, "longitude": 20.47},
}

CROP_MAP = {
    (2024, "3233333332303334"): "sorghum",
    (2024, "3233333332303335"): "corn",
    (2025, "3233333332303334"): "corn",
    (2025, "3233333332303335"): "corn",
}


def normalize_station_id(value):
    value = str(value).strip()
    if value.endswith(".0"):
        value = value[:-2]
    return value


def clean_column_name(name):
    return str(name).strip().replace(" ", "").replace("_", "").lower()


def load_sensor_data():
    files = sorted(ZENODO_DIR.glob("P18_BIORO_WeatherStationData_AgriDataValue_*.xlsx"))

    if not files:
        raise FileNotFoundError("No AgriDataValue sensor Excel files found.")

    rename_map = {
        "airtemperature": "air_temperature_sensor",
        "airhumidity": "air_humidity_sensor",
        "airpressure": "air_pressure_sensor",
        "dewpoint": "dew_point_sensor",
        "precipitation": "precipitation_sensor",
        "earthhumidity1": "soil_moisture_sensor",
        "earthtemperature1": "soil_temperature_sensor",
        "batteryvoltage": "battery_voltage",
        "solarpanelvoltage": "solar_panel_voltage",
    }

    frames = []

    for file in files:
        year_match = re.search(r"(2024|2025)", file.name)

        if not year_match:
            continue

        year = int(year_match.group(1))
        xls = pd.ExcelFile(file)

        for sheet in xls.sheet_names:
            station_id = normalize_station_id(sheet)

            if station_id not in FIELD_META:
                warnings.warn(f"Unknown station sheet: {sheet}")
                continue

            df = pd.read_excel(file, sheet_name=sheet)
            df.columns = [clean_column_name(c) for c in df.columns]

            if "communicateat" in df.columns:
                df["timestamp"] = pd.to_datetime(df["communicateat"], errors="coerce")
            elif "date" in df.columns and "time" in df.columns:
                df["timestamp"] = pd.to_datetime(
                    df["date"].astype(str) + " " + df["time"].astype(str),
                    errors="coerce",
                )
            else:
                raise ValueError(f"No timestamp found in {file.name}, sheet {sheet}")

            rename = {col: rename_map[col] for col in df.columns if col in rename_map}
            df = df.rename(columns=rename)

            meta = FIELD_META[station_id]

            df["station_id"] = station_id
            df["field_id"] = meta["field_id"]
            df["latitude"] = meta["latitude"]
            df["longitude"] = meta["longitude"]
            df["year"] = year
            df["crop"] = CROP_MAP[(year, station_id)]

            wanted = [
                "timestamp", "station_id", "field_id", "latitude", "longitude", "year", "crop",
                "air_temperature_sensor", "air_humidity_sensor", "air_pressure_sensor",
                "dew_point_sensor", "precipitation_sensor", "soil_moisture_sensor",
                "soil_temperature_sensor", "battery_voltage", "solar_panel_voltage",
            ]

            for col in wanted:
                if col not in df.columns:
                    df[col] = np.nan

            df = df[wanted]

            numeric_cols = [
                "air_temperature_sensor", "air_humidity_sensor", "air_pressure_sensor",
                "dew_point_sensor", "precipitation_sensor", "soil_moisture_sensor",
                "soil_temperature_sensor", "battery_voltage", "solar_panel_voltage",
            ]

            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            frames.append(df)

    sensors = pd.concat(frames, ignore_index=True)
    sensors = sensors.dropna(subset=["timestamp"])

    sensors["duplicate_timestamp"] = sensors.duplicated(
        subset=["station_id", "timestamp"], keep=False
    )

    sensors = sensors.sort_values(["station_id", "timestamp"]).drop_duplicates(
        subset=["station_id", "timestamp"], keep="first"
    ).reset_index(drop=True)

    sensors["flag_soil_temperature_range"] = (
        (sensors["soil_temperature_sensor"] < -20)
        | (sensors["soil_temperature_sensor"] > 70)
    )

    sensors["flag_air_temperature_range"] = (
        (sensors["air_temperature_sensor"] < -50)
        | (sensors["air_temperature_sensor"] > 60)
    )

    sensors["flag_air_humidity_range"] = (
        (sensors["air_humidity_sensor"] < 0)
        | (sensors["air_humidity_sensor"] > 100)
    )

    sensors["flag_soil_moisture_range"] = (
        (sensors["soil_moisture_sensor"] < 0)
        | (sensors["soil_moisture_sensor"] > 100)
    )

    sensors["flag_negative_precipitation"] = sensors["precipitation_sensor"] < 0

    flag_cols = [
        "flag_soil_temperature_range", "flag_air_temperature_range",
        "flag_air_humidity_range", "flag_soil_moisture_range",
        "flag_negative_precipitation",
    ]

    sensors["any_quality_flag"] = sensors[flag_cols].any(axis=1)

    return sensors


def build_hourly_sensor_table(sensors):
    continuous_cols = [
        "air_temperature_sensor", "air_humidity_sensor", "air_pressure_sensor",
        "dew_point_sensor", "soil_moisture_sensor", "soil_temperature_sensor",
        "battery_voltage", "solar_panel_voltage",
    ]

    frames = []

    for (station_id, year), group in sensors.groupby(["station_id", "year"]):
        group = group.copy()
        group["timestamp_hour"] = group["timestamp"].dt.floor("h")

        agg = {col: "median" for col in continuous_cols}

        agg.update({
            "precipitation_sensor": "sum",
            "field_id": "first",
            "latitude": "first",
            "longitude": "first",
            "crop": "first",
            "duplicate_timestamp": "max",
            "flag_soil_temperature_range": "max",
            "flag_air_temperature_range": "max",
            "flag_air_humidity_range": "max",
            "flag_soil_moisture_range": "max",
            "flag_negative_precipitation": "max",
            "any_quality_flag": "max",
        })

        hourly = group.groupby("timestamp_hour", as_index=False).agg(agg).rename(
            columns={"timestamp_hour": "timestamp"}
        )

        full_index = pd.date_range(
            start=hourly["timestamp"].min(),
            end=hourly["timestamp"].max(),
            freq="1h",
        )

        hourly = hourly.set_index("timestamp").reindex(full_index).rename_axis(
            "timestamp"
        ).reset_index()

        flag_cols = [
            "duplicate_timestamp", "flag_soil_temperature_range",
            "flag_air_temperature_range", "flag_air_humidity_range",
            "flag_soil_moisture_range", "flag_negative_precipitation",
            "any_quality_flag",
        ]

        for col in flag_cols:
            hourly[col] = hourly[col].fillna(False).astype(bool)

        meta = FIELD_META[station_id]

        hourly["station_id"] = station_id
        hourly["field_id"] = meta["field_id"]
        hourly["latitude"] = meta["latitude"]
        hourly["longitude"] = meta["longitude"]
        hourly["year"] = year
        hourly["crop"] = CROP_MAP[(year, station_id)]

        hourly["missing_sensor_record"] = hourly[
            ["soil_moisture_sensor", "soil_temperature_sensor", "air_temperature_sensor"]
        ].isna().all(axis=1)

        frames.append(hourly)

    result = pd.concat(frames, ignore_index=True)
    return result.sort_values(["field_id", "timestamp"]).reset_index(drop=True)


def load_openmeteo():
    files = sorted(ZENODO_DIR.glob("openmeteo*.xlsx"))
    frames = []

    for file in files:
        df = pd.read_excel(file)

        if "date" not in df.columns:
            warnings.warn(f"No date column in {file.name}")
            continue

        df["timestamp"] = pd.to_datetime(df["date"], errors="coerce")

        if "4599_2049" in file.name:
            field_id = "field_3334"
            latitude = 45.99
            longitude = 20.49
        elif "4598_2047" in file.name:
            field_id = "field_3335"
            latitude = 45.98
            longitude = 20.47
        else:
            warnings.warn(f"Unknown coordinates in {file.name}")
            continue

        rename = {
            col: f"openmeteo_{col}"
            for col in df.columns
            if col not in ["date", "timestamp"]
        }

        df = df.rename(columns=rename)

        df["field_id"] = field_id
        df["latitude"] = latitude
        df["longitude"] = longitude

        df = df.drop(columns=["date"], errors="ignore")
        frames.append(df)

    result = pd.concat(frames, ignore_index=True)
    result = result.drop_duplicates(subset=["field_id", "timestamp"], keep="first")

    return result.sort_values(["field_id", "timestamp"]).reset_index(drop=True)


if __name__ == "__main__":
    print("Reading Zenodo sensors...")

    sensors_raw = load_sensor_data()
    print("Raw sensor rows:", len(sensors_raw))

    sensors_hourly = build_hourly_sensor_table(sensors_raw)
    print("Hourly sensor rows:", len(sensors_hourly))

    sensors_hourly.to_parquet(OUT_DIR / "sensors_hourly.parquet", index=False)

    print("Reading OpenMeteo...")

    openmeteo = load_openmeteo()
    openmeteo.to_parquet(OUT_DIR / "openmeteo_hourly.parquet", index=False)

    print("\nSensor period:")
    print(sensors_hourly["timestamp"].min(), "->", sensors_hourly["timestamp"].max())

    print("\nRows by field:")
    print(sensors_hourly["field_id"].value_counts())

    print("\nMissing target:")
    print(
        sensors_hourly["soil_moisture_sensor"]
        .isna()
        .groupby(sensors_hourly["field_id"])
        .mean()
    )

    print("\nDONE")
    print(OUT_DIR / "sensors_hourly.parquet")
    print(OUT_DIR / "openmeteo_hourly.parquet")
