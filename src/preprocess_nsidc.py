from pathlib import Path
import re
import warnings
import h5py
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

NSIDC_DIR = BASE_DIR / "nsidc_data"
OUT_DIR = BASE_DIR / "processed" / "nsidc"

OUT_DIR.mkdir(parents=True, exist_ok=True)

FIELDS = {
    "field_3334": {"latitude": 45.99, "longitude": 20.49},
    "field_3335": {"latitude": 45.98, "longitude": 20.47},
}

DYNAMIC_VARIABLES = [
    "sm_surface", "sm_rootzone", "sm_profile",
    "soil_temp_layer1", "soil_temp_layer2", "soil_temp_layer3",
    "soil_temp_layer4", "soil_temp_layer5", "soil_temp_layer6",
    "land_evapotranspiration_flux", "precipitation_total_surface_flux",
    "soil_water_infiltration_flux", "overland_runoff_flux",
    "leaf_area_index", "vegetation_greenness_fraction",
    "surface_temp", "surface_pressure",
]

STATIC_VARIABLES = [
    "cell_elevation", "cell_land_fraction", "clsm_cdcr1",
    "clsm_cdcr2", "clsm_dzgt1", "clsm_dzgt2",
    "clsm_dzgt3", "clsm_dzgt4", "clsm_dzgt5",
    "clsm_dzgt6", "clsm_dzpr", "clsm_dzrz",
    "clsm_dzsf", "clsm_dztsurf", "clsm_poros",
    "clsm_wp", "clsm_veghght", "mwrtm_clay",
    "mwrtm_sand", "mwrtm_poros", "mwrtm_soilcls",
    "mwrtm_vegcls",
]


def parse_dynamic_timestamp(filename):
    match = re.search(r"(\d{8})T(\d{6})", filename)

    if not match:
        return pd.NaT

    return pd.to_datetime(
        match.group(1) + match.group(2),
        format="%Y%m%d%H%M%S",
        errors="coerce",
    )


def safe_value(dataset, index):
    value = np.asarray(dataset[index]).squeeze()

    try:
        value = float(value)
    except Exception:
        return np.nan

    if not np.isfinite(value):
        return np.nan

    fill_candidates = ["_FillValue", "missing_value", "fmissing_value"]

    for attr in fill_candidates:
        if attr in dataset.attrs:
            fill = dataset.attrs[attr]

            try:
                fill = float(np.asarray(fill).squeeze())

                if np.isclose(value, fill):
                    return np.nan
            except Exception:
                pass

    return value


def find_nearest_cells(h5):
    lats = np.asarray(h5["cell_lat"][:])
    lons = np.asarray(h5["cell_lon"][:])

    result = {}

    for field_id, meta in FIELDS.items():
        distance = (
            (lats - meta["latitude"]) ** 2
            + (lons - meta["longitude"]) ** 2
        )

        flat_index = np.nanargmin(distance)
        index = np.unravel_index(flat_index, distance.shape)

        result[field_id] = {
            "index": index,
            "grid_latitude": float(lats[index]),
            "grid_longitude": float(lons[index]),
        }

    return result


def get_dynamic_files():
    folders = sorted(NSIDC_DIR.glob("SPL4SMGP*"))
    files = []

    for folder in folders:
        files.extend(folder.rglob("*.nc4"))
        files.extend(folder.rglob("*.h5"))

    return sorted(set(files))


def preprocess_dynamic():
    files = get_dynamic_files()

    print(f"Found {len(files)} SPL4SMGP dynamic files")

    if not files:
        raise FileNotFoundError("No SPL4SMGP files found.")

    with h5py.File(files[0], "r") as h5:
        field_cells = find_nearest_cells(h5)

    print("\nNearest SMAP L4 cells:")

    for field_id, info in field_cells.items():
        print(
            field_id,
            "->",
            info["grid_latitude"],
            info["grid_longitude"],
            "index:",
            info["index"],
        )

    rows = []

    for i, file in enumerate(files, start=1):
        if i == 1 or i % 100 == 0 or i == len(files):
            print(f"Dynamic: {i}/{len(files)}")

        timestamp = parse_dynamic_timestamp(file.name)

        if pd.isna(timestamp):
            continue

        try:
            with h5py.File(file, "r") as h5:
                group = h5["Geophysical_Data"]

                for field_id, info in field_cells.items():
                    index = info["index"]

                    row = {
                        "timestamp": timestamp,
                        "field_id": field_id,
                        "smap_grid_latitude": info["grid_latitude"],
                        "smap_grid_longitude": info["grid_longitude"],
                        "source_file": file.name,
                    }

                    for var in DYNAMIC_VARIABLES:
                        if var in group:
                            row[f"smap_{var}"] = safe_value(group[var], index)

                    rows.append(row)

        except Exception as e:
            warnings.warn(f"Could not process {file.name}: {e}")

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["field_id", "timestamp"]).sort_values(
        ["field_id", "timestamp"]
    ).reset_index(drop=True)

    output = OUT_DIR / "smap_l4_3hourly.parquet"
    df.to_parquet(output, index=False)

    print("\nDynamic shape:")
    print(df.shape)

    print("\nDynamic period:")
    print(df["timestamp"].min(), "->", df["timestamp"].max())

    print("\nDynamic rows by field:")
    print(df["field_id"].value_counts())

    print("\nSaved:")
    print(output)

    return df


def get_static_file():
    folders = sorted(NSIDC_DIR.glob("SPL4SMLM*"))
    files = []

    for folder in folders:
        files.extend(folder.rglob("*.nc4"))
        files.extend(folder.rglob("*.h5"))

    files = sorted(set(files))

    if not files:
        return None

    return files[-1]


def preprocess_static():
    file = get_static_file()

    if file is None:
        raise FileNotFoundError("No SPL4SMLM static file found.")

    print("\nReading static file:")
    print(file.name)

    rows = []

    with h5py.File(file, "r") as h5:
        field_cells = find_nearest_cells(h5)
        group = h5["Land-Model-Constants_Data"]

        for field_id, info in field_cells.items():
            index = info["index"]

            row = {
                "field_id": field_id,
                "smap_static_grid_latitude": info["grid_latitude"],
                "smap_static_grid_longitude": info["grid_longitude"],
                "source_file": file.name,
            }

            for var in STATIC_VARIABLES:
                if var in group:
                    row[f"smap_static_{var}"] = safe_value(group[var], index)

            rows.append(row)

    df = pd.DataFrame(rows)

    output = OUT_DIR / "smap_static.parquet"
    df.to_parquet(output, index=False)

    print("\nStatic shape:")
    print(df.shape)

    print("\nStatic data:")
    print(df.T)

    print("\nSaved:")
    print(output)

    return df


def quality_check(dynamic, static):
    print("\n================================")
    print("NSIDC QUALITY CHECK")
    print("================================")

    print("\nDynamic duplicates:")
    print(dynamic.duplicated(subset=["field_id", "timestamp"]).sum())

    print("\nDynamic missingness:")
    print(dynamic.isna().mean().sort_values(ascending=False))

    for variable in ["smap_sm_surface", "smap_sm_rootzone", "smap_sm_profile"]:
        if variable in dynamic.columns:
            print(f"\n{variable}:")
            print(dynamic[variable].describe())

    print("\nStatic missingness:")
    print(static.isna().mean().sort_values(ascending=False))


if __name__ == "__main__":
    print("\n================================")
    print("1. SMAP L4 DYNAMIC")
    print("================================")

    dynamic = preprocess_dynamic()

    print("\n================================")
    print("2. SMAP L4 STATIC")
    print("================================")

    static = preprocess_static()

    quality_check(dynamic, static)

    print("\n================================")
    print("DONE")
    print("================================")

    print("\nNSIDC processed files:")
    print(OUT_DIR / "smap_l4_3hourly.parquet")
    print(OUT_DIR / "smap_static.parquet")
    print(OUT_DIR / "smap_1km_daily.parquet")
