from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat
from tqdm import tqdm


def _safe_array(x):
    try:
        return np.asarray(x, dtype=float).flatten()
    except Exception:
        return np.array([])


def _safe_float(x):
    try:
        arr = np.asarray(x, dtype=float).flatten()
        return float(arr[0]) if len(arr) > 0 else np.nan
    except Exception:
        return np.nan


def load_battery(mat_path):
    """
    Load one NASA PCoE battery .mat file and return discharge + impedance dataframes.
    """

    mat_path = Path(mat_path)
    cell_id = mat_path.stem

    mat = loadmat(mat_path, simplify_cells=True)
    battery = mat[cell_id]
    cycles = battery["cycle"]

    discharge_rows = []
    impedance_rows = []
    discharge_counter = 0

    for idx, cycle in enumerate(cycles):
        cycle_type = cycle.get("type")
        ambient_temp = cycle.get("ambient_temperature", np.nan)
        data = cycle.get("data", {})

        if cycle_type == "discharge":
            capacity = _safe_float(data.get("Capacity"))

            if np.isnan(capacity) or capacity < 0.5:
                continue

            voltage = _safe_array(data.get("Voltage_measured"))
            current = _safe_array(data.get("Current_measured"))
            temp = _safe_array(data.get("Temperature_measured"))
            time = _safe_array(data.get("Time"))

            if len(voltage) == 0 or len(time) == 0:
                continue

            discharge_counter += 1

            discharge_rows.append({
                "cell_id": cell_id,
                "discharge_cycle": discharge_counter,
                "cycle_index": idx,
                "capacity_ah": capacity,
                "voltage_start": voltage[0],
                "voltage_end": voltage[-1],
                "voltage_min": np.nanmin(voltage),
                "voltage_mean": np.nanmean(voltage),
                "current_mean": np.nanmean(current) if len(current) else np.nan,
                "temp_max": np.nanmax(temp) if len(temp) else np.nan,
                "temp_mean": np.nanmean(temp) if len(temp) else np.nan,
                "temp_start": temp[0] if len(temp) else np.nan,
                "duration_s": time[-1] - time[0] if len(time) else np.nan,
                "ambient_temp": ambient_temp,
                "raw_voltage": voltage.tolist(),
                "raw_current": current.tolist(),
                "raw_temp": temp.tolist(),
                "raw_time": time.tolist(),
            })

        elif cycle_type == "impedance":
            impedance_rows.append({
                "cell_id": cell_id,
                "cycle_index": idx,
                "ambient_temp": ambient_temp,
                "Re": _safe_float(data.get("Re")),
                "Rct": _safe_float(data.get("Rct")),
            })

    return {
        "discharge": pd.DataFrame(discharge_rows),
        "impedance": pd.DataFrame(impedance_rows),
    }


def summarise_cell(result):
    df = result["discharge"]

    if df.empty:
        print("No discharge data found.")
        return

    first_capacity = df["capacity_ah"].iloc[0]
    last_capacity = df["capacity_ah"].iloc[-1]
    retention = last_capacity / first_capacity

    print(f"Cell ID: {df['cell_id'].iloc[0]}")
    print(f"Discharge cycles: {len(df)}")
    print(f"First capacity: {first_capacity:.3f} Ah")
    print(f"Last capacity: {last_capacity:.3f} Ah")
    print(f"Final retention: {retention:.2%}")
    print(f"EoL reached: {retention <= 0.70}")


def load_all_batteries(raw_dir="data/raw", interim_dir="data/interim", overwrite=False):
    raw_dir = Path(raw_dir)
    interim_dir = Path(interim_dir)
    interim_dir.mkdir(parents=True, exist_ok=True)

    all_discharge = []
    all_impedance = []

    mat_files = sorted(raw_dir.rglob("*.mat"))

    if not mat_files:
        raise FileNotFoundError(f"No .mat files found in {raw_dir}")

    for mat_file in tqdm(mat_files, desc="Loading batteries"):

        # Create unique name using folder path + file name
        # Example:
        # data/raw/Batch_1/B0005.mat -> Batch_1_B0005
        relative_path = mat_file.relative_to(raw_dir)
        safe_name = "_".join(relative_path.with_suffix("").parts)

        discharge_path = interim_dir / f"{safe_name}_discharge.parquet"
        impedance_path = interim_dir / f"{safe_name}_impedance.parquet"

        if discharge_path.exists() and not overwrite:
            discharge_df = pd.read_parquet(discharge_path)

            impedance_df = (
                pd.read_parquet(impedance_path)
                if impedance_path.exists()
                else pd.DataFrame()
            )

        else:
            result = load_battery(mat_file)
            discharge_df = result["discharge"]
            impedance_df = result["impedance"]

            # Force unique cell_id to avoid duplicate names like B0005
            if not discharge_df.empty:
                discharge_df["cell_id"] = safe_name
                discharge_df["source_file"] = str(mat_file)

            if not impedance_df.empty:
                impedance_df["cell_id"] = safe_name
                impedance_df["source_file"] = str(mat_file)

            discharge_df.to_parquet(discharge_path, index=False)

            if not impedance_df.empty:
                impedance_df.to_parquet(impedance_path, index=False)

        all_discharge.append(discharge_df)

        if not impedance_df.empty:
            all_impedance.append(impedance_df)

    combined_discharge = pd.concat(all_discharge, ignore_index=True)

    combined_impedance = (
        pd.concat(all_impedance, ignore_index=True)
        if all_impedance
        else pd.DataFrame()
    )

    combined_discharge.to_parquet(
        interim_dir / "all_discharge.parquet",
        index=False
    )

    if not combined_impedance.empty:
        combined_impedance.to_parquet(
            interim_dir / "all_impedance.parquet",
            index=False
        )

    return {
        "discharge": combined_discharge,
        "impedance": combined_impedance,
    }

if __name__ == "__main__":
    result = load_all_batteries()
    print("\nLoaded discharge data:")
    print(result["discharge"].head())
    print(result["discharge"].shape)

    print("\nLoaded impedance data:")
    print(result["impedance"].head())
    print(result["impedance"].shape)