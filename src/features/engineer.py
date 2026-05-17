from pathlib import Path
import numpy as np
import pandas as pd


def compute_capacity_features(df):
    df = df.sort_values(["cell_id", "discharge_cycle"]).copy()

    df["initial_capacity_ah"] = df.groupby("cell_id")["capacity_ah"].transform("first")

    df["capacity_retention"] = df["capacity_ah"] / df["initial_capacity_ah"]

    df["capacity_fade"] = 1 - df["capacity_retention"]

    df["capacity_diff"] = df.groupby("cell_id")["capacity_ah"].diff()

    df["capacity_fade_rate"] = (
        df.groupby("cell_id")["capacity_retention"].diff()
    )

    df["capacity_roll_mean_5"] = (
        df.groupby("cell_id")["capacity_ah"]
        .rolling(window=5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    return df


def compute_rul_features(df, eol_threshold=0.70):
    df = df.sort_values(["cell_id", "discharge_cycle"]).copy()

    failure_cycles = {}

    for cell_id, group in df.groupby("cell_id"):
        failed = group[group["capacity_retention"] <= eol_threshold]

        if not failed.empty:
            failure_cycle = failed["discharge_cycle"].iloc[0]
        else:
            failure_cycle = group["discharge_cycle"].max()

        failure_cycles[cell_id] = failure_cycle

    df["failure_cycle"] = df["cell_id"].map(failure_cycles)
    df["rul"] = df["failure_cycle"] - df["discharge_cycle"]
    df["eol_reached"] = df["capacity_retention"] <= eol_threshold

    return df


def _time_to_voltage(raw_time, raw_voltage, threshold=3.0):
    try:
        time = np.asarray(raw_time, dtype=float)
        voltage = np.asarray(raw_voltage, dtype=float)

        below = np.where(voltage <= threshold)[0]

        if len(below) == 0:
            return np.nan

        return float(time[below[0]])
    except Exception:
        return np.nan


def _voltage_slope(raw_time, raw_voltage):
    try:
        time = np.asarray(raw_time, dtype=float)
        voltage = np.asarray(raw_voltage, dtype=float)

        if len(time) < 2 or len(voltage) < 2:
            return np.nan

        return float((voltage[-1] - voltage[0]) / (time[-1] - time[0]))
    except Exception:
        return np.nan


def compute_voltage_features(df):
    df = df.copy()

    df["time_to_3v"] = df.apply(
        lambda row: _time_to_voltage(row["raw_time"], row["raw_voltage"], threshold=3.0),
        axis=1
    )

    df["voltage_slope"] = df.apply(
        lambda row: _voltage_slope(row["raw_time"], row["raw_voltage"]),
        axis=1
    )

    return df


def compute_temperature_features(df):
    df = df.copy()

    df["temp_rise"] = df["temp_max"] - df["temp_start"]

    df["temp_roll_mean_5"] = (
        df.groupby("cell_id")["temp_max"]
        .rolling(window=5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    return df

def merge_impedance(discharge_df, impedance_df):
    if impedance_df is None or impedance_df.empty:
        discharge_df["Re"] = np.nan
        discharge_df["Rct"] = np.nan
        return discharge_df

    discharge_df = discharge_df.sort_values(
        ["cell_id", "cycle_index"]
    ).copy()

    impedance_df = impedance_df.sort_values(
        ["cell_id", "cycle_index"]
    ).copy()

    merged_frames = []

    for cell_id in discharge_df["cell_id"].unique():

        dcell = discharge_df[
            discharge_df["cell_id"] == cell_id
        ].copy()

        icell = impedance_df[
            impedance_df["cell_id"] == cell_id
        ].copy()

        if icell.empty:
            dcell["Re"] = np.nan
            dcell["Rct"] = np.nan
            merged_frames.append(dcell)
            continue

        merged = pd.merge_asof(
            dcell.sort_values("cycle_index"),
            icell[["cycle_index", "Re", "Rct"]].sort_values("cycle_index"),
            on="cycle_index",
            direction="backward"
        )

        merged_frames.append(merged)

    return pd.concat(merged_frames, ignore_index=True)


def build_features(
    discharge_path="data/interim/all_discharge.parquet",
    impedance_path="data/interim/all_impedance.parquet",
    output_path="data/processed/battery_features.parquet",
):
    discharge_path = Path(discharge_path)
    impedance_path = Path(impedance_path)
    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    discharge = pd.read_parquet(discharge_path)

    if impedance_path.exists():
        impedance = pd.read_parquet(impedance_path)
    else:
        impedance = pd.DataFrame()

    df = discharge.copy()

    df = compute_capacity_features(df)
    df = compute_rul_features(df)
    df = compute_voltage_features(df)
    df = compute_temperature_features(df)
    df = merge_impedance(df, impedance)

    df.to_parquet(output_path, index=False)

    return df


if __name__ == "__main__":
    features = build_features()

    print("Feature dataset created:")
    print(features.head())
    print(features.shape)
    print(features.columns)