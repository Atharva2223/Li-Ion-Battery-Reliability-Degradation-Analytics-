from pathlib import Path

from tensorboard import summary
import numpy as np
import pandas as pd
import json
import joblib

import matplotlib.pyplot as plt

from lifelines import WeibullFitter


def prepare_failure_data(features):
    """
    One row per battery cell for reliability modeling.
    """

    summary = (
        features.groupby("cell_id")
        .agg({
            "failure_cycle": "max",
            "eol_reached": "max"
        })
        .reset_index()
    )

    summary = summary.rename(columns={
        "failure_cycle": "time",
        "eol_reached": "event"
    })

    return summary


def fit_weibull(summary_df):
    """
    Fit Weibull survival model.
    """

    wf = WeibullFitter()

    wf.fit(
        durations=summary_df["time"],
        event_observed=summary_df["event"]
    )

    return wf


def compute_b_life(wf, percentile=0.10):
    """
    Compute B-life cycle.

    Example:
    B10 = cycle where 10% population has failed.
    """

    survival_probability = 1 - percentile

    return wf.percentile(survival_probability)


def print_reliability_summary(wf):
    beta = wf.rho_
    eta = wf.lambda_

    print("\n========== WEIBULL RELIABILITY ==========")

    print(f"Shape parameter β (rho): {beta:.4f}")
    print(f"Scale parameter η (lambda): {eta:.4f}")

    if beta < 1:
        interpretation = "Early-life failures"
    elif np.isclose(beta, 1, atol=0.1):
        interpretation = "Random failures"
    else:
        interpretation = "Wear-out ageing"

    print(f"Failure behavior: {interpretation}")

    b10 = compute_b_life(wf, 0.10)
    b50 = compute_b_life(wf, 0.50)

    print(f"B10 life: {b10:.2f} cycles")
    print(f"B50 life: {b50:.2f} cycles")

    print("=========================================\n")


def plot_survival_curve(wf):
    plt.figure(figsize=(10, 6))

    wf.plot_survival_function()

    plt.title("Weibull Survival Curve")
    plt.xlabel("Cycle")
    plt.ylabel("Survival Probability")

    plt.grid(True)

    plt.show()


def plot_hazard_curve(wf):
    plt.figure(figsize=(10, 6))

    wf.plot_hazard()

    plt.title("Weibull Hazard Function")
    plt.xlabel("Cycle")
    plt.ylabel("Hazard Rate")

    plt.grid(True)

    plt.show()


def run_weibull_analysis(
    feature_path="data/processed/battery_features.parquet"
):
    feature_path = Path(feature_path)

    features = pd.read_parquet(feature_path)

    summary = prepare_failure_data(features)
    wf = fit_weibull(summary)

    print_reliability_summary(wf)

    save_weibull_artifacts(wf, summary)

    plot_survival_curve(wf)

    plot_hazard_curve(wf)

    return wf, summary
def save_weibull_artifacts(wf, summary_df):
    """
    Save fitted Weibull model and reliability summary for dashboard use.
    """

    artifact_dir = Path("artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifact_dir / "weibull_model.pkl"
    summary_path = artifact_dir / "weibull_summary.json"

    beta = float(wf.rho_)
    eta = float(wf.lambda_)
    b10 = float(compute_b_life(wf, 0.10))
    b50 = float(compute_b_life(wf, 0.50))

    if beta < 1:
        interpretation = "Early-life failures"
    elif np.isclose(beta, 1, atol=0.1):
        interpretation = "Random failures"
    else:
        interpretation = "Wear-out ageing"

    payload = {
        "beta": beta,
        "eta": eta,
        "b10_life": b10,
        "b50_life": b50,
        "failure_behavior": interpretation,
        "num_cells": int(summary_df["cell_id"].nunique()),
        "num_failed_cells": int(summary_df["event"].sum()),
        "num_censored_cells": int((~summary_df["event"].astype(bool)).sum()),
    }

    joblib.dump(wf, model_path)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)

    print(f"Saved Weibull model to: {model_path}")
    print(f"Saved Weibull summary to: {summary_path}")

if __name__ == "__main__":
    wf, summary = run_weibull_analysis()