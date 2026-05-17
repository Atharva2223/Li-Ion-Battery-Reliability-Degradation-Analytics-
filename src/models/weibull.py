from pathlib import Path
import numpy as np
import pandas as pd

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

    plot_survival_curve(wf)

    plot_hazard_curve(wf)

    return wf, summary


if __name__ == "__main__":
    wf, summary = run_weibull_analysis()