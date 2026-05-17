from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from statsmodels.tsa.arima.model import ARIMA


EOL_THRESHOLD = 0.70


def load_dataset(feature_path="data/processed/battery_features.parquet"):
    return pd.read_parquet(feature_path)


def build_cell_predictions(df):
    """
    Forecast failure cycle using ARIMA on capacity retention.
    """

    predictions = []

    for cell_id, group in df.groupby("cell_id"):

        group = group.sort_values("discharge_cycle").copy()

        retention = group["capacity_retention"].values
        cycles = group["discharge_cycle"].values

        if len(retention) < 20:
            continue

        try:
            model = ARIMA(retention, order=(2, 1, 2))
            fitted = model.fit()

            future_steps = 300

            forecast = fitted.forecast(steps=future_steps)

            last_cycle = cycles[-1]

            future_cycles = np.arange(
                last_cycle + 1,
                last_cycle + future_steps + 1
            )

            below_eol = np.where(forecast <= EOL_THRESHOLD)[0]

            if len(below_eol) == 0:
                predicted_failure_cycle = future_cycles[-1]
            else:
                predicted_failure_cycle = future_cycles[below_eol[0]]

            actual_failure_cycle = group["failure_cycle"].iloc[0]

            predicted_rul = (
                predicted_failure_cycle - cycles
            )

            actual_rul = group["rul"].values

            for cycle, actual, pred in zip(
                cycles,
                actual_rul,
                predicted_rul
            ):
                predictions.append({
                    "cell_id": cell_id,
                    "cycle": cycle,
                    "actual_rul": actual,
                    "predicted_rul": pred
                })

        except Exception as e:
            print(f"ARIMA failed for {cell_id}: {e}")

    return pd.DataFrame(predictions)


def evaluate_predictions(pred_df):

    mae = mean_absolute_error(
        pred_df["actual_rul"],
        pred_df["predicted_rul"]
    )

    rmse = np.sqrt(
        mean_squared_error(
            pred_df["actual_rul"],
            pred_df["predicted_rul"]
        )
    )

    r2 = r2_score(
        pred_df["actual_rul"],
        pred_df["predicted_rul"]
    )

    print("\n========== ARIMA RUL BASELINE ==========")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.4f}")
    print("========================================\n")

    return {
        "model": "ARIMA baseline",
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }


def save_metrics(metrics):

    artifact_dir = Path("artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    with open(
        artifact_dir / "arima_metrics.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(metrics, f, indent=4)

    print("Saved ARIMA metrics.")


def run_pipeline():
    df = load_dataset()

    print("Loaded dataset shape:", df.shape)
    print("Number of cells:", df["cell_id"].nunique())

    pred_df = build_cell_predictions(df)

    print("Prediction dataframe shape:", pred_df.shape)

    if pred_df.empty:
        print("No ARIMA predictions were generated.")
        return pred_df, None

    metrics = evaluate_predictions(pred_df)

    save_metrics(metrics)

    return pred_df, metrics
if __name__ == "__main__":
    run_pipeline()