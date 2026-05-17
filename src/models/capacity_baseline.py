from pathlib import Path
import json
import joblib

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBRegressor


CAPACITY_FEATURE_COLUMNS = [
    "capacity_ah",
    "initial_capacity_ah",
    "capacity_retention",
    "capacity_fade",
    "capacity_diff",
    "capacity_fade_rate",
    "capacity_roll_mean_5",
]


def load_dataset(feature_path="data/processed/battery_features.parquet"):
    return pd.read_parquet(feature_path)


def prepare_dataset(df):
    model_df = df.dropna(subset=CAPACITY_FEATURE_COLUMNS + ["rul"]).copy()

    X = model_df[CAPACITY_FEATURE_COLUMNS]
    y = model_df["rul"]
    groups = model_df["cell_id"]

    return X, y, groups, model_df


def split_dataset(X, y, groups):
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42
    )

    train_idx, test_idx = next(splitter.split(X, y, groups))

    return (
        X.iloc[train_idx],
        X.iloc[test_idx],
        y.iloc[train_idx],
        y.iloc[test_idx],
    )


def train_model(X_train, y_train):
    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print("\n========== CAPACITY-ONLY XGBOOST BASELINE ==========")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.4f}")
    print("====================================================\n")

    return {
        "model": "XGBoost capacity-only baseline",
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }


def save_artifacts(model, metrics):
    artifact_dir = Path("artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, artifact_dir / "xgboost_capacity_baseline.pkl")

    with open(artifact_dir / "capacity_baseline_features.json", "w", encoding="utf-8") as f:
        json.dump({"feature_columns": CAPACITY_FEATURE_COLUMNS}, f, indent=4)

    with open(artifact_dir / "capacity_baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print("Saved capacity baseline artifacts.")


def run_pipeline():
    df = load_dataset()

    X, y, groups, model_df = prepare_dataset(df)

    X_train, X_test, y_train, y_test = split_dataset(X, y, groups)

    model = train_model(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)

    save_artifacts(model, metrics)

    return model, metrics


if __name__ == "__main__":
    run_pipeline()