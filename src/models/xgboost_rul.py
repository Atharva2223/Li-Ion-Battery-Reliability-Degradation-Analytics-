from pathlib import Path

import numpy as np
import pandas as pd
import json
import joblib

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.model_selection import GroupShuffleSplit

from xgboost import XGBRegressor

import matplotlib.pyplot as plt


FEATURE_COLUMNS = [
    "capacity_retention",
    "capacity_fade",
    "capacity_fade_rate",
    "voltage_mean",
    "voltage_slope",
    "time_to_3v",
    "temp_max",
    "temp_rise",
    "duration_s",
    "Re",
    "Rct"
]


def load_dataset(
    feature_path="data/processed/battery_features.parquet"
):
    df = pd.read_parquet(feature_path)

    return df


def prepare_dataset(df):
    """
    Select features and clean missing values.
    """

    model_df = df.copy()

    model_df = model_df.dropna(
        subset=FEATURE_COLUMNS + ["rul"]
    )

    X = model_df[FEATURE_COLUMNS]

    y = model_df["rul"]

    groups = model_df["cell_id"]

    return X, y, groups, model_df


def split_dataset(X, y, groups):
    """
    Battery-wise split to avoid leakage.
    """

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42
    )

    train_idx, test_idx = next(
        splitter.split(X, y, groups)
    )

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    return X_train, X_test, y_train, y_test


def train_xgboost(X_train, y_train):

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

def save_model_artifacts(model):
    """
    Save trained XGBoost model and feature column list for dashboard use.
    """

    artifact_dir = Path("artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifact_dir / "xgboost_rul_model.pkl"
    feature_path = artifact_dir / "feature_columns.json"

    joblib.dump(model, model_path)

    feature_payload = {
        "feature_columns": FEATURE_COLUMNS
    }

    with open(feature_path, "w", encoding="utf-8") as f:
        json.dump(feature_payload, f, indent=4)

    print(f"Saved model to: {model_path}")
    print(f"Saved feature columns to: {feature_path}")
    print("Feature columns saved:")
    print(FEATURE_COLUMNS)
def evaluate_model(model, X_test, y_test):

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)

    rmse = np.sqrt(
        mean_squared_error(y_test, preds)
    )

    r2 = r2_score(y_test, preds)

    print("\n========== XGBOOST RUL PERFORMANCE ==========")

    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.4f}")

    print("=============================================\n")

    return preds


def plot_predictions(y_test, preds):

    plt.figure(figsize=(8, 8))

    plt.scatter(
        y_test,
        preds,
        alpha=0.6
    )

    lims = [
        min(y_test.min(), preds.min()),
        max(y_test.max(), preds.max())
    ]

    plt.plot(
        lims,
        lims,
        linestyle="--"
    )

    plt.xlabel("Actual RUL")
    plt.ylabel("Predicted RUL")

    plt.title("Actual vs Predicted RUL")

    plt.grid(True)

    plt.show()


def plot_feature_importance(model):

    importance = pd.Series(
        model.feature_importances_,
        index=FEATURE_COLUMNS
    ).sort_values(ascending=False)

    plt.figure(figsize=(10, 6))

    importance.plot(kind="bar")

    plt.title("XGBoost Feature Importance")
    plt.ylabel("Importance Score")

    plt.grid(True)

    plt.show()

    print("\nFeature Importance:")
    print(importance)


def run_pipeline():

    df = load_dataset()

    X, y, groups, model_df = prepare_dataset(df)

    X_train, X_test, y_train, y_test = split_dataset(
        X,
        y,
        groups
    )

    model = train_xgboost(
        X_train,
        y_train
    )
    save_model_artifacts(model)

    preds = evaluate_model(
        model,
        X_test,
        y_test
    )

    plot_predictions(
        y_test,
        preds
    )

    plot_feature_importance(model)

    return model


if __name__ == "__main__":
    model = run_pipeline()