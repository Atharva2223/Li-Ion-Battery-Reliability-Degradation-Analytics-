from pathlib import Path

import pandas as pd
import shap
import matplotlib.pyplot as plt

from src.models.xgboost_rul import (
    FEATURE_COLUMNS,
    load_dataset,
    prepare_dataset,
    split_dataset,
    train_xgboost,
)


def run_shap_analysis():
    df = load_dataset("data/processed/battery_features.parquet")

    X, y, groups, model_df = prepare_dataset(df)

    X_train, X_test, y_train, y_test = split_dataset(X, y, groups)

    model = train_xgboost(X_train, y_train)

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_test)

    print("SHAP analysis complete.")
    print("Test shape:", X_test.shape)

    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=FEATURE_COLUMNS,
        show=True
    )

    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=FEATURE_COLUMNS,
        plot_type="bar",
        show=True
    )

    return model, X_test, shap_values


if __name__ == "__main__":
    run_shap_analysis()