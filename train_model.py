"""
train_model.py
----------------
Trains and compares several regression algorithms on the Housing dataset,
selects the best performer, and saves it (as a full sklearn Pipeline,
including preprocessing) for use by the Streamlit app.

Usage:
    python train_model.py --data data/Housing.csv
"""

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ----------------------------------------------------------------------
# Column definitions -- matches the standard "Housing Prices Dataset"
# (price, area, bedrooms, bathrooms, stories, mainroad, guestroom,
#  basement, hotwaterheating, airconditioning, parking, prefarea,
#  furnishingstatus)
# ----------------------------------------------------------------------
TARGET = "price"
NUMERIC_FEATURES = ["area", "bedrooms", "bathrooms", "stories", "parking"]
BINARY_FEATURES = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea",
]
CATEGORICAL_FEATURES = ["furnishingstatus"]  # furnished / semi-furnished / unfurnished

ALL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set([TARGET] + ALL_FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")
    return df


def build_preprocessor() -> ColumnTransformer:
    # Binary yes/no columns -> map to 1/0 before the pipeline (simplest + interpretable)
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES + BINARY_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def encode_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in BINARY_FEATURES:
        df[col] = df[col].map({"yes": 1, "no": 0})
    return df


def evaluate(model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    return {
        "r2": round(r2_score(y_test, preds), 4),
        "mae": round(mean_absolute_error(y_test, preds), 2),
        "rmse": round(root_mean_squared_error(y_test, preds), 2),
    }


def main(data_path: str, output_dir: str):
    df = load_data(data_path)
    df = encode_binary_columns(df)

    X = df[ALL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    candidates = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=None, random_state=42, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
    }

    results = {}
    fitted_pipelines = {}

    for name, estimator in candidates.items():
        pipeline = Pipeline(
            steps=[("preprocess", build_preprocessor()), ("model", estimator)]
        )
        # 5-fold CV on the training set for a robust comparison
        cv_scores = cross_val_score(
            pipeline, X_train, y_train, cv=5, scoring="r2", n_jobs=-1
        )
        pipeline.fit(X_train, y_train)
        test_metrics = evaluate(pipeline, X_test, y_test)
        test_metrics["cv_r2_mean"] = round(cv_scores.mean(), 4)
        test_metrics["cv_r2_std"] = round(cv_scores.std(), 4)

        results[name] = test_metrics
        fitted_pipelines[name] = pipeline

        print(f"\n{name}")
        for k, v in test_metrics.items():
            print(f"  {k}: {v}")

    # Pick the best model by test R2
    best_name = max(results, key=lambda k: results[k]["r2"])
    best_pipeline = fitted_pipelines[best_name]

    print(f"\n>>> Best model: {best_name} (R2 = {results[best_name]['r2']})")

    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "house_price_model.pkl")
    joblib.dump(best_pipeline, model_path)

    metadata = {
        "best_model": best_name,
        "metrics": results,
        "features": {
            "numeric": NUMERIC_FEATURES,
            "binary": BINARY_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
        },
        "target": TARGET,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved model to: {model_path}")
    print(f"Saved metadata to: {os.path.join(output_dir, 'metadata.json')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train house price prediction model")
    parser.add_argument(
        "--data",
        type=str,
        default="data/Housing.csv",
        help="Path to the full Housing.csv dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models",
        help="Directory to save the trained model + metadata",
    )
    args = parser.parse_args()
    main(args.data, args.output_dir)
