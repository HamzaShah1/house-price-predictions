"""
Train a Random Forest on the Ames data and save a *single* pipeline
(preprocessing + model) to   model_rf_small.joblib   in the repo root.

Usage from a terminal or the GitHub Codespace:
    python -m src.train_model
"""
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from .data_loader import load_train
from .feature_engineering import build_preprocessor

MODEL_PATH = Path(__file__).resolve().parents[1] / "model_rf_small.joblib"


def train(random_state: int = 42) -> None:
    df = load_train()

    y = df["SalePrice"]
    X = df.drop(columns=["SalePrice"])

    preprocessor = build_preprocessor(df)

    model = RandomForestRegressor(
        n_estimators=400,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=random_state,
    )

    pipeline = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("rf", model),
        ]
    )

    pipeline.fit(X, y)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model trained and saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()
