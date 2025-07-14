"""
Lightweight helper that loads the saved pipeline and
returns predictions for any DataFrame with the same columns.
"""
from pathlib import Path
import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parents[1] / "model_rf_small.joblib"
_pipeline = None  # cached after first load


def _load_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = joblib.load(MODEL_PATH)
    return _pipeline


def predict(df: pd.DataFrame) -> pd.Series:
    """Return a Series of predicted SalePrice values (in dollars)."""
    pipe = _load_pipeline()
    return pd.Series(pipe.predict(df), index=df.index, name="SalePrice")
