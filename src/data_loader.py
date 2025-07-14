"""
Simple helpers for loading the raw Ames Housing CSV files.

Assumes you have put the Kaggle files in   data/train.csv   and   data/test.csv
inside the repository root.
"""
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _path(filename: str) -> Path:
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{filename} not found in {DATA_DIR}. "
            "Download the raw dataset from Kaggle and place it there."
        )
    return path


def load_train() -> pd.DataFrame:
    """Return the training DataFrame, including SalePrice."""
    return pd.read_csv(_path("train.csv"))


def load_test() -> pd.DataFrame:
    """Return the test DataFrame (the one without SalePrice)."""
    return pd.read_csv(_path("test.csv"))
