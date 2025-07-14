"""Read the raw Ames housing CSV and return a pandas DataFrame."""
from pathlib import Path
import pandas as pd

DEFAULT_PATH = Path(__file__).parents[1] / "data" / "train.csv"


def load_raw(path: Path | None = None) -> pd.DataFrame:
    path = path or DEFAULT_PATH
    return pd.read_csv(path)
