from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

def load_ames() -> pd.DataFrame:
    """Return the raw Ames Housing dataset as a DataFrame."""
    return pd.read_csv(DATA_DIR / "ames.csv")

if __name__ == "__main__":
    df = load_ames()
    print("Rows, columns:", df.shape)
    print(df.head().to_string(index=False))
