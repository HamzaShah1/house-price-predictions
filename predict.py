import argparse, joblib, pandas as pd
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent / "model_rf.joblib"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--GrLivArea", type=float, required=True)
    parser.add_argument("--LotArea", type=float, required=True)
    parser.add_argument("--OverallQual", type=int, required=True)
    parser.add_argument("--YearBuilt", type=int, required=True)
    args = parser.parse_args()
    model = joblib.load(MODEL_PATH)
    df = pd.DataFrame([vars(args)])
    price = model.predict(df)[0]
    print(f"Predicted sale price: £{price:,.0f}")

if __name__ == "__main__":
    main()
