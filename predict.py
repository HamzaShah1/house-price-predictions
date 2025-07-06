import argparse, joblib, pandas as pd
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent / "model_rf_small.joblib"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--GrLivArea", type=float, default=0)
    p.add_argument("--LotArea", type=float, default=0)
    p.add_argument("--OverallQual", type=int, default=0)
    p.add_argument("--YearBuilt", type=int, default=0)
    args = p.parse_args()

    model = joblib.load(MODEL_PATH)
    cols = model.feature_names_in_
    row = {c: 0 for c in cols}
    row["Gr Liv Area"] = args.GrLivArea
    row["Lot Area"] = args.LotArea
    row["Overall Qual"] = args.OverallQual
    row["Year Built"] = args.YearBuilt

    df = pd.DataFrame([row], columns=cols)
    price = model.predict(df)[0]
    print(f"Predicted sale price: £{price:,.0f}")

if __name__ == "__main__":
    main()
