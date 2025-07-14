#!/usr/bin/env python
"""
Example:
    python predict.py \
        --gr-liv-area 1500 \
        --lot-area 6000 \
        --overall-qual 6 \
        --year-built 1995
"""
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import joblib
from src.feature_engineering import build_preprocessor

MODEL_PATH = Path(__file__).with_name("model_rf_small.joblib")


def _build_row(args) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "GrLivArea": args.gr_liv_area,
                "LotArea": args.lot_area,
                "OverallQual": args.overall_qual,
                "YearBuilt": args.year_built,
            }
        ]
    )


def main() -> None:
    parser = ArgumentParser(description="Predict Ames house sale price")
    parser.add_argument("--gr-liv-area", type=float, required=True)
    parser.add_argument("--lot-area", type=float, required=True)
    parser.add_argument("--overall-qual", type=int, required=True)
    parser.add_argument("--year-built", type=int, required=True)
    args = parser.parse_args()

    # Load the pipeline
    pipe = joblib.load(MODEL_PATH)

    # The pipeline already knows its preprocessor, so just predict
    price = pipe.predict(_build_row(args))[0]
    print(f"Predicted sale price: £{price:,.0f}")


if __name__ == "__main__":
    main()
