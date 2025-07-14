"""
Feature‑engineering steps for the Ames dataset.

We build a scikit‑learn ColumnTransformer that

* fills in missing numeric values with the median,
* fills in missing categoric values with the most frequent value,
* then one‑hot encodes categoric columns.

The function build_preprocessor(df) inspects the DataFrame you pass
and returns a *fitted* transformer ready to plug into a model pipeline.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """Inspect *df* and return a fitted ColumnTransformer."""
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )

    # Fit so the object is ready for transform() calls
    preprocessor.fit(df)

    return preprocessor
