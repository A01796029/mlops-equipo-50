# src/modeling/pipeline.py

from typing import Tuple, List
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def infer_feature_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    cat_cols = X.select_dtypes(include=["category", "object"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    return num_cols, cat_cols

def _get_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols, cat_cols = infer_feature_types(X)
    numeric = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", _get_ohe())
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ],
        remainder="drop",
        n_jobs=None,
        verbose_feature_names_out=False,
    )

def build_pipeline(model, X: pd.DataFrame) -> Pipeline:
    pre = build_preprocessor(X)
    return Pipeline(steps=[
        ("preprocess", pre),
        ("model", model),
    ])