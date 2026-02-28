"""
Preprocesses raw input (data/raw schema) into the feature matrix expected by the model.

Steps driven entirely by AppConfig — no hardcoded column names:
  1. Drop configured columns (e.g. session_id)
  2. Fill NaN values per fillna map (e.g. encryption_used -> "Unencrypted")
  3. One-hot encode categorical columns using fixed, config-specified categories
  4. Select and order model_features columns

Input:  DataFrame with data/raw column schema (attack_detected may or may not be present)
Output: DataFrame with exactly the columns in cfg.model_features, in that order
"""

import pandas as pd

from src.config import AppConfig


def preprocess(df: pd.DataFrame, cfg: AppConfig) -> pd.DataFrame:
    df = df.copy()

    # 1. Drop columns not needed for inference (e.g. session_id)
    cols_to_drop = [c for c in cfg.input.drop_columns if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # 2. Fill NaN values
    df = df.fillna(cfg.preprocessing.fillna)

    # 3. One-hot encode using fixed categories from config
    #    Produces col_CATEGORY = 1/0 for each known category, regardless of
    #    what values are actually present in the input batch.
    for col, categories in cfg.preprocessing.one_hot_encode.items():
        for cat in categories:
            df[f"{col}_{cat}"] = (df[col] == cat).astype(int)
        df = df.drop(columns=[col])

    # 4. Select model features in the exact order the model was trained on
    missing = [f for f in cfg.model_features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features after preprocessing: {missing}")

    return df[cfg.model_features]