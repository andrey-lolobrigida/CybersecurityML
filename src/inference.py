"""
Prediction logic: takes raw input (data/raw schema), preprocesses it, and returns results.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.config import AppConfig
from src.preprocessing import preprocess


def predict(df: pd.DataFrame, model: RandomForestClassifier, cfg: AppConfig) -> pd.DataFrame:
    """
    Run inference on one or more raw input rows.

    Args:
        df:    DataFrame in data/raw schema (attack_detected column is ignored if present).
        model: Loaded RandomForestClassifier.
        cfg:   AppConfig loaded from config.yaml.

    Returns:
        DataFrame with columns:
            prediction  — 0 (normal) or 1 (attack)
            probability — model's estimated probability of attack
    """
    X = preprocess(df, cfg)
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    return pd.DataFrame(
        {"prediction": predictions, "probability": probabilities},
        index=df.index,
    )