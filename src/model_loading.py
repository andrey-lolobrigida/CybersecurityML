"""
Loads a trained model artifact from the path specified in AppConfig.
"""

from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier

from src.config import AppConfig


def load_model(cfg: AppConfig) -> RandomForestClassifier:
    path: Path = cfg.model_path
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    return joblib.load(path)