"""
Pipeline orchestrator — intended to be consumed by the REST API layer.

The API calls `build_pipeline()` once at startup to load config and model,
then passes the resulting Pipeline into `run()` on each request.
"""

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.config import AppConfig, load_config
from src.inference import predict
from src.model_loading import load_model


@dataclass(frozen=True)
class Pipeline:
    cfg: AppConfig
    model: RandomForestClassifier


def build_pipeline() -> Pipeline:
    """Load config and model artifact. Call once at application startup."""
    cfg = load_config()
    model = load_model(cfg)
    return Pipeline(cfg=cfg, model=model)


_LABELS = {0: "Normal session", 1: "Possible attack session"}


def run(row: dict, pipeline: Pipeline) -> dict:
    """
    Run inference on a single raw input row.

    Args:
        row:      Dict with data/raw column schema (attack_detected is ignored if present).
        pipeline: Pipeline instance from build_pipeline().

    Returns:
        {"prediction": 0 | 1, "label": str, "probability": float}
    """
    df = pd.DataFrame([row])
    result = predict(df, pipeline.model, pipeline.cfg)
    prediction = int(result["prediction"].iloc[0])
    return {
        "prediction": prediction,
        "label": _LABELS[prediction],
        "probability": float(result["probability"].iloc[0]),
    }