"""
Train a Random Forest classifier using parameters from config.yaml.

Reads data/interim/encoded.csv, trains on the configured feature set,
logs evaluation metrics to stdout, and saves the model artifact to models/.
No reports or visualisations are written — this script is for quick iteration.

Usage:
    uv run training/model_training.py
"""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "config.yaml"
MODELS_DIR = ROOT / "models"


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: Path) -> dict:
    log.info("Loading config   path=%s", path)
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if "training" not in cfg:
        log.error("config.yaml has no 'training' section")
        sys.exit(1)
    return cfg["training"]


# ── Data ──────────────────────────────────────────────────────────────────────

def load_data(cfg: dict) -> tuple[pd.DataFrame, pd.Series]:
    data_cfg  = cfg["data"]
    data_path = ROOT / data_cfg["input"]

    log.info("Reading data     path=%s", data_path)
    df = pd.read_csv(data_path)
    log.info("Loaded           shape=%s", df.shape)

    features = data_cfg["features"]
    target   = data_cfg["target"]

    missing = [c for c in features if c not in df.columns]
    if missing:
        log.error("Feature columns not found in dataset: %s", missing)
        sys.exit(1)
    if target not in df.columns:
        log.error("Target column '%s' not found in dataset", target)
        sys.exit(1)

    X = df[features]
    y = df[target]

    class_counts = y.value_counts().to_dict()
    log.info(
        "Features: %d  |  Target: '%s'  |  Classes: %s",
        len(features), target,
        {int(k): v for k, v in sorted(class_counts.items())},
    )
    return X, y


# ── Split ─────────────────────────────────────────────────────────────────────

def split_data(
    X: pd.DataFrame, y: pd.Series, cfg: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    split_cfg = cfg["split"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=split_cfg["test_size"],
        stratify=y if split_cfg["stratified"] else None,
        random_state=split_cfg["random_state"],
    )
    log.info(
        "Split            train=%d  test=%d  (stratified=%s  seed=%d)",
        len(X_train), len(X_test),
        split_cfg["stratified"], split_cfg["random_state"],
    )
    return X_train, X_test, y_train, y_test


# ── Training ──────────────────────────────────────────────────────────────────

def train(X_train: pd.DataFrame, y_train: pd.Series, cfg: dict) -> RandomForestClassifier:
    model_cfg = cfg["model"]
    hp        = model_cfg["hyperparameters"]
    log.info("Model type       %s", model_cfg["type"])
    log.info("Hyperparameters  %s", hp)

    model = RandomForestClassifier(**hp)

    log.info("Training ...")
    model.fit(X_train, y_train)
    log.info("Training complete")
    return model


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    metrics = {
        "accuracy":    round(accuracy_score(y_test, y_pred), 4),
        "sensitivity": round(recall_score(y_test, y_pred), 4),      # TP / (TP + FN)
        "specificity": round(tn / (tn + fp), 4),                     # TN / (TN + FP)
        "precision":   round(precision_score(y_test, y_pred), 4),
        "f1":          round(f1_score(y_test, y_pred), 4),
        "roc_auc":     round(roc_auc_score(y_test, y_proba), 4),
    }

    log.info("── Evaluation results ──────────────────────────────────────────────")
    for name, value in metrics.items():
        log.info("  %-14s %s", name, value)
    log.info("── Confusion matrix ────────────────────────────────────────────────")
    log.info("  TN=%-6d  FP=%-6d", tn, fp)
    log.info("  FN=%-6d  TP=%-6d", fn, tp)
    log.info("────────────────────────────────────────────────────────────────────")


# ── Artifact ──────────────────────────────────────────────────────────────────

def save_model(model: RandomForestClassifier) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = MODELS_DIR / f"random_forest_{timestamp}.joblib"
    joblib.dump(model, path)
    log.info("Model saved      path=%s", path)
    return path


# ── Entry point ───────────────────────────────────────────────────────────────

log = setup_logging()


def main() -> None:
    cfg = load_config(CONFIG_FILE)

    X, y = load_data(cfg)
    X_train, X_test, y_train, y_test = split_data(X, y, cfg)
    model = train(X_train, y_train, cfg)
    evaluate(model, X_test, y_test)
    save_model(model)


if __name__ == "__main__":
    main()