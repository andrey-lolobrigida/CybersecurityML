"""
Random Forest hyperparameter search on the cybersecurity intrusion dataset.

- GridSearchCV over accuracy on the training set
- Final evaluation on the held-out test set
- Results saved to reports/rf_results/
- Confusion matrix saved to visualization/rf_results/
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "processed" / "features.csv"
RESULTS_DIR = ROOT / "reports" / "rf_results"
VIZ_DIR = ROOT / "visualization" / "rf_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR.mkdir(parents=True, exist_ok=True)

CLASS_LABELS = ["Normal", "Attack"]
TARGET = "attack_detected"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# ── Hyperparameter grid ───────────────────────────────────────────────────────

PARAM_GRID = {
    "n_estimators":     [100, 300, 500, 1000],
    "criterion":        ["gini", "entropy"],
    "max_depth":        [None, 10, 20],
    "min_samples_split":[2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features":     ["sqrt", "log2", None],
}


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy":    round(accuracy_score(y_true, y_pred), 4),
        "sensitivity": round(recall_score(y_true, y_pred), 4),
        "specificity": round(tn / (tn + fp), 4),
        "precision":   round(precision_score(y_true, y_pred), 4),
        "f1":          round(f1_score(y_true, y_pred), 4),
        "roc_auc":     round(roc_auc_score(y_true, y_proba), 4),
    }


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm: np.ndarray, run_id: str) -> Path:
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm_norm,
        annot=cm,
        fmt="d",
        cmap="Blues",
        vmin=0, vmax=1,
        xticklabels=CLASS_LABELS,
        yticklabels=CLASS_LABELS,
        linewidths=0.5,
        ax=ax,
        cbar=False,
    )
    ax.set_title("Random Forest — Confusion Matrix", fontsize=12)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()

    path = VIZ_DIR / f"confusion_matrix_{run_id}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    df = pd.read_csv(DATA_FILE)
    X = df.drop(columns=[TARGET]).values
    y = df[TARGET].values
    feature_names = df.drop(columns=[TARGET]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

    n_candidates = 1
    for values in PARAM_GRID.values():
        n_candidates *= len(values)
    print(f"Grid: {n_candidates} combinations × {CV_FOLDS} folds = {n_candidates * CV_FOLDS} fits\n")

    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        param_grid=PARAM_GRID,
        scoring="recall",
        cv=CV_FOLDS,
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred  = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    cm      = confusion_matrix(y_test, y_pred)
    metrics = compute_metrics(y_test, y_pred, y_proba)

    print(f"\nBest CV sensitivity : {grid_search.best_score_:.4f}")
    print(f"Test accuracy    : {metrics['accuracy']:.4f}")
    print(f"Test sensitivity : {metrics['sensitivity']:.4f}")
    print(f"Test specificity : {metrics['specificity']:.4f}")
    print(f"Test AUC         : {metrics['roc_auc']:.4f}")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]

    output = {
        "run_id": run_id,
        "dataset": str(DATA_FILE.relative_to(ROOT)),
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "stratified": True,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
        },
        "grid_search": {
            "scoring": "recall",
            "cv_folds": CV_FOLDS,
            "param_grid": PARAM_GRID,
            "n_candidates": n_candidates,
            "best_cv_sensitivity": round(grid_search.best_score_, 4),
            "best_params": grid_search.best_params_,
        },
        "test_metrics": metrics,
    }

    json_path = RESULTS_DIR / f"{run_id}.json"
    json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nSaved → {json_path}")

    viz_path = plot_confusion_matrix(cm, run_id)
    print(f"Saved → {viz_path}")


if __name__ == "__main__":
    main()