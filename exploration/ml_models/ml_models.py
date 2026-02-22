"""
Baseline ML model comparison on the cybersecurity intrusion dataset.

Trains Logistic Regression, Naive Bayes, SGD Classifier, and Random Forest,
evaluates them, and saves results to reports/model_results/.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "processed" / "reduced.csv"
RESULTS_DIR = ROOT / "reports" / "model_results"
VIZ_DIR = ROOT / "visualization" / "model_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR.mkdir(parents=True, exist_ok=True)

CLASS_LABELS = ["Normal", "Attack"]

RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET = "attack_detected"


# ── Model definitions ─────────────────────────────────────────────────────────
# Each model is defined in its own block so hyperparameters can be tuned
# independently without touching anything else in the script.

logistic_regression = LogisticRegression(
    l1_ratio=1.0,
    C=1.0,              # inverse regularisation strength — smaller = stronger
    solver="liblinear",
    max_iter=1000,
    random_state=RANDOM_STATE,
)

naive_bayes = GaussianNB(
    var_smoothing=1e-9,  # portion of largest variance added for stability
)

sgd = Pipeline([
    ("scaler", StandardScaler()),   # SGD is very sensitive to feature scale
    ("clf", SGDClassifier(
        loss="modified_huber",  # smooth hinge — supports predict_proba, much faster than SVC
        penalty="l2",
        alpha=1e-4,             # regularisation strength (higher = stronger)
        max_iter=1000,
        tol=1e-3,
        random_state=RANDOM_STATE,
    )),
])

random_forest = RandomForestClassifier(
    n_estimators=100,
    criterion="gini",
    max_depth=None,     # grow until pure leaves
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=None,
    random_state=RANDOM_STATE,
)

MODELS = [logistic_regression, naive_bayes, sgd, random_forest]


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_confusion_matrices(cms: list[tuple[str, np.ndarray]], run_id: str) -> Path:
    """Grid of annotated confusion matrices, one per model."""
    n = len(cms)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, cm) in zip(axes, cms):
        # row-normalised matrix for colour intensity (raw counts in annotations)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        sns.heatmap(
            cm_norm,
            annot=cm,           # show raw counts
            fmt="d",
            cmap="Blues",
            vmin=0, vmax=1,
            xticklabels=CLASS_LABELS,
            yticklabels=CLASS_LABELS,
            linewidths=0.5,
            ax=ax,
            cbar=False,
        )
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    fig.suptitle("Confusion Matrices", fontsize=13, y=1.02)
    plt.tight_layout()

    path = VIZ_DIR / f"confusion_matrices_{run_id}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# ── Helpers ───────────────────────────────────────────────────────────────────

def model_name(model) -> str:
    """Return the classifier class name, unwrapping Pipeline if needed."""
    if isinstance(model, Pipeline):
        return type(model.steps[-1][1]).__name__
    return type(model).__name__


def was_scaled(model) -> bool:
    """Return True if the model pipeline contains a StandardScaler step."""
    return isinstance(model, Pipeline) and any(
        isinstance(step, StandardScaler) for _, step in model.steps
    )


def sanitize_params(params: dict) -> dict:
    """Make get_params() output JSON-serialisable.

    - Drops 'steps' (redundant Pipeline metadata).
    - Drops top-level step-object keys (e.g. 'clf', 'scaler') whose values are
      estimator instances — the prefixed keys ('clf__loss', etc.) already carry
      the same information cleanly.
    - Converts any remaining non-primitive to repr().
    """
    _primitives = (bool, int, float, str, type(None))
    result = {}
    for k, v in params.items():
        if k == "steps":
            continue
        if "__" not in k and not isinstance(v, _primitives) and k != "scaler":
            continue  # top-level step objects (e.g. 'clf') — skip; keep 'scaler' for class identity
        result[k] = v if isinstance(v, _primitives) else repr(v)
    return result


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy":    round(accuracy_score(y_true, y_pred), 4),
        "sensitivity": round(recall_score(y_true, y_pred), 4),          # TP / (TP + FN)
        "specificity": round(tn / (tn + fp), 4),                         # TN / (TN + FP)
        "precision":   round(precision_score(y_true, y_pred), 4),
        "f1":          round(f1_score(y_true, y_pred), 4),
        "roc_auc":     round(roc_auc_score(y_true, y_proba), 4),
    }


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
    print(f"Class balance (test) — 0: {(y_test == 0).sum()}  1: {(y_test == 1).sum()}\n")

    results = []
    cms: list[tuple[str, np.ndarray]] = []

    for model in MODELS:
        name = model_name(model)
        print(f"Training {name} ...", end=" ", flush=True)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = compute_metrics(y_test, y_pred, y_proba)
        cms.append((name, confusion_matrix(y_test, y_pred)))

        print(
            f"acc={metrics['accuracy']:.4f}  "
            f"sens={metrics['sensitivity']:.4f}  "
            f"spec={metrics['specificity']:.4f}  "
            f"auc={metrics['roc_auc']:.4f}"
        )

        results.append({
            "model": name,
            "scaled": was_scaled(model),
            "hyperparameters": sanitize_params(model.get_params()),
            "metrics": metrics,
        })

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
            "n_train": len(X_train),
            "n_test": len(X_test),
        },
        "models": results,
    }

    out_path = RESULTS_DIR / f"{run_id}.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nSaved → {out_path}")

    viz_path = plot_confusion_matrices(cms, run_id)
    print(f"Saved → {viz_path}")


if __name__ == "__main__":
    main()