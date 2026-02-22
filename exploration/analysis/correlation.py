"""
Feature-relationship visualizations for the cybersecurity intrusion dataset.
- Charts saved to visualization/
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from scipy.stats import chi2_contingency
from sklearn.feature_selection import mutual_info_classif

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "processed" / "features.csv"
VIZ_DIR = ROOT / "visualization"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="darkgrid", palette="muted")


# ── Helpers ───────────────────────────────────────────────────────────────────

def save_fig(name: str) -> Path:
    path = VIZ_DIR / f"{name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_correlation_matrix(df: pd.DataFrame) -> Path:
    """Heatmap of Pearson correlations (lower triangle only)."""
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        corr,
        mask=mask,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7},
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Correlation Matrix", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    return save_fig("correlation_matrix")


def plot_class_distributions(
    X: pd.DataFrame,
    y: pd.Series,
    features: list[str] | None = None,
) -> Path:
    """Box + violin plots for each feature, split by attack_detected class.

    Args:
        features: columns to plot; defaults to all columns in X.
                  One-hot encoded binary columns (nunique == 2) are not very
                  informative here — consider filtering them out before calling.
    """
    if features is None:
        features = list(X.columns)
    n = len(features)

    class_map = {0: "Normal", 1: "Attack"} if set(y.unique()) == {0, 1} else {v: str(v) for v in sorted(y.unique())}
    target_order = [class_map[k] for k in sorted(class_map)]
    palette = "flare"

    plot_df = X[features].copy()
    plot_df["_class"] = y.map(class_map).values

    fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(14, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i, feature in enumerate(features):
        title = feature.replace("_", " ").title()

        sns.boxplot(
            data=plot_df, x="_class", y=feature, hue="_class",
            order=target_order, ax=axes[i, 0],
            palette=palette, legend=False,
        )
        axes[i, 0].set_title(f"Box Plot: {title}", fontsize=11)
        axes[i, 0].set_xlabel("")
        axes[i, 0].set_ylabel(feature)

        sns.violinplot(
            data=plot_df, x="_class", y=feature, hue="_class",
            order=target_order, ax=axes[i, 1],
            palette=palette, legend=False,
            inner="quartile",   # draws Q1 / median / Q3 lines inside each violin
        )
        axes[i, 1].set_title(f"Violin Plot: {title}", fontsize=11)
        axes[i, 1].set_xlabel("")
        axes[i, 1].set_ylabel("")

    fig.suptitle("Feature Distributions by Class (Normal vs Attack)", fontsize=14, y=1.002)
    plt.tight_layout()
    return save_fig("class_distributions")


def plot_binary_feature_association(X: pd.DataFrame, y: pd.Series) -> Path:
    """Chi-squared test + Cramér's V for binary (OHE) features vs the target class.

    Left panel : prevalence of feature=1 for Normal vs Attack classes.
    Right panel: Cramér's V effect size, annotated with significance stars.
    """
    binary_cols = [col for col in X.columns if X[col].nunique() <= 2]

    records = []
    for col in binary_cols:
        ct = pd.crosstab(X[col], y)
        chi2, p, _, _ = chi2_contingency(ct)
        n = len(y)
        cramers_v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
        rates = X.groupby(y)[col].mean()
        records.append({
            "feature": col,
            "p_value": p,
            "cramers_v": cramers_v,
            "rate_normal": rates.get(0, 0.0),
            "rate_attack": rates.get(1, 0.0),
        })

    results = (
        pd.DataFrame(records)
        .sort_values("cramers_v", ascending=True)   # ascending → highest at top
        .reset_index(drop=True)
    )

    def sig_label(pval: float) -> str:
        if pval < 0.001: return "***"
        if pval < 0.01:  return "**"
        if pval < 0.05:  return "*"
        return "ns"

    results["sig"] = results["p_value"].apply(sig_label)

    n_feat = len(results)
    palette = sns.color_palette("muted")
    y_pos = np.arange(n_feat)
    bar_w = 0.38

    fig, (ax_prev, ax_cv) = plt.subplots(1, 2, figsize=(16, max(4, round(n_feat * 0.45))))

    # ── Left: class-wise prevalence ───────────────────────────────────────────
    ax_prev.barh(y_pos + bar_w / 2, results["rate_normal"], bar_w,
                 label="Normal", color=palette[0])
    ax_prev.barh(y_pos - bar_w / 2, results["rate_attack"], bar_w,
                 label="Attack", color=palette[3])
    ax_prev.set_yticks(y_pos)
    ax_prev.set_yticklabels(results["feature"], fontsize=9)
    ax_prev.set_xlabel("Proportion  (feature = 1)")
    ax_prev.set_title("Class-wise Prevalence of Binary Features")
    ax_prev.legend()

    # ── Right: Cramér's V + significance stars ────────────────────────────────
    bars = ax_cv.barh(y_pos, results["cramers_v"], color=palette[2])
    for bar, sig in zip(bars, results["sig"]):
        ax_cv.text(
            bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
            sig, va="center", ha="left", fontsize=9,
        )
    ax_cv.set_yticks(y_pos)
    ax_cv.set_yticklabels([])
    ax_cv.set_xlabel("Cramér's V  (effect size)")
    ax_cv.set_title("Association Strength  (χ² test)")
    ax_cv.set_xlim(0, results["cramers_v"].max() * 1.2)
    ax_cv.text(
        0.99, 0.01, "*** p<0.001   ** p<0.01   * p<0.05   ns: p≥0.05",
        transform=ax_cv.transAxes, ha="right", va="bottom", fontsize=7, color="grey",
    )

    fig.suptitle("Binary Feature Association with attack_detected", fontsize=14)
    plt.tight_layout()
    return save_fig("binary_feature_association")


def plot_mutual_information(X: pd.DataFrame, y: pd.Series) -> Path:
    """Horizontal bar chart of mutual information scores against the target."""
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_df = (
        pd.DataFrame({"feature": X.columns, "mi_score": mi_scores})
        .sort_values("mi_score", ascending=True)  # ascending for horizontal bars
    )

    fig, ax = plt.subplots(figsize=(9, max(4, round(len(mi_df) * 0.35))))
    bars = ax.barh(mi_df["feature"], mi_df["mi_score"], color=sns.color_palette("muted")[0])

    # value labels at end of each bar
    for bar, val in zip(bars, mi_df["mi_score"]):
        ax.text(
            val + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", ha="left", fontsize=8,
        )

    ax.set_xlabel("Mutual Information Score")
    ax.set_title("Mutual Information vs attack_detected", fontsize=13)
    ax.set_xlim(0, mi_df["mi_score"].max() * 1.15)
    plt.tight_layout()
    return save_fig("mutual_information")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    df = pd.read_csv(DATA_FILE)
    X = df.drop(columns=["attack_detected"])
    y = df["attack_detected"]

    path = plot_correlation_matrix(X)
    print(f"Saved → {path}")

    path = plot_mutual_information(X, y)
    print(f"Saved → {path}")

    path = plot_binary_feature_association(X, y)
    print(f"Saved → {path}")

    # Skip binary OHE columns — box/violin on 0/1 flags carry no distributional info
    continuous_features = [col for col in X.columns if X[col].nunique() > 2]
    path = plot_class_distributions(X, y, features=continuous_features)
    print(f"Saved → {path}")


if __name__ == "__main__":
    main()