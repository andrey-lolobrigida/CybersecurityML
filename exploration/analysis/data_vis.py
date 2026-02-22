"""
Per-column visualizations for the cybersecurity intrusion dataset.
- Charts saved to visualization/
- Markdown report saved to reports/data/data_visualization.md
"""

import logging

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, must be set before pyplot import

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

# --- Paths ---
ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "raw" / "cybersecurity_intrusion_data.csv"
VIZ_DIR = ROOT / "visualization"
REPORT_DIR = ROOT / "reports" / "data"
REPORT_FILE = REPORT_DIR / "variables_visualization.md"

VIZ_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# --- Style ---
sns.set_theme(style="darkgrid", palette="muted")
PALETTE = sns.color_palette("muted")

SKIP_COLS = {"session_id"}

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────

def save_fig(name: str) -> Path:
    path = VIZ_DIR / f"{name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# ── Plot types ───────────────────────────────────────────────────────────────

def plot_histogram(df: pd.DataFrame, col: str) -> Path:
    """Histogram + KDE for continuous numeric columns."""
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df[col].dropna(), kde=True, ax=ax, color=PALETTE[0])
    ax.set_title(col.replace("_", " ").title(), fontsize=13)
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    return save_fig(col)


def plot_countplot(df: pd.DataFrame, col: str, rotate: bool = False) -> Path:
    """Sorted bar chart with percentage labels for discrete / low-cardinality columns."""
    fig, ax = plt.subplots(figsize=(8, 4))
    total = len(df[col].dropna())

    if pd.api.types.is_numeric_dtype(df[col]):
        # histplot keeps a true numeric axis — no categorical string conversion
        sns.histplot(df[col], discrete=True, shrink=0.8, ax=ax, color=PALETTE[0])
    else:
        order = [str(v) for v in df[col].value_counts().index]
        plot_df = df.assign(**{col: df[col].astype(str)})
        sns.countplot(data=plot_df, x=col, order=order, ax=ax, hue=col, palette="muted", legend=False)

    for p in ax.patches:
        if p.get_height() > 0:
            pct = f"{100 * p.get_height() / total:.1f}%"
            ax.annotate(
                pct,
                (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom", fontsize=9,
            )

    ax.set_title(col.replace("_", " ").title(), fontsize=13)
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    if rotate:
        plt.xticks(rotation=30, ha="right")
    return save_fig(col)


def plot_donut(df: pd.DataFrame, col: str) -> Path:
    """Donut chart for binary / low-cardinality categorical columns."""
    counts = df[col].value_counts(dropna=False)
    colors = sns.color_palette("muted", len(counts))

    fig, ax = plt.subplots(figsize=(6, 5))
    wedges, _, autotexts = ax.pie(
        counts,
        labels=None,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
        wedgeprops={"width": 0.5},
        pctdistance=0.75,
    )
    for at in autotexts:
        at.set_fontsize(11)

    ax.legend(
        wedges,
        [str(k) for k in counts.index],
        title=col,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )
    ax.set_title(col.replace("_", " ").title(), fontsize=13)
    return save_fig(col)


# ── Column-type dispatch ─────────────────────────────────────────────────────

def choose_plot(df: pd.DataFrame, col: str) -> Path:
    series = df[col]
    n_unique = series.nunique(dropna=False)
    dtype = series.dtype

    if dtype == "object" or str(dtype) == "category":
        return plot_donut(df, col) if n_unique <= 5 else plot_countplot(df, col, rotate=True)
    elif dtype == "float64":
        return plot_histogram(df, col)
    elif str(dtype).startswith("int"):
        if n_unique <= 2:
            return plot_donut(df, col)
        elif n_unique <= 15:
            return plot_countplot(df, col, rotate=False)
        else:
            return plot_histogram(df, col)
    else:
        return plot_countplot(df, col, rotate=True)


# ── Markdown report ──────────────────────────────────────────────────────────

def build_markdown(df: pd.DataFrame, col_paths: dict[str, Path]) -> str:
    lines = [
        "# Data Visualization Report",
        "",
        f"**Dataset:** `{DATA_FILE.name}` — {len(df):,} rows, {len(df.columns)} columns",
        "",
        "---",
        "",
    ]

    for col, img_path in col_paths.items():
        null_count = df[col].isna().sum()
        null_pct = null_count / len(df) * 100
        n_unique = df[col].nunique(dropna=False)
        dtype = str(df[col].dtype)
        # path relative to reports/data/ so the markdown links resolve correctly
        rel_path = "../../" + img_path.relative_to(ROOT).as_posix()

        lines += [
            f"## {col.replace('_', ' ').title()}",
            "",
            f"| dtype | unique values | null count |",
            f"|-------|---------------|------------|",
            f"| `{dtype}` | {n_unique} | {null_count} ({null_pct:.1f}%) |",
            "",
            f"![{col}]({rel_path})",
            "",
            "---",
            "",
        ]

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("Loading %s", DATA_FILE)
    df = pd.read_csv(DATA_FILE)
    df["encryption_used"] = df["encryption_used"].fillna("unencrypted")

    col_paths: dict[str, Path] = {}
    for col in df.columns:
        if col in SKIP_COLS:
            log.info("  Skipping  %s", col)
            continue
        log.info("  Plotting  %s ...", col)
        col_paths[col] = choose_plot(df, col)

    md = build_markdown(df, col_paths)
    REPORT_FILE.write_text(md, encoding="utf-8")

    log.info("\nSaved %d charts  →  %s", len(col_paths), VIZ_DIR)
    log.info("Report written   →  %s", REPORT_FILE)


if __name__ == "__main__":
    main()