"""
Data structure analysis for the cybersecurity intrusion dataset.
Outputs a summary report to reports/data/.
"""

import logging
from pathlib import Path

import pandas as pd

# --- Paths ---
ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "raw" / "cybersecurity_intrusion_data.csv"
REPORT_DIR = ROOT / "reports" / "data"
REPORT_FILE = REPORT_DIR / "data_structure.md"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def analyse(df: pd.DataFrame) -> str:
    lines: list[str] = [
        "# Data Structure Report",
        "",
        f"**Dataset:** `{DATA_FILE.name}`",
        "",
    ]

    def section(title: str) -> None:
        lines.append(f"## {title}")
        lines.append("")

    # ── 1. Shape ─────────────────────────────────────────────────
    section("Shape")
    lines += [
        f"| Rows | Columns |",
        f"|------|---------|",
        f"| {df.shape[0]:,} | {df.shape[1]} |",
        "",
    ]

    # ── 2. Sample ─────────────────────────────────────────────────
    section("Sample (first 10 rows)")
    lines += [df.head(10).to_markdown(index=False), ""]

    # ── 3. Column summary ─────────────────────────────────────────
    section("Column Summary")
    null_counts = df.isna().sum()
    null_pct = (null_counts / len(df) * 100).round(2)
    summary = pd.DataFrame({
        "dtype": df.dtypes,
        "nulls": null_counts,
        "null_%": null_pct,
        "unique": df.nunique(),
    })
    lines += [summary.to_markdown(), ""]

    # ── 4. Numeric columns ────────────────────────────────────────
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if num_cols:
        section("Numeric Columns — Descriptive Stats")
        stats = df[num_cols].describe().T
        stats["skew"] = df[num_cols].skew().round(4)
        lines += [stats.to_markdown(floatfmt=".4f"), ""]

    # ── 5. Categorical columns ────────────────────────────────────
    cat_cols = (
        df.select_dtypes(include=["object", "category"])
        .drop(columns="session_id", errors="ignore")
        .columns.tolist()
    )
    if cat_cols:
        section("Categorical Columns — Value Counts")
        for col in cat_cols:
            vc = df[col].value_counts(dropna=False)
            vc_pct = (vc / len(df) * 100).round(2)
            tbl = pd.DataFrame({"count": vc, "%": vc_pct})
            lines += [f"### {col}", "", tbl.to_markdown(), ""]

    # ── 6. Target variable ────────────────────────────────────────
    target = "attack_detected"
    if target in df.columns:
        section(f"Target Variable: `{target}`")
        vc = df[target].value_counts(dropna=False)
        vc_pct = (vc / len(df) * 100).round(2)
        tbl = pd.DataFrame({"count": vc, "%": vc_pct})
        imbalance = vc.max() / vc.min()
        lines += [
            tbl.to_markdown(),
            "",
            f"**Imbalance ratio (majority / minority):** {imbalance:.2f}",
            "",
        ]

    # ── 7. Duplicates ─────────────────────────────────────────────
    section("Duplicates")
    n_dupes = df.duplicated().sum()
    lines += [f"Duplicate rows: **{n_dupes}** ({n_dupes / len(df) * 100:.2f}%)", ""]

    return "\n".join(lines)


def main() -> None:
    log.info("Loading %s ...", DATA_FILE)
    df = pd.read_csv(DATA_FILE)

    log.info("Analysing ...")
    report = analyse(df)

    REPORT_FILE.write_text(report, encoding="utf-8")
    log.info("Report written → %s", REPORT_FILE)


if __name__ == "__main__":
    main()