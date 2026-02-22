"""
Produces data/interim/reduced.csv — a column-reduced subset for experimentation.

Loaded from data/processed/features.csv (not encoded.csv) because succeeded_logins
is a derived variable that only exists there.
"""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT / "data" / "processed" / "features.csv"
OUTPUT_FILE = ROOT / "data" / "processed" / "reduced.csv"

KEEP_COLS = [
    "browser_type_Unknown",
    "browser_type_Chrome",
    "login_attempts",
    "ip_reputation_score",
    "failed_logins",
    "succeeded_logins",
    "attack_detected",      # target — kept so the file is usable for modelling
]

df = pd.read_csv(INPUT_FILE)

missing = [c for c in KEEP_COLS if c not in df.columns]
if missing:
    raise ValueError(f"Columns not found in source file: {missing}")

df[KEEP_COLS].to_csv(OUTPUT_FILE, index=False)

print(f"Saved → {OUTPUT_FILE}")
print(f"Shape : {df[KEEP_COLS].shape}")