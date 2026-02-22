"""
Feature creation for the cybersecurity intrusion dataset.
Reads from data/interim/encoded.csv and saves to data/processed/features.csv.

New variables
-------------
succeeded_logins                    : login_attempts - failed_logins  (clipped at 0)
login_attempts_per_session_duration : login_attempts / session_duration (NaN when duration == 0)
log_session_duration                : log1p(session_duration)
log_login_attempts_per_session_duration : log1p(login_attempts_per_session_duration) (NaN propagated)
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT / "data" / "interim" / "encoded.csv"
OUTPUT_FILE = ROOT / "data" / "processed" / "features.csv"

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

raw_succeeded = df["login_attempts"] - df["failed_logins"]
clipped_count = (raw_succeeded < 0).sum()
df["succeeded_logins"] = raw_succeeded.clip(lower=0)

zero_duration_count = (df["session_duration"] == 0).sum()
df["login_attempts_per_session_duration"] = df["login_attempts"] / df["session_duration"].replace(0, float("nan"))

# log1p = log(1 + x): safe for zeros, avoids -inf
df["log_session_duration"] = np.log1p(df["session_duration"])
df["log_login_attempts_per_session_duration"] = np.log1p(df["login_attempts_per_session_duration"])

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print(f"Saved → {OUTPUT_FILE}")
print(f"Shape : {df.shape}")
print()
print("── Quality report ───────────────────────────────────────────────────────")
print(f"  succeeded_logins                        clipped : {clipped_count}")
print(f"  succeeded_logins                        NaNs    : {df['succeeded_logins'].isna().sum()}")
print(f"  login_attempts_per_session_duration     NaNs    : {df['login_attempts_per_session_duration'].isna().sum()}  (zero-duration rows: {zero_duration_count})")
print(f"  log_session_duration                    NaNs    : {df['log_session_duration'].isna().sum()}")
print(f"  log_login_attempts_per_session_duration NaNs    : {df['log_login_attempts_per_session_duration'].isna().sum()}")
print()
new_cols = [
    "succeeded_logins",
    "login_attempts_per_session_duration",
    "log_session_duration",
    "log_login_attempts_per_session_duration",
]
print(df[new_cols].describe().round(3))