"""
Samples 10 rows from the raw dataset, posts each to the deployed API,
and writes the inputs and outputs to data/inference_output/.

API URL is read from the API_URL env var (set it in .env).
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "raw" / "cybersecurity_intrusion_data.csv"
OUTPUT_DIR = REPO_ROOT / "data" / "inference_output"
SEED = 42
N_SAMPLES = 10

API_URL = os.environ["API_URL"].rstrip("/")
PREDICT_URL = f"{API_URL}/predict"

# Columns sent to the API (drop the label; session_id is optional but kept for traceability)
INPUT_COLS = [
    "session_id",
    "network_packet_size",
    "protocol_type",
    "login_attempts",
    "session_duration",
    "encryption_used",
    "ip_reputation_score",
    "failed_logins",
    "browser_type",
    "unusual_time_access",
]


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    samples = df.sample(n=N_SAMPLES, random_state=SEED)[INPUT_COLS]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    inputs: list[dict] = []
    outputs: list[dict] = []

    for _, row in samples.iterrows():
        payload = row.dropna().to_dict()
        # unusual_time_access comes in as float from pandas — cast to int
        if "unusual_time_access" in payload:
            payload["unusual_time_access"] = int(payload["unusual_time_access"])

        response = requests.post(PREDICT_URL, json=payload, timeout=10)
        response.raise_for_status()

        result = response.json()
        session_id = payload.get("session_id")

        inputs.append(payload)
        outputs.append({"session_id": session_id, **result})

        print(f"{session_id or '?'}  →  {result['label']}")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    (OUTPUT_DIR / f"{ts}_inputs.json").write_text(json.dumps(inputs, indent=2))
    (OUTPUT_DIR / f"{ts}_outputs.json").write_text(json.dumps(outputs, indent=2))
    print(f"\nSaved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()