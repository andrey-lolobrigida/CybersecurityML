"""
Runs local inference on the entire raw dataset using the src pipeline
(no REST API involved) and saves results to data/inference_output/.
"""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.main import build_pipeline, run

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "raw" / "cybersecurity_intrusion_data.csv"
OUTPUT_DIR = REPO_ROOT / "data" / "inference_output"


def main() -> None:
    pipeline = build_pipeline()

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows — running inference...")

    results = []
    for _, row in df.iterrows():
        print(f"Running inference for session_id={row.get('session_id')}")
        result = run(row.to_dict(), pipeline)
        results.append({
            "session_id": row.get("session_id"),
            "prediction": result["prediction"],
            "label": result["label"],
            "probability": result["probability"],
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"{ts}_local_inference.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"Saved {len(results)} predictions → {out_path}")


if __name__ == "__main__":
    main()