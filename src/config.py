"""
Typed dataclass interface over the `app` section of config.yaml.

Usage:
    from src.config import load_config

    cfg = load_config()
    print(cfg.model_path)          # Path to .joblib artifact
    print(cfg.model_features)      # List of feature column names
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "config.yaml"


@dataclass(frozen=True)
class InputConfig:
    drop_columns: list[str]


@dataclass(frozen=True)
class PreprocessingConfig:
    fillna: dict[str, str]           # column -> fill value
    one_hot_encode: dict[str, list[str]]  # column -> ordered categories


@dataclass(frozen=True)
class AppConfig:
    model_path: Path
    input: InputConfig
    preprocessing: PreprocessingConfig
    model_features: list[str]


def load_config(path: Path = CONFIG_FILE) -> AppConfig:
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    app = raw["app"]

    return AppConfig(
        model_path=ROOT / app["model_path"],
        input=InputConfig(
            drop_columns=app["input"]["drop_columns"],
        ),
        preprocessing=PreprocessingConfig(
            fillna=app["preprocessing"]["fillna"],
            one_hot_encode=app["preprocessing"]["one_hot_encode"],
        ),
        model_features=app["model_features"],
    )