import os
from pathlib import Path


def _default_data_root() -> Path:
    if os.name == "nt":
        return Path(r"D:\horse_racing_ai\data")
    return Path("/mnt/d/horse_racing_ai/data")


DATA_ROOT = Path(os.environ.get("HORSE_RACING_DATA_ROOT", _default_data_root()))

DB_PATH = DATA_ROOT / "hr.db"

FEAT_DIR = DATA_ROOT / "feature"

MODEL_DIR = DATA_ROOT / "model"

RAW_DIR = DATA_ROOT / "raw"

PRE_RACE_ODDS_RAW_DIR = DATA_ROOT / "raw_odds_snapshot"


def ensure_data_dirs() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PRE_RACE_ODDS_RAW_DIR.mkdir(parents=True, exist_ok=True)
