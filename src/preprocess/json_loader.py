import json
from pathlib import Path
from typing import Any


def load_json(file_path: Path) -> Any:
    with open(file_path, "r") as f:
        data = json.load(f)
        return data