import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.paths import DB_PATH
from src.data.quality import collect_db_quality, format_db_quality


def main() -> None:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--db", type=Path, default=DB_PATH)
    args = arg_parser.parse_args()

    report = collect_db_quality(args.db)
    print(format_db_quality(report))


if __name__ == "__main__":
    main()
