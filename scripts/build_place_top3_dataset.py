import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.paths import DB_PATH, FEAT_DIR
from src.features.place_top3 import DEFAULT_DATASET_NAME, build_place_top3_dataset


def main() -> None:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--db", type=Path, default=DB_PATH)
    arg_parser.add_argument("--output", type=Path, default=FEAT_DIR / DEFAULT_DATASET_NAME)
    arg_parser.add_argument("--engine", choices=["auto", "pyarrow", "fastparquet"], default="auto")
    args = arg_parser.parse_args()

    n_rows = build_place_top3_dataset(args.db, args.output, engine=args.engine)
    print(f"Wrote {n_rows:,} rows to {args.output}")


if __name__ == "__main__":
    main()
