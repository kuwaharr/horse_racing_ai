import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.paths import DB_PATH, FEAT_DIR
from src.features.place_top3 import build_place_top3_dataset, default_dataset_name


def main() -> None:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--db", type=Path, default=DB_PATH)
    arg_parser.add_argument("--mode", choices=["early", "late"], default="late")
    arg_parser.add_argument("--output", type=Path, default=None)
    arg_parser.add_argument("--engine", choices=["auto", "pyarrow", "fastparquet"], default="auto")
    arg_parser.add_argument("--history-features", action="store_true")
    args = arg_parser.parse_args()

    output = args.output or FEAT_DIR / default_dataset_name(args.mode, history_features=args.history_features)
    n_rows = build_place_top3_dataset(
        args.db,
        output,
        mode=args.mode,
        engine=args.engine,
        history_features=args.history_features,
    )
    print(f"Wrote {n_rows:,} rows to {output}")


if __name__ == "__main__":
    main()
