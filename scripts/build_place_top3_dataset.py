import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.paths import DB_PATH, FEAT_DIR
from src.features.place_top3 import (
    build_win_top1_compat_dataset,
    build_win_top1_compat_eval_odds_dataset,
    build_place_top3_dataset,
    build_place_top3_eval_odds_dataset,
    default_eval_odds_dataset_name,
    default_training_dataset_name,
    default_win_compat_eval_odds_dataset_name,
    default_win_compat_training_dataset_name,
)


def main() -> None:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--db", type=Path, default=DB_PATH)
    arg_parser.add_argument(
        "--kind",
        choices=["training", "eval-odds", "win-training", "win-eval-odds"],
        default="training",
    )
    arg_parser.add_argument("--output", type=Path, default=None)
    arg_parser.add_argument("--engine", choices=["auto", "pyarrow", "fastparquet"], default="auto")
    arg_parser.add_argument("--no-history-features", action="store_true")
    arg_parser.add_argument("--no-pedigree-features", action="store_true")
    args = arg_parser.parse_args()

    if args.kind == "training":
        output = args.output or FEAT_DIR / default_training_dataset_name()
        n_rows = build_place_top3_dataset(
            args.db,
            output,
            engine=args.engine,
            history_features=not args.no_history_features,
            pedigree_features=not args.no_pedigree_features,
        )
    elif args.kind == "eval-odds":
        output = args.output or FEAT_DIR / default_eval_odds_dataset_name()
        n_rows = build_place_top3_eval_odds_dataset(
            args.db,
            output,
            engine=args.engine,
        )
    elif args.kind == "win-training":
        output = args.output or FEAT_DIR / default_win_compat_training_dataset_name()
        n_rows = build_win_top1_compat_dataset(
            args.db,
            output,
            engine=args.engine,
            history_features=not args.no_history_features,
            pedigree_features=not args.no_pedigree_features,
        )
    else:
        output = args.output or FEAT_DIR / default_win_compat_eval_odds_dataset_name()
        n_rows = build_win_top1_compat_eval_odds_dataset(
            args.db,
            output,
            engine=args.engine,
        )
    print(f"Wrote {n_rows:,} rows to {output}")


if __name__ == "__main__":
    main()
