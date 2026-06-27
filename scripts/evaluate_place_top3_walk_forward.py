import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.paths import FEAT_DIR
from src.features.place_top3 import default_dataset_name
from src.models.place_top3_lgbm import evaluate_place_top3_lgbm_walk_forward, format_walk_forward_report


def main() -> None:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--early-dataset", type=Path, default=FEAT_DIR / default_dataset_name("early", history_features=True))
    arg_parser.add_argument("--late-dataset", type=Path, default=FEAT_DIR / default_dataset_name("late", history_features=True))
    arg_parser.add_argument("--engine", choices=["auto", "pyarrow", "fastparquet"], default="auto")
    arg_parser.add_argument("--n-splits", type=int, default=4)
    arg_parser.add_argument("--min-train-ratio", type=float, default=0.5)
    arg_parser.add_argument("--stake", type=float, default=100.0)
    arg_parser.add_argument("--min-rule-selections", type=int, default=30)
    args = arg_parser.parse_args()

    report = evaluate_place_top3_lgbm_walk_forward(
        early_dataset_path=args.early_dataset,
        late_dataset_path=args.late_dataset,
        engine=args.engine,
        n_splits=args.n_splits,
        min_train_ratio=args.min_train_ratio,
        stake=args.stake,
        min_rule_selections=args.min_rule_selections,
    )
    print(format_walk_forward_report(report))


if __name__ == "__main__":
    main()
