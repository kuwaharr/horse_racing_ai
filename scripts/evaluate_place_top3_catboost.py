import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.paths import FEAT_DIR
from src.features.place_top3 import default_eval_odds_dataset_name, default_training_dataset_name
from src.models.place_top3_catboost import _catboost_params, evaluate_place_top3_catboost_walk_forward
from src.models.place_top3_lgbm import format_walk_forward_report


def main() -> None:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--training-dataset", type=Path, default=FEAT_DIR / default_training_dataset_name())
    arg_parser.add_argument("--odds-dataset", type=Path, default=FEAT_DIR / default_eval_odds_dataset_name())
    arg_parser.add_argument("--engine", choices=["auto", "pyarrow", "fastparquet"], default="auto")
    arg_parser.add_argument("--n-splits", type=int, default=4)
    arg_parser.add_argument("--min-train-ratio", type=float, default=0.5)
    arg_parser.add_argument("--stake", type=float, default=100.0)
    arg_parser.add_argument("--min-rule-selections", type=int, default=30)
    arg_parser.add_argument("--iterations", type=int, default=500)
    arg_parser.add_argument("--learning-rate", type=float, default=0.03)
    arg_parser.add_argument("--depth", type=int, default=6)
    arg_parser.add_argument("--l2-leaf-reg", type=float, default=5.0)
    arg_parser.add_argument("--random-seed", type=int, default=42)
    args = arg_parser.parse_args()

    catboost_params = _catboost_params(
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        l2_leaf_reg=args.l2_leaf_reg,
        random_seed=args.random_seed,
    )
    report = evaluate_place_top3_catboost_walk_forward(
        training_dataset_path=args.training_dataset,
        odds_dataset_path=args.odds_dataset,
        engine=args.engine,
        n_splits=args.n_splits,
        min_train_ratio=args.min_train_ratio,
        stake=args.stake,
        min_rule_selections=args.min_rule_selections,
        catboost_params=catboost_params,
    )
    print(format_walk_forward_report(report))


if __name__ == "__main__":
    main()
