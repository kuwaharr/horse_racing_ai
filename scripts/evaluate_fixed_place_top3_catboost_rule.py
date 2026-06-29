import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.paths import FEAT_DIR
from src.features.place_top3 import default_eval_odds_dataset_name, default_training_dataset_name
from src.models.place_top3_catboost import _catboost_params, evaluate_fixed_place_top3_catboost_rule_walk_forward
from src.models.place_top3_lgbm import format_fixed_rule_report


def _optional_int(value: str) -> int | None:
    if value.lower() in {"none", "null", ""}:
        return None
    return int(value)


def _optional_int_list(value: str) -> list[int] | None:
    if value.lower() in {"none", "null", ""}:
        return None
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def _optional_str_list(value: str) -> list[str] | None:
    if value.lower() in {"none", "null", ""}:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def main() -> None:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--training-dataset", type=Path, default=FEAT_DIR / default_training_dataset_name())
    arg_parser.add_argument("--odds-dataset", type=Path, default=FEAT_DIR / default_eval_odds_dataset_name())
    arg_parser.add_argument("--engine", choices=["auto", "pyarrow", "fastparquet"], default="auto")
    arg_parser.add_argument("--n-splits", type=int, default=4)
    arg_parser.add_argument("--min-train-ratio", type=float, default=0.5)
    arg_parser.add_argument("--stake", type=float, default=100.0)
    arg_parser.add_argument("--pred-min", type=float, default=0.40)
    arg_parser.add_argument("--odds-min", type=float, default=3.0)
    arg_parser.add_argument("--odds-max", type=float, default=5.0)
    arg_parser.add_argument("--distance-min", type=_optional_int, default=1800)
    arg_parser.add_argument("--distance-max", type=_optional_int, default=2200)
    arg_parser.add_argument("--track-id", type=_optional_int, default=None)
    arg_parser.add_argument("--include-track-ids", type=_optional_int_list, default=None)
    arg_parser.add_argument("--exclude-track-ids", type=_optional_int_list, default=None)
    arg_parser.add_argument("--surface-id", type=_optional_int, default=None)
    arg_parser.add_argument("--drop-feature-patterns", type=_optional_str_list, default=None)
    arg_parser.add_argument("--train-surface-id", type=_optional_int, default=None)
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
    report = evaluate_fixed_place_top3_catboost_rule_walk_forward(
        training_dataset_path=args.training_dataset,
        odds_dataset_path=args.odds_dataset,
        engine=args.engine,
        n_splits=args.n_splits,
        min_train_ratio=args.min_train_ratio,
        stake=args.stake,
        pred_min=args.pred_min,
        odds_min=args.odds_min,
        odds_max=args.odds_max,
        distance_min=args.distance_min,
        distance_max=args.distance_max,
        track_id=args.track_id,
        include_track_ids=args.include_track_ids,
        exclude_track_ids=args.exclude_track_ids,
        surface_id=args.surface_id,
        drop_feature_patterns=args.drop_feature_patterns,
        train_surface_id=args.train_surface_id,
        catboost_params=catboost_params,
    )
    print(format_fixed_rule_report(report))


if __name__ == "__main__":
    main()
