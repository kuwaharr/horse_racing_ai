import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.paths import FEAT_DIR
from src.features.place_top3 import default_eval_odds_dataset_name, default_training_dataset_name
from src.models.place_top3_lgbm import evaluate_fixed_place_top3_rule_walk_forward, format_fixed_rule_report


def _optional_int(value: str) -> int | None:
    if value.lower() in {"none", "null", ""}:
        return None
    return int(value)


def _optional_int_list(value: str) -> list[int] | None:
    if value.lower() in {"none", "null", ""}:
        return None
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def main() -> None:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--training-dataset", type=Path, default=FEAT_DIR / default_training_dataset_name())
    arg_parser.add_argument("--odds-dataset", type=Path, default=FEAT_DIR / default_eval_odds_dataset_name())
    arg_parser.add_argument("--engine", choices=["auto", "pyarrow", "fastparquet"], default="auto")
    arg_parser.add_argument("--n-splits", type=int, default=4)
    arg_parser.add_argument("--min-train-ratio", type=float, default=0.5)
    arg_parser.add_argument("--stake", type=float, default=100.0)
    arg_parser.add_argument("--pred-min", type=float, default=0.35)
    arg_parser.add_argument("--odds-min", type=float, default=3.0)
    arg_parser.add_argument("--odds-max", type=float, default=5.0)
    arg_parser.add_argument("--distance-min", type=_optional_int, default=1800)
    arg_parser.add_argument("--distance-max", type=_optional_int, default=2200)
    arg_parser.add_argument("--track-id", type=_optional_int, default=None)
    arg_parser.add_argument("--include-track-ids", type=_optional_int_list, default=None)
    arg_parser.add_argument("--exclude-track-ids", type=_optional_int_list, default=None)
    arg_parser.add_argument("--surface-id", type=_optional_int, default=None)
    args = arg_parser.parse_args()

    report = evaluate_fixed_place_top3_rule_walk_forward(
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
    )
    print(format_fixed_rule_report(report))


if __name__ == "__main__":
    main()
