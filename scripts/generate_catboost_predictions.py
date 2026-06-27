import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.paths import FEAT_DIR, MODEL_DIR
from src.features.place_top3 import default_eval_odds_dataset_name, default_training_dataset_name
from src.models.place_top3_catboost import build_catboost_walk_forward_predictions


DEFAULT_OUTPUT = MODEL_DIR / "catboost_place_top3_predictions.parquet"


def _optional_str_list(value: str) -> list[str] | None:
    if value.lower() in {"none", "null", ""}:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def main() -> None:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--training-dataset", type=Path, default=FEAT_DIR / default_training_dataset_name())
    arg_parser.add_argument("--odds-dataset", type=Path, default=FEAT_DIR / default_eval_odds_dataset_name())
    arg_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arg_parser.add_argument("--engine", choices=["auto", "pyarrow", "fastparquet"], default="auto")
    arg_parser.add_argument("--n-splits", type=int, default=4)
    arg_parser.add_argument("--min-train-ratio", type=float, default=0.5)
    arg_parser.add_argument("--stake", type=float, default=100.0)
    arg_parser.add_argument("--drop-feature-patterns", type=_optional_str_list, default=None)
    args = arg_parser.parse_args()

    report = build_catboost_walk_forward_predictions(
        training_dataset_path=args.training_dataset,
        odds_dataset_path=args.odds_dataset,
        engine=args.engine,
        n_splits=args.n_splits,
        min_train_ratio=args.min_train_ratio,
        stake=args.stake,
        drop_feature_patterns=args.drop_feature_patterns,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report["predictions"].to_parquet(args.output, index=False, engine=args.engine)

    print(f"Training dataset: {report['training_dataset_path']}")
    print(f"Evaluation odds dataset: {report['odds_dataset_path']}")
    print(f"Output: {args.output}")
    print(f"Rows: {len(report['predictions']):,}")
    print(f"Folds: {report['n_splits']}")
    print(f"Dropped features: {len(report['dropped_features'])}")
    print("fold  test_start  test_end    train_rows  test_rows  test_races      AUC  logloss    brier")
    for row in report["folds"]:
        test_end = row["test_end"] or "end"
        print(
            f"{row['fold']:>4}  {row['test_start']}  {test_end:<10}  "
            f"{row['train_rows']:>10,}  {row['test_rows']:>9,}  {row['test_races']:>10,}  "
            f"{row['auc']:>7.5f}  {row['logloss']:>7.5f}  {row['brier']:>7.5f}"
        )


if __name__ == "__main__":
    main()
