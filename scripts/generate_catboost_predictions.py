import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.paths import FEAT_DIR, MODEL_DIR
from src.features.place_top3 import default_eval_odds_dataset_name, default_training_dataset_name
from src.models.place_top3_catboost import _catboost_params, build_catboost_walk_forward_predictions


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
    arg_parser.add_argument("--train-surface-id", type=int, default=None)
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
    report = build_catboost_walk_forward_predictions(
        training_dataset_path=args.training_dataset,
        odds_dataset_path=args.odds_dataset,
        engine=args.engine,
        n_splits=args.n_splits,
        min_train_ratio=args.min_train_ratio,
        stake=args.stake,
        drop_feature_patterns=args.drop_feature_patterns,
        train_surface_id=args.train_surface_id,
        catboost_params=catboost_params,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report["predictions"].to_parquet(args.output, index=False, engine=args.engine)

    print(f"Training dataset: {report['training_dataset_path']}")
    print(f"Evaluation odds dataset: {report['odds_dataset_path']}")
    print(f"Output: {args.output}")
    print(f"Rows: {len(report['predictions']):,}")
    print(f"Folds: {report['n_splits']}")
    print(f"Train surface id: {report['train_filter']['surface_id']}")
    print(f"Dropped features: {len(report['dropped_features'])}")
    print(f"CatBoost params: {report['catboost_params']}")
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
