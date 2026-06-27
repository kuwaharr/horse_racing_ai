import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.paths import FEAT_DIR, MODEL_DIR
from src.features.place_top3 import default_eval_odds_dataset_name, default_training_dataset_name
from src.models.place_top3_catboost import (
    _drop_feature_patterns,
    _filter_by_surface,
    _fit_and_evaluate_catboost_split,
)
from src.models.place_top3_lgbm import _make_walk_forward_splits, _read_parquet, _split_by_start_date


DEFAULT_OUTPUT = MODEL_DIR / "catboost_feature_importance.csv"


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
    arg_parser.add_argument("--top-n", type=int, default=50)
    args = arg_parser.parse_args()

    try:
        import pandas as pd
    except ModuleNotFoundError as e:
        raise RuntimeError("CatBoost特徴量重要度の出力には pandas が必要です。") from e

    training_df = _read_parquet(args.training_dataset, args.engine)
    training_df, dropped_features = _drop_feature_patterns(training_df, args.drop_feature_patterns)
    odds_df = _read_parquet(args.odds_dataset, args.engine)

    dates = sorted(training_df["date"].dropna().unique())
    splits = _make_walk_forward_splits(dates, n_splits=args.n_splits, min_train_ratio=args.min_train_ratio)

    rows = []
    fold_rows = []
    for fold_idx, (test_start, test_end) in enumerate(splits, start=1):
        train_df, test_df = _split_by_start_date(training_df, test_start, test_end)
        train_df = _filter_by_surface(train_df, args.train_surface_id)
        if train_df.empty:
            raise ValueError(f"No training rows after train surface filter: surface_id={args.train_surface_id}")
        split_report = _fit_and_evaluate_catboost_split(
            train_df=train_df,
            test_df=test_df,
            odds_df=odds_df,
            stake=args.stake,
            min_rule_selections=1,
        )
        fold_rows.append(
            {
                "fold": fold_idx,
                "test_start": test_start,
                "test_end": test_end or "end",
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "auc": split_report["metrics"]["auc"],
            }
        )
        for item in split_report["feature_importance"]:
            rows.append(
                {
                    "fold": fold_idx,
                    "feature": item["feature"],
                    "importance": item["importance"],
                }
            )

    importance_df = pd.DataFrame(rows)
    summary = (
        importance_df.groupby("feature", as_index=False)
        .agg(
            mean_importance=("importance", "mean"),
            max_importance=("importance", "max"),
            min_importance=("importance", "min"),
            folds=("fold", "nunique"),
        )
        .sort_values(["mean_importance", "max_importance"], ascending=False)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"Training dataset: {args.training_dataset}")
    print(f"Evaluation odds dataset: {args.odds_dataset}")
    print(f"Output: {args.output}")
    print(f"Rows: {len(summary):,}")
    print(f"Dropped features: {len(dropped_features)}")
    print(f"Train surface id: {args.train_surface_id}")
    print("")
    print("fold  test_start  test_end    train_rows  test_rows      AUC")
    for row in fold_rows:
        print(
            f"{row['fold']:>4}  {row['test_start']}  {row['test_end']:<10}  "
            f"{row['train_rows']:>10,}  {row['test_rows']:>9,}  {row['auc']:>7.5f}"
        )
    print("")
    print("feature                                      mean_importance  max_importance  folds")
    for row in summary.head(args.top_n).itertuples(index=False):
        print(
            f"{row.feature:<44} {row.mean_importance:>15.6f}  "
            f"{row.max_importance:>14.6f}  {row.folds:>5}"
        )


if __name__ == "__main__":
    main()
