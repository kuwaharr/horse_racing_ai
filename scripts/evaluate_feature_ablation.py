import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.paths import FEAT_DIR
from src.features.place_top3 import default_eval_odds_dataset_name, default_training_dataset_name
from src.models.place_top3_lgbm import (
    _apply_fixed_rule,
    _fit_and_evaluate_split,
    _make_walk_forward_splits,
    _read_parquet,
    _selection_summary,
    _split_by_start_date,
)


ABLATION_GROUPS = {
    "course_affinity": [
        "_track_past_",
        "_surface_past_",
        "_distance_band_past_",
        "_course_past_",
    ],
    "connection_affinity": [
        "horse_jockey_",
        "jockey_trainer_",
        "horse_trainer_",
    ],
    "speed_pace": [
        "time_per_200",
        "finish_diff",
        "finish_3f",
        "corner4",
    ],
    "race_relative": [
        "race_",
    ],
    "recent_form": [
        "_recent",
    ],
    "distance_weight_fit": [
        "horse_distance_diff",
        "horse_distance_above",
        "horse_distance_below",
        "horse_past_avg_weight",
        "horse_weight_diff_avg",
        "horse_prev_distance",
    ],
}


def _columns_for_patterns(columns: list[str], patterns: list[str]) -> list[str]:
    protected = {
        "race_id",
        "date",
        "target_top3",
        "horse_number",
        "track_id",
        "surface_id",
        "distance",
        "race_size",
    }
    return [
        col
        for col in columns
        if col not in protected and any(pattern in col for pattern in patterns)
    ]


def _evaluate_variant(
    name: str,
    training_df,
    odds_df,
    drop_cols: list[str],
    n_splits: int,
    min_train_ratio: float,
    stake: float,
    pred_min: float,
    odds_min: float,
    odds_max: float,
    distance_min: int | None,
    distance_max: int | None,
):
    try:
        import pandas as pd
    except ModuleNotFoundError as e:
        raise RuntimeError("特徴量ablation評価には pandas が必要です。") from e

    eval_training_df = training_df.drop(columns=drop_cols)
    dates = sorted(eval_training_df["date"].dropna().unique())
    splits = _make_walk_forward_splits(dates, n_splits=n_splits, min_train_ratio=min_train_ratio)

    fold_rows = []
    selected_rows = []
    for fold_idx, (test_start, test_end) in enumerate(splits, start=1):
        train_df, test_df = _split_by_start_date(eval_training_df, test_start, test_end)
        split_report = _fit_and_evaluate_split(
            train_df=train_df,
            test_df=test_df,
            odds_df=odds_df,
            stake=stake,
            pred_thresholds=None,
            expected_value_thresholds=None,
            min_rule_selections=1,
        )
        selected = _apply_fixed_rule(
            split_report["eval_df"],
            pred_min=pred_min,
            odds_min=odds_min,
            odds_max=odds_max,
            distance_min=distance_min,
            distance_max=distance_max,
            track_id=None,
            include_track_ids=None,
            exclude_track_ids=None,
            surface_id=None,
        ).copy()
        selected["fold"] = fold_idx
        selected_rows.append(selected)
        summary = _selection_summary(selected, stake)
        fold_rows.append(
            {
                "fold": fold_idx,
                "auc": split_report["metrics"]["auc"],
                "selections": summary["selections"],
                "return_mid_pct": summary["return_mid_pct"],
            }
        )

    all_selected = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()
    overall = _selection_summary(all_selected, stake)
    avg_auc = sum(row["auc"] for row in fold_rows) / len(fold_rows)
    min_fold_return_mid = min(row["return_mid_pct"] for row in fold_rows if row["return_mid_pct"] is not None)

    return {
        "variant": name,
        "dropped_columns": len(drop_cols),
        "avg_auc": avg_auc,
        "selections": overall["selections"],
        "hits": overall["hits"],
        "hit_rate_pct": overall["hit_rate_pct"],
        "return_mid_pct": overall["return_mid_pct"],
        "min_fold_return_mid_pct": min_fold_return_mid,
    }


def _format_report(rows: list[dict]) -> str:
    lines = [
        "Feature ablation results",
        "variant                dropped  avg_auc  selections  hits  hit_rate  return_mid  min_fold_mid",
    ]
    baseline = rows[0]
    for row in rows:
        auc_delta = row["avg_auc"] - baseline["avg_auc"]
        return_delta = row["return_mid_pct"] - baseline["return_mid_pct"]
        lines.append(
            f"{row['variant']:<22} {row['dropped_columns']:>7}  "
            f"{row['avg_auc']:>7.5f} ({auc_delta:+.5f})  "
            f"{row['selections']:>10,}  {row['hits']:>4,}  "
            f"{row['hit_rate_pct']:>7.2f}%  "
            f"{row['return_mid_pct']:>9.2f}% ({return_delta:+.2f})  "
            f"{row['min_fold_return_mid_pct']:>12.2f}%"
        )
    return "\n".join(lines)


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
    arg_parser.add_argument("--distance-min", type=int, default=1800)
    arg_parser.add_argument("--distance-max", type=int, default=2200)
    args = arg_parser.parse_args()

    training_df = _read_parquet(args.training_dataset, args.engine)
    odds_df = _read_parquet(args.odds_dataset, args.engine)
    all_columns = list(training_df.columns)

    variants = [("all_features", [])]
    variants.extend(
        (f"no_{name}", _columns_for_patterns(all_columns, patterns))
        for name, patterns in ABLATION_GROUPS.items()
    )

    rows = [
        _evaluate_variant(
            name=name,
            training_df=training_df,
            odds_df=odds_df,
            drop_cols=drop_cols,
            n_splits=args.n_splits,
            min_train_ratio=args.min_train_ratio,
            stake=args.stake,
            pred_min=args.pred_min,
            odds_min=args.odds_min,
            odds_max=args.odds_max,
            distance_min=args.distance_min,
            distance_max=args.distance_max,
        )
        for name, drop_cols in variants
    ]
    print(_format_report(rows))


if __name__ == "__main__":
    main()
