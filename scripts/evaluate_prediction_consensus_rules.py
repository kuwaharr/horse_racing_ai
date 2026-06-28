import argparse
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.paths import MODEL_DIR
from src.models.place_top3_lgbm import _selection_summary


DEFAULT_BASE_PREDICTIONS = MODEL_DIR / "catboost_place_top3_predictions_affinity_lift_trial.parquet"
DEFAULT_SECONDARY_PREDICTIONS = MODEL_DIR / "catboost_place_top3_predictions_affinity_lift_no_horse_id.parquet"


CONSENSUS_RULES = [
    {
        "name": "union_value_odds",
        "description": "base >= 0.50 or no_horse_id >= 0.40, odds 4.0-6.0",
        "mode": "union",
        "base_pred_min": 0.50,
        "secondary_pred_min": 0.40,
        "avg_pred_min": None,
        "odds_min": 4.0,
        "odds_max": 6.0,
        "distance_min": 1800,
        "distance_max": 2200,
        "include_track_ids": None,
        "exclude_track_ids": [7, 10],
        "surface_id": None,
    },
    {
        "name": "union_value_odds_turf",
        "description": "union_value_odds, turf only, selected tracks",
        "mode": "union",
        "base_pred_min": 0.50,
        "secondary_pred_min": 0.40,
        "avg_pred_min": None,
        "odds_min": 4.0,
        "odds_max": 6.0,
        "distance_min": 1800,
        "distance_max": 2200,
        "include_track_ids": [5, 6, 8, 9],
        "exclude_track_ids": None,
        "surface_id": 0,
    },
    {
        "name": "union_wide_value",
        "description": "wider coverage, base >= 0.35 or no_horse_id >= 0.40",
        "mode": "union",
        "base_pred_min": 0.35,
        "secondary_pred_min": 0.40,
        "avg_pred_min": None,
        "odds_min": 3.0,
        "odds_max": 5.0,
        "distance_min": 1800,
        "distance_max": 2200,
        "include_track_ids": [4, 5, 6, 8, 9],
        "exclude_track_ids": None,
        "surface_id": None,
    },
    {
        "name": "union_broad_30_value",
        "description": "near 30 percent race coverage, selected tracks",
        "mode": "union",
        "base_pred_min": 0.35,
        "secondary_pred_min": 0.35,
        "avg_pred_min": None,
        "odds_min": 3.0,
        "odds_max": 6.0,
        "distance_min": None,
        "distance_max": None,
        "include_track_ids": [4, 5, 6, 8, 9],
        "exclude_track_ids": None,
        "surface_id": None,
    },
    {
        "name": "union_mid_20_value",
        "description": "near 20 percent race coverage, selected tracks",
        "mode": "union",
        "base_pred_min": 0.35,
        "secondary_pred_min": 0.35,
        "avg_pred_min": None,
        "odds_min": 3.0,
        "odds_max": 5.0,
        "distance_min": 1400,
        "distance_max": 2200,
        "include_track_ids": [4, 5, 6, 8, 9],
        "exclude_track_ids": None,
        "surface_id": None,
    },
    {
        "name": "union_mid_20_balanced",
        "description": "near 20 percent race coverage, balanced return",
        "mode": "union",
        "base_pred_min": 0.45,
        "secondary_pred_min": 0.45,
        "avg_pred_min": None,
        "odds_min": 2.5,
        "odds_max": 6.0,
        "distance_min": None,
        "distance_max": None,
        "include_track_ids": None,
        "exclude_track_ids": None,
        "surface_id": None,
    },
    {
        "name": "union_mid_20_stable",
        "description": "near 20 percent race coverage, stronger fold stability",
        "mode": "union",
        "base_pred_min": 0.50,
        "secondary_pred_min": 0.40,
        "avg_pred_min": None,
        "odds_min": 2.5,
        "odds_max": 6.0,
        "distance_min": 1400,
        "distance_max": None,
        "include_track_ids": None,
        "exclude_track_ids": [6],
        "surface_id": None,
    },
    {
        "name": "union_broad_37_value",
        "description": "higher race coverage, all tracks",
        "mode": "union",
        "base_pred_min": 0.35,
        "secondary_pred_min": 0.35,
        "avg_pred_min": None,
        "odds_min": 3.0,
        "odds_max": 6.0,
        "distance_min": None,
        "distance_max": None,
        "include_track_ids": None,
        "exclude_track_ids": None,
        "surface_id": None,
    },
    {
        "name": "consensus_high_return",
        "description": "base >= 0.45 and no_horse_id >= 0.40, selected tracks",
        "mode": "intersection",
        "base_pred_min": 0.45,
        "secondary_pred_min": 0.40,
        "avg_pred_min": None,
        "odds_min": 3.0,
        "odds_max": 6.0,
        "distance_min": 1800,
        "distance_max": 2200,
        "include_track_ids": None,
        "exclude_track_ids": [3, 7, 10],
        "surface_id": None,
    },
    {
        "name": "consensus_precise",
        "description": "base >= 0.45 and no_horse_id >= 0.45, selected tracks",
        "mode": "intersection",
        "base_pred_min": 0.45,
        "secondary_pred_min": 0.45,
        "avg_pred_min": None,
        "odds_min": 3.0,
        "odds_max": 6.0,
        "distance_min": 1800,
        "distance_max": 2200,
        "include_track_ids": None,
        "exclude_track_ids": [3, 7, 10],
        "surface_id": None,
    },
    {
        "name": "consensus_volume",
        "description": "base >= 0.35 and no_horse_id >= 0.45, selected tracks",
        "mode": "intersection",
        "base_pred_min": 0.35,
        "secondary_pred_min": 0.45,
        "avg_pred_min": None,
        "odds_min": 3.0,
        "odds_max": 6.0,
        "distance_min": 1800,
        "distance_max": 2200,
        "include_track_ids": None,
        "exclude_track_ids": [3, 7, 10],
        "surface_id": None,
    },
    {
        "name": "avg_balanced",
        "description": "average prediction >= 0.425, selected tracks",
        "mode": "average",
        "base_pred_min": None,
        "secondary_pred_min": None,
        "avg_pred_min": 0.425,
        "odds_min": 3.0,
        "odds_max": 6.0,
        "distance_min": 1800,
        "distance_max": 2200,
        "include_track_ids": None,
        "exclude_track_ids": [3, 7, 10],
        "surface_id": None,
    },
]


def _format_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}%"


def _load_consensus_predictions(base_path: Path, secondary_path: Path, engine: str):
    try:
        import pandas as pd
    except ModuleNotFoundError as e:
        raise RuntimeError("保存済み予測の合議評価には pandas が必要です。") from e

    base = pd.read_parquet(base_path, engine=engine)
    secondary = pd.read_parquet(secondary_path, engine=engine)
    keys = ["race_id", "horse_number"]
    secondary_pred = secondary[keys + ["pred_top3"]].rename(
        columns={"pred_top3": "pred_top3_secondary"}
    )
    predictions = base.merge(secondary_pred, on=keys, how="inner")
    predictions = predictions.rename(columns={"pred_top3": "pred_top3_base"})
    predictions["pred_top3_avg"] = (
        predictions["pred_top3_base"] + predictions["pred_top3_secondary"]
    ) / 2
    return predictions


def _apply_consensus_rule(predictions, rule: dict[str, Any]):
    selected = predictions[
        (predictions["place_odds_mid"] >= rule["odds_min"])
        & (predictions["place_odds_mid"] < rule["odds_max"])
    ].copy()
    if rule["distance_min"] is not None:
        selected = selected[selected["distance"] >= rule["distance_min"]]
    if rule["distance_max"] is not None:
        selected = selected[selected["distance"] < rule["distance_max"]]
    if rule["include_track_ids"] is not None:
        selected = selected[selected["track_id"].isin(rule["include_track_ids"])]
    if rule["exclude_track_ids"] is not None:
        selected = selected[~selected["track_id"].isin(rule["exclude_track_ids"])]
    if rule["surface_id"] is not None:
        selected = selected[selected["surface_id"] == rule["surface_id"]]

    if rule["mode"] == "intersection":
        selected = selected[
            (selected["pred_top3_base"] >= rule["base_pred_min"])
            & (selected["pred_top3_secondary"] >= rule["secondary_pred_min"])
        ]
    elif rule["mode"] == "union":
        selected = selected[
            (selected["pred_top3_base"] >= rule["base_pred_min"])
            | (selected["pred_top3_secondary"] >= rule["secondary_pred_min"])
        ]
    elif rule["mode"] == "average":
        selected = selected[selected["pred_top3_avg"] >= rule["avg_pred_min"]]
    else:
        raise ValueError(f"Unknown consensus mode: {rule['mode']}")

    selected["pred_top3"] = selected["pred_top3_avg"]
    return selected


def _summarize_rule(predictions, rule: dict[str, Any], stake: float) -> dict[str, Any]:
    selected = _apply_consensus_rule(predictions, rule)
    overall = _selection_summary(selected, stake)
    fold_returns = []
    fold_selections = []
    for _, group in selected.groupby("fold", observed=True):
        fold_summary = _selection_summary(group, stake)
        fold_returns.append(fold_summary["return_mid_pct"])
        fold_selections.append(fold_summary["selections"])

    result = dict(rule)
    result.update(overall)
    result["min_fold_return_mid_pct"] = min(fold_returns) if fold_returns else None
    result["max_fold_return_mid_pct"] = max(fold_returns) if fold_returns else None
    result["min_fold_selections"] = min(fold_selections) if fold_selections else 0
    return result


def main() -> None:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--base-predictions", type=Path, default=DEFAULT_BASE_PREDICTIONS)
    arg_parser.add_argument(
        "--secondary-predictions",
        type=Path,
        default=DEFAULT_SECONDARY_PREDICTIONS,
    )
    arg_parser.add_argument("--engine", choices=["auto", "pyarrow", "fastparquet"], default="auto")
    arg_parser.add_argument("--stake", type=float, default=100.0)
    args = arg_parser.parse_args()

    predictions = _load_consensus_predictions(
        args.base_predictions,
        args.secondary_predictions,
        args.engine,
    )
    total_races = int(predictions["race_id"].nunique())
    rows = [_summarize_rule(predictions, rule, args.stake) for rule in CONSENSUS_RULES]

    print(f"Base predictions: {args.base_predictions}")
    print(f"Secondary predictions: {args.secondary_predictions}")
    print(f"Rows: {len(predictions):,}")
    print(f"Races: {total_races:,}")
    print("")
    print(
        "rule                   races  buy_rate  selections  hits  hit_rate  return_mid  "
        "min_mid  max_mid  min_fold_n  description"
    )
    for row in rows:
        buy_rate_pct = None if total_races == 0 else row["races"] / total_races * 100
        print(
            f"{row['name']:<22} {row['races']:>5,}  {_format_pct(buy_rate_pct):>8}  "
            f"{row['selections']:>10,}  "
            f"{row['hits']:>4,}  {_format_pct(row['hit_rate_pct']):>8}  "
            f"{_format_pct(row['return_mid_pct']):>10}  "
            f"{_format_pct(row['min_fold_return_mid_pct']):>7}  "
            f"{_format_pct(row['max_fold_return_mid_pct']):>7}  "
            f"{row['min_fold_selections']:>10,}  {row['description']}"
        )


if __name__ == "__main__":
    main()
