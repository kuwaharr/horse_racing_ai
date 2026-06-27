import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.paths import MODEL_DIR
from src.models.place_top3_lgbm import _apply_fixed_rule, _selection_summary


DEFAULT_PREDICTIONS = MODEL_DIR / "catboost_place_top3_predictions.parquet"

RULE_TIERS = [
    {
        "name": "high_return",
        "description": "fewer bets, higher historical return",
        "pred_min": 0.45,
        "odds_min": 3.0,
        "odds_max": 6.0,
        "distance_min": 1800,
        "distance_max": 2200,
        "include_track_ids": None,
        "exclude_track_ids": None,
        "surface_id": None,
    },
    {
        "name": "middle_turf",
        "description": "middle volume, turf only",
        "pred_min": 0.35,
        "odds_min": 3.0,
        "odds_max": 5.0,
        "distance_min": 1800,
        "distance_max": 2200,
        "include_track_ids": None,
        "exclude_track_ids": None,
        "surface_id": 0,
    },
    {
        "name": "volume_tracks",
        "description": "more bets, selected tracks",
        "pred_min": 0.35,
        "odds_min": 3.0,
        "odds_max": 5.0,
        "distance_min": 1800,
        "distance_max": 2200,
        "include_track_ids": [4, 5, 6, 8, 9],
        "exclude_track_ids": None,
        "surface_id": None,
    },
]

NO_HORSE_ID_HIGH_RETURN_TIERS = [
    {
        "name": "high_return_no_horse_id",
        "description": "horse_id dropped model, fewer bets, selected tracks",
        "pred_min": 0.45,
        "odds_min": 3.0,
        "odds_max": 6.0,
        "distance_min": 1800,
        "distance_max": 2200,
        "include_track_ids": None,
        "exclude_track_ids": [3, 7, 10],
        "surface_id": None,
    },
]

TIER_SETS = {
    "affinity_lift": RULE_TIERS,
    "no_horse_id_high_return": NO_HORSE_ID_HIGH_RETURN_TIERS,
}


def _format_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}%"


def _summarize_tier(predictions, tier: dict, stake: float) -> dict:
    selected = _apply_fixed_rule(
        predictions,
        pred_min=tier["pred_min"],
        odds_min=tier["odds_min"],
        odds_max=tier["odds_max"],
        distance_min=tier["distance_min"],
        distance_max=tier["distance_max"],
        track_id=None,
        include_track_ids=tier["include_track_ids"],
        exclude_track_ids=tier["exclude_track_ids"],
        surface_id=tier["surface_id"],
    )
    overall = _selection_summary(selected, stake)
    fold_returns = []
    fold_selections = []
    for _, group in selected.groupby("fold", observed=True):
        fold_summary = _selection_summary(group, stake)
        fold_returns.append(fold_summary["return_mid_pct"])
        fold_selections.append(fold_summary["selections"])

    result = dict(tier)
    result.update(overall)
    result["min_fold_return_mid_pct"] = min(fold_returns) if fold_returns else None
    result["max_fold_return_mid_pct"] = max(fold_returns) if fold_returns else None
    result["min_fold_selections"] = min(fold_selections) if fold_selections else 0
    return result


def main() -> None:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    arg_parser.add_argument("--engine", choices=["auto", "pyarrow", "fastparquet"], default="auto")
    arg_parser.add_argument("--stake", type=float, default=100.0)
    arg_parser.add_argument("--tier-set", choices=sorted(TIER_SETS), default="affinity_lift")
    args = arg_parser.parse_args()

    try:
        import pandas as pd
    except ModuleNotFoundError as e:
        raise RuntimeError("保存済み予測のルール段階評価には pandas が必要です。") from e

    predictions = pd.read_parquet(args.predictions, engine=args.engine)
    rows = [_summarize_tier(predictions, tier, args.stake) for tier in TIER_SETS[args.tier_set]]

    print(f"Predictions: {args.predictions}")
    print(f"Tier set: {args.tier_set}")
    print(f"Rows: {len(predictions):,}")
    print(f"Races: {int(predictions['race_id'].nunique()):,}")
    print("")
    print(
        "tier            races  selections  hits  hit_rate  return_mid  "
        "min_mid  max_mid  min_fold_n  description"
    )
    for row in rows:
        print(
            f"{row['name']:<15} {row['races']:>5,}  {row['selections']:>10,}  "
            f"{row['hits']:>4,}  {_format_pct(row['hit_rate_pct']):>8}  "
            f"{_format_pct(row['return_mid_pct']):>10}  "
            f"{_format_pct(row['min_fold_return_mid_pct']):>7}  "
            f"{_format_pct(row['max_fold_return_mid_pct']):>7}  "
            f"{row['min_fold_selections']:>10,}  {row['description']}"
        )


if __name__ == "__main__":
    main()
