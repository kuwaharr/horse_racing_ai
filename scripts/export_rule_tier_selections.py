import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.paths import MODEL_DIR
from src.models.place_top3_lgbm import _apply_fixed_rule
from scripts.evaluate_rule_tiers_from_predictions import RULE_TIERS


DEFAULT_PREDICTIONS = MODEL_DIR / "catboost_place_top3_predictions.parquet"
DEFAULT_OUTPUT = MODEL_DIR / "rule_tier_selections.csv"


def _tier_condition_text(tier: dict) -> str:
    track = "all"
    if tier["include_track_ids"] is not None:
        track = "include:" + ",".join(str(v) for v in tier["include_track_ids"])
    elif tier["exclude_track_ids"] is not None:
        track = "exclude:" + ",".join(str(v) for v in tier["exclude_track_ids"])
    surface = "all" if tier["surface_id"] is None else str(tier["surface_id"])
    return (
        f"pred>={tier['pred_min']:.2f};"
        f"odds=[{tier['odds_min']:.1f},{tier['odds_max']:.1f});"
        f"distance=[{tier['distance_min']},{tier['distance_max']});"
        f"track={track};surface={surface}"
    )


def _tier_selections(predictions, tier: dict):
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
    ).copy()
    selected["tier"] = tier["name"]
    selected["tier_description"] = tier["description"]
    selected["tier_condition"] = _tier_condition_text(tier)
    return selected


def main() -> None:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    arg_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arg_parser.add_argument("--engine", choices=["auto", "pyarrow", "fastparquet"], default="auto")
    args = arg_parser.parse_args()

    try:
        import pandas as pd
    except ModuleNotFoundError as e:
        raise RuntimeError("ルール段階別の候補出力には pandas が必要です。") from e

    predictions = pd.read_parquet(args.predictions, engine=args.engine)
    selections = pd.concat(
        [_tier_selections(predictions, tier) for tier in RULE_TIERS],
        ignore_index=True,
    )
    selections["place_odds_mid"] = (selections["place_odds_min"] + selections["place_odds_max"]) / 2
    selections = selections.sort_values(
        ["date", "race_id", "tier", "pred_top3", "place_odds_mid"],
        ascending=[True, True, True, False, True],
    )
    output_columns = [
        "tier",
        "tier_description",
        "tier_condition",
        "fold",
        "date",
        "race_id",
        "horse_number",
        "target_top3",
        "pred_top3",
        "place_odds_min",
        "place_odds_mid",
        "place_odds_max",
        "track_id",
        "surface_id",
        "distance",
        "race_size",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    selections[output_columns].to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"Predictions: {args.predictions}")
    print(f"Output: {args.output}")
    print(f"Rows: {len(selections):,}")
    print(f"Races: {int(selections['race_id'].nunique()):,}")
    print("")
    print("tier            rows")
    for tier, group in selections.groupby("tier", sort=False):
        print(f"{tier:<15} {len(group):>5,}")


if __name__ == "__main__":
    main()
