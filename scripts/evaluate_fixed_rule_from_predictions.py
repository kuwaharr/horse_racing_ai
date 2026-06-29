import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.paths import MODEL_DIR
from src.models.place_top3_lgbm import _apply_fixed_rule, _fixed_rule_breakdown, _selection_summary


DEFAULT_PREDICTIONS = MODEL_DIR / "catboost_place_top3_predictions.parquet"


def _optional_int(value: str) -> int | None:
    if value.lower() in {"none", "null", ""}:
        return None
    return int(value)


def _optional_int_list(value: str) -> list[int] | None:
    if value.lower() in {"none", "null", ""}:
        return None
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def _optional_float(value: str) -> float | None:
    if value.lower() in {"none", "null", ""}:
        return None
    return float(value)


def _format_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}%"


def main() -> None:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    arg_parser.add_argument("--engine", choices=["auto", "pyarrow", "fastparquet"], default="auto")
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
    arg_parser.add_argument("--pred-rank-max", type=_optional_int, default=None)
    arg_parser.add_argument("--ev-mid-min", type=_optional_float, default=None)
    args = arg_parser.parse_args()

    try:
        import pandas as pd
    except ModuleNotFoundError as e:
        raise RuntimeError("保存済み予測評価には pandas が必要です。") from e

    predictions = pd.read_parquet(args.predictions, engine=args.engine)
    if args.pred_rank_max is not None:
        predictions = predictions.copy()
        predictions["pred_rank"] = predictions.groupby("race_id")["pred_top3"].rank(
            method="first",
            ascending=False,
        )
    if args.ev_mid_min is not None:
        predictions = predictions.copy()
        predictions["expected_value_mid"] = predictions["pred_top3"] * predictions["place_odds_mid"]
    selected = _apply_fixed_rule(
        predictions,
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
    if args.pred_rank_max is not None:
        selected = selected[selected["pred_rank"] <= args.pred_rank_max]
    if args.ev_mid_min is not None:
        selected = selected[selected["expected_value_mid"] >= args.ev_mid_min]
    selected = selected.copy()
    selected["month"] = selected["date"].str.slice(0, 7)
    overall = _selection_summary(selected, args.stake)

    print(f"Predictions: {args.predictions}")
    print(f"Rows: {len(predictions):,}")
    print(f"Races: {int(predictions['race_id'].nunique()):,}")
    print(
        "Rule: "
        f"pred_top3>={args.pred_min:.2f}, "
        f"odds_mid=[{args.odds_min:.1f},{args.odds_max:.1f}), "
        f"distance=[{args.distance_min},{args.distance_max}), "
        f"track_id={args.track_id}, "
        f"include_track_ids={args.include_track_ids}, "
        f"exclude_track_ids={args.exclude_track_ids}, "
        f"surface_id={args.surface_id}, "
        f"pred_rank_max={args.pred_rank_max}, "
        f"ev_mid_min={args.ev_mid_min}"
    )
    print("")
    print("Overall")
    print("races  selections  hits  hit_rate  return_min  return_mid  return_max")
    print(
        f"{overall['races']:>5,}  {overall['selections']:>10,}  {overall['hits']:>4,}  "
        f"{_format_pct(overall['hit_rate_pct']):>8}  {_format_pct(overall['return_min_pct']):>9}  "
        f"{_format_pct(overall['return_mid_pct']):>9}  {_format_pct(overall['return_max_pct']):>9}"
    )

    for section_name, key in [("Fold", "fold"), ("Monthly", "month"), ("Track", "track_id"), ("Surface", "surface_id")]:
        print("")
        print(f"{section_name} breakdown")
        print("value       races  selections  hits  hit_rate  return_min  return_mid  return_max")
        for row in _fixed_rule_breakdown(selected, key, args.stake):
            print(
                f"{row['group_value']:<10}  {row['races']:>5,}  {row['selections']:>10,}  {row['hits']:>4,}  "
                f"{_format_pct(row['hit_rate_pct']):>8}  {_format_pct(row['return_min_pct']):>9}  "
                f"{_format_pct(row['return_mid_pct']):>9}  {_format_pct(row['return_max_pct']):>9}"
            )


if __name__ == "__main__":
    main()
