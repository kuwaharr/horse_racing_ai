import argparse
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.evaluate_prediction_consensus_rules import (
    DEFAULT_BASE_PREDICTIONS,
    DEFAULT_SECONDARY_PREDICTIONS,
    _load_consensus_predictions,
)
from src.models.place_top3_lgbm import _selection_summary


def _format_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}%"


def _candidate_rules(modes: list[str]) -> list[dict[str, Any]]:
    pred_thresholds = [0.30, 0.32, 0.35, 0.38, 0.40, 0.45, 0.50]
    avg_thresholds = [0.35, 0.375, 0.40, 0.425, 0.45, 0.475, 0.50]
    odds_ranges = [
        (2.5, 5.0),
        (2.5, 6.0),
        (3.0, 5.0),
        (3.0, 6.0),
        (4.0, 6.0),
    ]
    distance_ranges = [
        (None, None),
        (1400, 1800),
        (1400, 2200),
        (1400, None),
        (1800, 2200),
        (2200, None),
    ]
    track_filters = [
        ("all", None, None),
        ("exclude_7_10", None, [7, 10]),
        ("exclude_3_7_10", None, [3, 7, 10]),
        ("include_4_5_6_8_9", [4, 5, 6, 8, 9], None),
    ]
    surface_filters = [None, 0, 1]

    candidates = []
    for mode in modes:
        thresholds = (
            [(None, None, avg_threshold) for avg_threshold in avg_thresholds]
            if mode == "average"
            else [
                (base_threshold, secondary_threshold, None)
                for base_threshold in pred_thresholds
                for secondary_threshold in pred_thresholds
            ]
        )
        for base_pred_min, secondary_pred_min, avg_pred_min in thresholds:
            for odds_min, odds_max in odds_ranges:
                for distance_min, distance_max in distance_ranges:
                    for track_label, include_track_ids, exclude_track_ids in track_filters:
                        for surface_id in surface_filters:
                            candidates.append(
                                {
                                    "mode": mode,
                                    "base_pred_min": base_pred_min,
                                    "secondary_pred_min": secondary_pred_min,
                                    "avg_pred_min": avg_pred_min,
                                    "odds_min": odds_min,
                                    "odds_max": odds_max,
                                    "distance_min": distance_min,
                                    "distance_max": distance_max,
                                    "track_label": track_label,
                                    "include_track_ids": include_track_ids,
                                    "exclude_track_ids": exclude_track_ids,
                                    "surface_id": surface_id,
                                }
                            )
    return candidates


def _rule_key(rule: dict[str, Any]) -> str:
    distance = f"[{rule['distance_min']},{rule['distance_max']})"
    surface = "all" if rule["surface_id"] is None else str(rule["surface_id"])
    if rule["mode"] == "average":
        score = f"avg>={rule['avg_pred_min']:.3f}"
    else:
        score = (
            f"base>={rule['base_pred_min']:.2f}&"
            f"secondary>={rule['secondary_pred_min']:.2f}"
        )
    return (
        f"{rule['mode']}|{score}|"
        f"odds=[{rule['odds_min']:.1f},{rule['odds_max']:.1f})|"
        f"distance={distance}|track={rule['track_label']}|surface={surface}"
    )


def _apply_rule(predictions, rule: dict[str, Any]):
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
        raise ValueError(f"Unknown mode: {rule['mode']}")
    return selected


def _evaluate_rule(
    predictions,
    rule: dict[str, Any],
    stake: float,
    min_fold_selections: int,
    min_fold_return_mid: float | None,
    total_races: int,
) -> dict[str, Any] | None:
    selected = _apply_rule(predictions, rule)
    overall = _selection_summary(selected, stake)
    fold_returns = []
    fold_selections = []
    for _, group in selected.groupby("fold", observed=True):
        fold_summary = _selection_summary(group, stake)
        fold_returns.append(fold_summary["return_mid_pct"])
        fold_selections.append(fold_summary["selections"])

    if len(fold_returns) < predictions["fold"].nunique():
        return None
    if min(fold_selections) < min_fold_selections:
        return None
    if min_fold_return_mid is not None and min(fold_returns) < min_fold_return_mid:
        return None

    result = dict(rule)
    result.update(overall)
    result["rule_key"] = _rule_key(rule)
    result["buy_rate_pct"] = None if total_races == 0 else result["races"] / total_races * 100
    result["min_fold_return_mid_pct"] = min(fold_returns)
    result["max_fold_return_mid_pct"] = max(fold_returns)
    result["min_fold_selections"] = min(fold_selections)
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
    arg_parser.add_argument("--min-selections", type=int, default=70)
    arg_parser.add_argument("--min-fold-selections", type=int, default=10)
    arg_parser.add_argument("--min-fold-return-mid", type=float, default=None)
    arg_parser.add_argument("--min-buy-rate", type=float, default=None)
    arg_parser.add_argument("--max-buy-rate", type=float, default=None)
    arg_parser.add_argument(
        "--modes",
        nargs="+",
        choices=["intersection", "union", "average"],
        default=["intersection", "union", "average"],
    )
    arg_parser.add_argument("--top-n", type=int, default=20)
    args = arg_parser.parse_args()

    predictions = _load_consensus_predictions(
        args.base_predictions,
        args.secondary_predictions,
        args.engine,
    )
    total_races = int(predictions["race_id"].nunique())
    results = []
    for rule in _candidate_rules(args.modes):
        result = _evaluate_rule(
            predictions,
            rule,
            stake=args.stake,
            min_fold_selections=args.min_fold_selections,
            min_fold_return_mid=args.min_fold_return_mid,
            total_races=total_races,
        )
        if result is None or result["selections"] < args.min_selections:
            continue
        if args.min_buy_rate is not None and result["buy_rate_pct"] < args.min_buy_rate:
            continue
        if args.max_buy_rate is not None and result["buy_rate_pct"] > args.max_buy_rate:
            continue
        results.append(result)

    results = sorted(
        results,
        key=lambda row: (
            row["return_mid_pct"],
            row["min_fold_return_mid_pct"],
            row["selections"],
        ),
        reverse=True,
    )

    print(f"Base predictions: {args.base_predictions}")
    print(f"Secondary predictions: {args.secondary_predictions}")
    print(f"Rows: {len(predictions):,}")
    print(f"Races: {total_races:,}")
    print(f"Min selections: {args.min_selections}")
    print(f"Min fold selections: {args.min_fold_selections}")
    print(f"Min fold return mid: {args.min_fold_return_mid}")
    print(f"Min buy rate: {args.min_buy_rate}")
    print(f"Max buy rate: {args.max_buy_rate}")
    print(f"Modes: {','.join(args.modes)}")
    print("")
    print(
        "rule_key                                                        races  buy_rate  selections  "
        "hits  hit_rate  return_mid  min_mid  max_mid  min_fold_n"
    )
    for row in results[: args.top_n]:
        print(
            f"{row['rule_key']:<63} "
            f"{row['races']:>5,}  {_format_pct(row['buy_rate_pct']):>8}  "
            f"{row['selections']:>10,}  {row['hits']:>4,}  "
            f"{_format_pct(row['hit_rate_pct']):>8}  {_format_pct(row['return_mid_pct']):>10}  "
            f"{_format_pct(row['min_fold_return_mid_pct']):>7}  "
            f"{_format_pct(row['max_fold_return_mid_pct']):>7}  "
            f"{row['min_fold_selections']:>10,}"
        )


if __name__ == "__main__":
    main()
