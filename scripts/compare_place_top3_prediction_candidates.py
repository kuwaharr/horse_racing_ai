import argparse
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.evaluate_prediction_consensus_rules import CONSENSUS_RULES, _apply_consensus_rule
from src.data.paths import MODEL_DIR
from src.models.place_top3_lgbm import _apply_fixed_rule, _selection_summary


DEFAULT_PEDIGREE_PREDICTIONS = MODEL_DIR / "catboost_place_top3_predictions_pedigree.parquet"
DEFAULT_NO_PEDIGREE_PREDICTIONS = MODEL_DIR / "catboost_place_top3_predictions_no_pedigree.parquet"


SINGLE_MODEL_CANDIDATES = [
    {
        "name": "no_pedigree_single_mid20_value",
        "prediction_set": "no_pedigree",
        "description": "best current single-model rule near 20 percent buy rate",
        "pred_min": 0.36,
        "odds_min": 3.0,
        "odds_max": 6.0,
        "distance_min": 1600,
        "distance_max": 2200,
        "include_track_ids": None,
        "exclude_track_ids": None,
        "surface_id": None,
    },
    {
        "name": "pedigree_single_mid20_value",
        "prediction_set": "pedigree",
        "description": "best current pedigree-only rule near 20 percent buy rate",
        "pred_min": 0.34,
        "odds_min": 3.0,
        "odds_max": 6.0,
        "distance_min": 1600,
        "distance_max": 2200,
        "include_track_ids": [4, 5, 6, 8, 9],
        "exclude_track_ids": None,
        "surface_id": None,
    },
]


CONSENSUS_CANDIDATES = [
    {
        "name": "pedigree_mid_20_stable",
        "rule_name": "pedigree_mid_20_stable_value",
        "description": "best current pedigree/no-pedigree consensus rule near 20 percent buy rate",
    },
]


def _format_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}%"


def _load_predictions(path: Path, engine: str):
    try:
        import pandas as pd
    except ModuleNotFoundError as e:
        raise RuntimeError("予測候補比較には pandas が必要です。") from e

    return pd.read_parquet(path, engine=engine)


def _candidate_summary(
    name: str,
    description: str,
    selected,
    total_races: int,
    stake: float,
) -> dict[str, Any]:
    fold_returns = []
    fold_selections = []
    for _, group in selected.groupby("fold", observed=True):
        fold_summary = _selection_summary(group, stake)
        fold_returns.append(fold_summary["return_mid_pct"])
        fold_selections.append(fold_summary["selections"])

    summary = _selection_summary(selected, stake)
    summary.update(
        {
            "name": name,
            "description": description,
            "buy_rate_pct": None if total_races == 0 else summary["races"] / total_races * 100,
            "min_fold_return_mid_pct": min(fold_returns) if fold_returns else None,
            "max_fold_return_mid_pct": max(fold_returns) if fold_returns else None,
            "min_fold_selections": min(fold_selections) if fold_selections else 0,
        }
    )
    return summary


def _evaluate_single_candidate(candidate: dict[str, Any], predictions_by_name, total_races: int, stake: float):
    predictions = predictions_by_name[candidate["prediction_set"]]
    selected = _apply_fixed_rule(
        predictions,
        pred_min=candidate["pred_min"],
        odds_min=candidate["odds_min"],
        odds_max=candidate["odds_max"],
        distance_min=candidate["distance_min"],
        distance_max=candidate["distance_max"],
        track_id=None,
        include_track_ids=candidate["include_track_ids"],
        exclude_track_ids=candidate["exclude_track_ids"],
        surface_id=candidate["surface_id"],
    )
    return _candidate_summary(candidate["name"], candidate["description"], selected, total_races, stake)


def _evaluate_consensus_candidate(
    candidate: dict[str, Any],
    pedigree_predictions,
    no_pedigree_predictions,
    total_races: int,
    stake: float,
):
    keys = ["race_id", "horse_number"]
    secondary = no_pedigree_predictions[keys + ["pred_top3"]].rename(
        columns={"pred_top3": "pred_top3_secondary"}
    )
    consensus_predictions = pedigree_predictions.merge(secondary, on=keys, how="inner")
    consensus_predictions = consensus_predictions.rename(columns={"pred_top3": "pred_top3_base"})
    consensus_predictions["pred_top3_avg"] = (
        consensus_predictions["pred_top3_base"] + consensus_predictions["pred_top3_secondary"]
    ) / 2

    rules_by_name = {rule["name"]: rule for rule in CONSENSUS_RULES}
    selected = _apply_consensus_rule(consensus_predictions, rules_by_name[candidate["rule_name"]])
    return _candidate_summary(candidate["name"], candidate["description"], selected, total_races, stake)


def main() -> None:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--pedigree-predictions", type=Path, default=DEFAULT_PEDIGREE_PREDICTIONS)
    arg_parser.add_argument("--no-pedigree-predictions", type=Path, default=DEFAULT_NO_PEDIGREE_PREDICTIONS)
    arg_parser.add_argument("--engine", choices=["auto", "pyarrow", "fastparquet"], default="auto")
    arg_parser.add_argument("--stake", type=float, default=100.0)
    args = arg_parser.parse_args()

    pedigree_predictions = _load_predictions(args.pedigree_predictions, args.engine)
    no_pedigree_predictions = _load_predictions(args.no_pedigree_predictions, args.engine)
    total_races = int(pedigree_predictions["race_id"].nunique())
    predictions_by_name = {
        "pedigree": pedigree_predictions,
        "no_pedigree": no_pedigree_predictions,
    }

    rows = [
        _evaluate_single_candidate(candidate, predictions_by_name, total_races, args.stake)
        for candidate in SINGLE_MODEL_CANDIDATES
    ]
    rows.extend(
        _evaluate_consensus_candidate(
            candidate,
            pedigree_predictions,
            no_pedigree_predictions,
            total_races,
            args.stake,
        )
        for candidate in CONSENSUS_CANDIDATES
    )
    rows = sorted(
        rows,
        key=lambda row: (
            row["return_mid_pct"],
            row["min_fold_return_mid_pct"],
            row["selections"],
        ),
        reverse=True,
    )

    print(f"Pedigree predictions: {args.pedigree_predictions}")
    print(f"No-pedigree predictions: {args.no_pedigree_predictions}")
    print(f"Races: {total_races:,}")
    print("")
    print(
        "candidate                         races  buy_rate  selections  hits  hit_rate  "
        "return_mid  min_mid  max_mid  min_fold_n  description"
    )
    for row in rows:
        print(
            f"{row['name']:<33} {row['races']:>5,}  {_format_pct(row['buy_rate_pct']):>8}  "
            f"{row['selections']:>10,}  {row['hits']:>4,}  "
            f"{_format_pct(row['hit_rate_pct']):>8}  "
            f"{_format_pct(row['return_mid_pct']):>10}  "
            f"{_format_pct(row['min_fold_return_mid_pct']):>7}  "
            f"{_format_pct(row['max_fold_return_mid_pct']):>7}  "
            f"{row['min_fold_selections']:>10,}  {row['description']}"
        )


if __name__ == "__main__":
    main()
