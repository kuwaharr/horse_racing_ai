import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.evaluate_prediction_consensus_rules import (
    CONSENSUS_RULES,
    DEFAULT_BASE_PREDICTIONS,
    DEFAULT_SECONDARY_PREDICTIONS,
    _apply_consensus_rule,
    _load_consensus_predictions,
)
from src.models.place_top3_lgbm import _selection_summary


def _format_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}%"


def _print_breakdown(selected, group_col: str, stake: float) -> None:
    print("")
    print(f"Breakdown by {group_col}")
    print("group       races  selections  hits  hit_rate  return_mid  return_min  return_max")
    for value, group in selected.groupby(group_col, observed=True):
        summary = _selection_summary(group, stake)
        print(
            f"{str(value):<10} {summary['races']:>5,}  {summary['selections']:>10,}  "
            f"{summary['hits']:>4,}  {_format_pct(summary['hit_rate_pct']):>8}  "
            f"{_format_pct(summary['return_mid_pct']):>10}  "
            f"{_format_pct(summary['return_min_pct']):>10}  "
            f"{_format_pct(summary['return_max_pct']):>10}"
        )


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
    arg_parser.add_argument("--rule-name", default="union_value_odds")
    args = arg_parser.parse_args()

    try:
        import pandas as pd
    except ModuleNotFoundError as e:
        raise RuntimeError("合議ルールの分解確認には pandas が必要です。") from e

    rules = {rule["name"]: rule for rule in CONSENSUS_RULES}
    if args.rule_name not in rules:
        choices = ", ".join(sorted(rules))
        raise ValueError(f"Unknown rule name: {args.rule_name}. choices: {choices}")

    predictions = _load_consensus_predictions(
        args.base_predictions,
        args.secondary_predictions,
        args.engine,
    )
    selected = _apply_consensus_rule(predictions, rules[args.rule_name]).copy()
    selected["date"] = pd.to_datetime(selected["date"])
    selected["year_month"] = selected["date"].dt.strftime("%Y-%m")

    overall = _selection_summary(selected, args.stake)
    print(f"Rule: {args.rule_name}")
    print(f"Base predictions: {args.base_predictions}")
    print(f"Secondary predictions: {args.secondary_predictions}")
    print(
        "Overall: "
        f"races={overall['races']:,} selections={overall['selections']:,} "
        f"hits={overall['hits']:,} hit_rate={_format_pct(overall['hit_rate_pct'])} "
        f"return_mid={_format_pct(overall['return_mid_pct'])}"
    )

    for group_col in ["fold", "year_month", "track_id", "surface_id"]:
        _print_breakdown(selected, group_col, args.stake)


if __name__ == "__main__":
    main()
