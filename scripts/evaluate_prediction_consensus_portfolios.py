import argparse
import sys
from pathlib import Path
from typing import Any

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


PORTFOLIOS = [
    {
        "name": "value_odds_only",
        "description": "highest stable return, fewer bets",
        "rules": ["union_value_odds"],
    },
    {
        "name": "turf_value_only",
        "description": "highest return, very few bets",
        "rules": ["union_value_odds_turf"],
    },
    {
        "name": "value_plus_consensus",
        "description": "balanced volume and return",
        "rules": ["union_value_odds", "consensus_high_return"],
    },
    {
        "name": "value_plus_precise",
        "description": "slightly smaller balanced portfolio",
        "rules": ["union_value_odds", "consensus_precise"],
    },
    {
        "name": "value_plus_volume",
        "description": "more bets, lower return",
        "rules": ["union_value_odds", "consensus_high_return", "consensus_volume"],
    },
    {
        "name": "wide_value",
        "description": "near 10 percent race coverage, lower return",
        "rules": ["union_wide_value"],
    },
    {
        "name": "wide_plus_consensus",
        "description": "near 10 percent coverage with high-return consensus add-on",
        "rules": ["union_wide_value", "consensus_high_return"],
    },
    {
        "name": "broad_30_value",
        "description": "near 30 percent race coverage, lower return",
        "rules": ["union_broad_30_value"],
    },
    {
        "name": "broad_37_value",
        "description": "higher race coverage, lower return",
        "rules": ["union_broad_37_value"],
    },
]


def _format_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}%"


def _portfolio_selections(predictions, rules_by_name: dict[str, dict[str, Any]], rule_names: list[str]):
    try:
        import pandas as pd
    except ModuleNotFoundError as e:
        raise RuntimeError("合議ポートフォリオ評価には pandas が必要です。") from e

    parts = []
    for rule_name in rule_names:
        selected = _apply_consensus_rule(predictions, rules_by_name[rule_name]).copy()
        selected["portfolio_rule_name"] = rule_name
        parts.append(selected)
    raw = pd.concat(parts, ignore_index=True)
    matched_rules = (
        raw.groupby(["race_id", "horse_number"], observed=True)["portfolio_rule_name"]
        .apply(lambda values: ",".join(sorted(set(values))))
        .rename("matched_rules")
        .reset_index()
    )
    deduped = raw.sort_values(["race_id", "horse_number", "portfolio_rule_name"])
    deduped = deduped.drop_duplicates(["race_id", "horse_number"])
    return deduped.merge(matched_rules, on=["race_id", "horse_number"], how="left")


def _summarize_portfolio(
    predictions,
    rules_by_name: dict[str, dict[str, Any]],
    portfolio: dict[str, Any],
    stake: float,
) -> dict[str, Any]:
    selected = _portfolio_selections(predictions, rules_by_name, portfolio["rules"])
    overall = _selection_summary(selected, stake)
    fold_returns = []
    fold_selections = []
    for _, group in selected.groupby("fold", observed=True):
        fold_summary = _selection_summary(group, stake)
        fold_returns.append(fold_summary["return_mid_pct"])
        fold_selections.append(fold_summary["selections"])

    result = dict(portfolio)
    result.update(overall)
    result["min_fold_return_mid_pct"] = min(fold_returns) if fold_returns else None
    result["max_fold_return_mid_pct"] = max(fold_returns) if fold_returns else None
    result["min_fold_selections"] = min(fold_selections) if fold_selections else 0
    return result


def _summarize_group(selected, stake: float) -> dict[str, Any]:
    summary = _selection_summary(selected, stake)
    fold_returns = []
    fold_selections = []
    for _, group in selected.groupby("fold", observed=True):
        fold_summary = _selection_summary(group, stake)
        fold_returns.append(fold_summary["return_mid_pct"])
        fold_selections.append(fold_summary["selections"])
    summary["min_fold_return_mid_pct"] = min(fold_returns) if fold_returns else None
    summary["max_fold_return_mid_pct"] = max(fold_returns) if fold_returns else None
    summary["min_fold_selections"] = min(fold_selections) if fold_selections else 0
    return summary


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
    arg_parser.add_argument(
        "--breakdown-portfolio",
        choices=[portfolio["name"] for portfolio in PORTFOLIOS],
        default=None,
    )
    args = arg_parser.parse_args()

    predictions = _load_consensus_predictions(
        args.base_predictions,
        args.secondary_predictions,
        args.engine,
    )
    rules_by_name = {rule["name"]: rule for rule in CONSENSUS_RULES}
    rows = [
        _summarize_portfolio(predictions, rules_by_name, portfolio, args.stake)
        for portfolio in PORTFOLIOS
    ]

    print(f"Base predictions: {args.base_predictions}")
    print(f"Secondary predictions: {args.secondary_predictions}")
    print(f"Rows: {len(predictions):,}")
    print(f"Races: {int(predictions['race_id'].nunique()):,}")
    print("")
    print(
        "portfolio              races  selections  hits  hit_rate  return_mid  "
        "min_mid  max_mid  min_fold_n  rules"
    )
    for row in rows:
        print(
            f"{row['name']:<22} {row['races']:>5,}  {row['selections']:>10,}  "
            f"{row['hits']:>4,}  {_format_pct(row['hit_rate_pct']):>8}  "
            f"{_format_pct(row['return_mid_pct']):>10}  "
            f"{_format_pct(row['min_fold_return_mid_pct']):>7}  "
            f"{_format_pct(row['max_fold_return_mid_pct']):>7}  "
            f"{row['min_fold_selections']:>10,}  {', '.join(row['rules'])}"
        )

    if args.breakdown_portfolio is not None:
        portfolios_by_name = {portfolio["name"]: portfolio for portfolio in PORTFOLIOS}
        portfolio = portfolios_by_name[args.breakdown_portfolio]
        selected = _portfolio_selections(predictions, rules_by_name, portfolio["rules"])
        print("")
        print(f"Breakdown for {args.breakdown_portfolio}")
        print(
            "matched_rules                         races  selections  hits  hit_rate  "
            "return_mid  min_mid  max_mid  min_fold_n"
        )
        for matched_rules, group in selected.groupby("matched_rules", sort=False):
            row = _summarize_group(group, args.stake)
            print(
                f"{matched_rules:<35} {row['races']:>5,}  {row['selections']:>10,}  "
                f"{row['hits']:>4,}  {_format_pct(row['hit_rate_pct']):>8}  "
                f"{_format_pct(row['return_mid_pct']):>10}  "
                f"{_format_pct(row['min_fold_return_mid_pct']):>7}  "
                f"{_format_pct(row['max_fold_return_mid_pct']):>7}  "
                f"{row['min_fold_selections']:>10,}"
            )


if __name__ == "__main__":
    main()
