import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.evaluate_prediction_consensus_portfolios import PORTFOLIOS
from scripts.evaluate_prediction_consensus_rules import (
    CONSENSUS_RULES,
    DEFAULT_BASE_PREDICTIONS,
    DEFAULT_SECONDARY_PREDICTIONS,
    _apply_consensus_rule,
    _load_consensus_predictions,
)
from src.data.paths import MODEL_DIR


DEFAULT_OUTPUT = MODEL_DIR / "consensus_portfolio_selections.csv"


def _track_text(rule: dict) -> str:
    if rule["include_track_ids"] is not None:
        return "include:" + ",".join(str(v) for v in rule["include_track_ids"])
    if rule["exclude_track_ids"] is not None:
        return "exclude:" + ",".join(str(v) for v in rule["exclude_track_ids"])
    return "all"


def _rule_condition_text(rule: dict) -> str:
    surface = "all" if rule["surface_id"] is None else str(rule["surface_id"])
    if rule["mode"] == "average":
        score = f"avg>={rule['avg_pred_min']:.3f}"
    else:
        joiner = "|" if rule["mode"] == "union" else "&"
        score = (
            f"base>={rule['base_pred_min']:.2f}"
            f"{joiner}secondary>={rule['secondary_pred_min']:.2f}"
        )
    return (
        f"mode={rule['mode']};{score};"
        f"odds=[{rule['odds_min']:.1f},{rule['odds_max']:.1f});"
        f"distance=[{rule['distance_min']},{rule['distance_max']});"
        f"track={_track_text(rule)};surface={surface}"
    )


def _portfolio_selections(predictions, rules_by_name: dict, portfolio: dict):
    try:
        import pandas as pd
    except ModuleNotFoundError as e:
        raise RuntimeError("合議ポートフォリオの候補出力には pandas が必要です。") from e

    parts = []
    for rule_name in portfolio["rules"]:
        rule = rules_by_name[rule_name]
        selected = _apply_consensus_rule(predictions, rule).copy()
        selected["matched_rule"] = rule_name
        selected["matched_condition"] = _rule_condition_text(rule)
        parts.append(selected)

    raw = pd.concat(parts, ignore_index=True)
    rule_matches = (
        raw.groupby(["race_id", "horse_number"], observed=True)["matched_rule"]
        .apply(lambda values: ",".join(sorted(set(values))))
        .rename("matched_rules")
        .reset_index()
    )
    condition_matches = (
        raw.groupby(["race_id", "horse_number"], observed=True)["matched_condition"]
        .apply(lambda values: " || ".join(sorted(set(values))))
        .rename("matched_conditions")
        .reset_index()
    )
    deduped = raw.sort_values(["race_id", "horse_number", "matched_rule"]).drop_duplicates(
        ["race_id", "horse_number"]
    )
    deduped = deduped.drop(columns=["matched_rule", "matched_condition"])
    deduped = deduped.merge(rule_matches, on=["race_id", "horse_number"], how="left")
    deduped = deduped.merge(condition_matches, on=["race_id", "horse_number"], how="left")
    deduped["portfolio"] = portfolio["name"]
    deduped["portfolio_description"] = portfolio["description"]
    return deduped


def main() -> None:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--base-predictions", type=Path, default=DEFAULT_BASE_PREDICTIONS)
    arg_parser.add_argument(
        "--secondary-predictions",
        type=Path,
        default=DEFAULT_SECONDARY_PREDICTIONS,
    )
    arg_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arg_parser.add_argument("--engine", choices=["auto", "pyarrow", "fastparquet"], default="auto")
    arg_parser.add_argument(
        "--portfolio",
        choices=[portfolio["name"] for portfolio in PORTFOLIOS],
        default="value_plus_consensus",
    )
    args = arg_parser.parse_args()

    predictions = _load_consensus_predictions(
        args.base_predictions,
        args.secondary_predictions,
        args.engine,
    )
    rules_by_name = {rule["name"]: rule for rule in CONSENSUS_RULES}
    portfolios_by_name = {portfolio["name"]: portfolio for portfolio in PORTFOLIOS}
    selections = _portfolio_selections(
        predictions,
        rules_by_name,
        portfolios_by_name[args.portfolio],
    )
    selections = selections.sort_values(
        ["date", "race_id", "pred_top3_avg", "place_odds_mid"],
        ascending=[True, True, False, True],
    )

    output_columns = [
        "portfolio",
        "portfolio_description",
        "matched_rules",
        "matched_conditions",
        "fold",
        "date",
        "race_id",
        "horse_number",
        "target_top3",
        "pred_top3_base",
        "pred_top3_secondary",
        "pred_top3_avg",
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

    print(f"Base predictions: {args.base_predictions}")
    print(f"Secondary predictions: {args.secondary_predictions}")
    print(f"Portfolio: {args.portfolio}")
    print(f"Output: {args.output}")
    print(f"Rows: {len(selections):,}")
    print(f"Races: {int(selections['race_id'].nunique()):,}")
    print("")
    print("matched_rules                         rows")
    for matched_rules, group in selections.groupby("matched_rules", sort=False):
        print(f"{matched_rules:<35} {len(group):>5,}")


if __name__ == "__main__":
    main()
