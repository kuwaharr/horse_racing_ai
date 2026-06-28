import argparse
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.evaluate_prediction_consensus_portfolios import (
    PORTFOLIOS,
    _summarize_portfolio,
)
from scripts.evaluate_prediction_consensus_rules import (
    CONSENSUS_RULES,
    _load_consensus_predictions,
)
from scripts.search_prediction_consensus_rules import (
    _candidate_rules,
    _evaluate_rule,
)
from src.data.paths import FEAT_DIR, MODEL_DIR
from src.features.place_top3 import default_eval_odds_dataset_name, default_training_dataset_name
from src.models.place_top3_catboost import build_catboost_walk_forward_predictions


PEDIGREE_DROP_PATTERNS = [
    "sire_",
    "dam_",
    "broodmare_sire_",
    "race_sire_",
    "race_broodmare_sire_",
    "pedigree_available",
]


def _format_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}%"


def _write_predictions(
    training_dataset: Path,
    odds_dataset: Path,
    output: Path,
    engine: str,
    n_splits: int,
    min_train_ratio: float,
    drop_feature_patterns: list[str] | None,
) -> dict[str, Any]:
    report = build_catboost_walk_forward_predictions(
        training_dataset_path=training_dataset,
        odds_dataset_path=odds_dataset,
        engine=engine,
        n_splits=n_splits,
        min_train_ratio=min_train_ratio,
        drop_feature_patterns=drop_feature_patterns,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    report["predictions"].to_parquet(output, index=False, engine=engine)
    return report


def _print_prediction_report(label: str, output: Path, report: dict[str, Any]) -> None:
    print(f"{label}: {output}")
    print(f"  rows: {len(report['predictions']):,}")
    print(f"  dropped features: {len(report['dropped_features'])}")
    print("  fold  test_start  test_end    train_rows  test_rows  test_races      AUC")
    for row in report["folds"]:
        test_end = row["test_end"] or "end"
        print(
            f"  {row['fold']:>4}  {row['test_start']}  {test_end:<10}  "
            f"{row['train_rows']:>10,}  {row['test_rows']:>9,}  "
            f"{row['test_races']:>10,}  {row['auc']:>7.5f}"
        )


def _print_portfolios(predictions, stake: float) -> None:
    total_races = int(predictions["race_id"].nunique())
    rules_by_name = {rule["name"]: rule for rule in CONSENSUS_RULES}
    rows = [
        _summarize_portfolio(predictions, rules_by_name, portfolio, stake)
        for portfolio in PORTFOLIOS
    ]

    print("")
    print("Portfolio results")
    print("portfolio              races  buy_rate  selections  hits  hit_rate  return_mid  min_mid")
    for row in rows:
        buy_rate_pct = None if total_races == 0 else row["races"] / total_races * 100
        print(
            f"{row['name']:<22} {row['races']:>5,}  {_format_pct(buy_rate_pct):>8}  "
            f"{row['selections']:>10,}  {row['hits']:>4,}  "
            f"{_format_pct(row['hit_rate_pct']):>8}  "
            f"{_format_pct(row['return_mid_pct']):>10}  "
            f"{_format_pct(row['min_fold_return_mid_pct']):>7}"
        )


def _print_rule_search(
    predictions,
    stake: float,
    min_buy_rate: float,
    max_buy_rate: float,
    min_selections: int,
    min_fold_selections: int,
    top_n: int,
) -> None:
    total_races = int(predictions["race_id"].nunique())
    results = []
    for rule in _candidate_rules(["intersection", "union", "average"]):
        result = _evaluate_rule(
            predictions,
            rule,
            stake=stake,
            min_fold_selections=min_fold_selections,
            min_fold_return_mid=None,
            total_races=total_races,
        )
        if result is None or result["selections"] < min_selections:
            continue
        if result["buy_rate_pct"] < min_buy_rate or result["buy_rate_pct"] > max_buy_rate:
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

    print("")
    print(f"Top searched rules ({min_buy_rate:.1f}-{max_buy_rate:.1f}% buy rate)")
    print("rule_key                                                        races  buy_rate  selections  return_mid  min_mid")
    for row in results[:top_n]:
        print(
            f"{row['rule_key']:<63} {row['races']:>5,}  "
            f"{_format_pct(row['buy_rate_pct']):>8}  {row['selections']:>10,}  "
            f"{_format_pct(row['return_mid_pct']):>10}  "
            f"{_format_pct(row['min_fold_return_mid_pct']):>7}"
        )


def main() -> None:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--training-dataset", type=Path, default=FEAT_DIR / default_training_dataset_name())
    arg_parser.add_argument("--odds-dataset", type=Path, default=FEAT_DIR / default_eval_odds_dataset_name())
    arg_parser.add_argument(
        "--pedigree-predictions",
        type=Path,
        default=MODEL_DIR / "catboost_place_top3_predictions_pedigree.parquet",
    )
    arg_parser.add_argument(
        "--no-pedigree-predictions",
        type=Path,
        default=MODEL_DIR / "catboost_place_top3_predictions_no_pedigree.parquet",
    )
    arg_parser.add_argument("--engine", choices=["auto", "pyarrow", "fastparquet"], default="auto")
    arg_parser.add_argument("--n-splits", type=int, default=4)
    arg_parser.add_argument("--min-train-ratio", type=float, default=0.5)
    arg_parser.add_argument("--stake", type=float, default=100.0)
    arg_parser.add_argument("--skip-predictions", action="store_true")
    arg_parser.add_argument("--min-buy-rate", type=float, default=18.0)
    arg_parser.add_argument("--max-buy-rate", type=float, default=22.0)
    arg_parser.add_argument("--min-selections", type=int, default=70)
    arg_parser.add_argument("--min-fold-selections", type=int, default=10)
    arg_parser.add_argument("--top-n", type=int, default=10)
    args = arg_parser.parse_args()

    print(f"Training dataset: {args.training_dataset}")
    print(f"Evaluation odds dataset: {args.odds_dataset}")

    if not args.skip_predictions:
        pedigree_report = _write_predictions(
            training_dataset=args.training_dataset,
            odds_dataset=args.odds_dataset,
            output=args.pedigree_predictions,
            engine=args.engine,
            n_splits=args.n_splits,
            min_train_ratio=args.min_train_ratio,
            drop_feature_patterns=None,
        )
        no_pedigree_report = _write_predictions(
            training_dataset=args.training_dataset,
            odds_dataset=args.odds_dataset,
            output=args.no_pedigree_predictions,
            engine=args.engine,
            n_splits=args.n_splits,
            min_train_ratio=args.min_train_ratio,
            drop_feature_patterns=PEDIGREE_DROP_PATTERNS,
        )
        print("")
        _print_prediction_report("Pedigree predictions", args.pedigree_predictions, pedigree_report)
        _print_prediction_report("No-pedigree predictions", args.no_pedigree_predictions, no_pedigree_report)

    predictions = _load_consensus_predictions(
        args.pedigree_predictions,
        args.no_pedigree_predictions,
        args.engine,
    )
    print("")
    print(f"Consensus rows: {len(predictions):,}")
    print(f"Consensus races: {int(predictions['race_id'].nunique()):,}")
    _print_portfolios(predictions, args.stake)
    _print_rule_search(
        predictions,
        stake=args.stake,
        min_buy_rate=args.min_buy_rate,
        max_buy_rate=args.max_buy_rate,
        min_selections=args.min_selections,
        min_fold_selections=args.min_fold_selections,
        top_n=args.top_n,
    )


if __name__ == "__main__":
    main()
