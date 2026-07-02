import argparse
import sys
from itertools import combinations
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.evaluate_bet_portfolios import (
    WIN_STABLE_RULE, WIN_ROI_RULE, WIN_BROAD_RULE,
    PLACE_RULE, PLACE_BROAD_RULE, PLACE_STABLE_RULE,
    WIDE_ROI_RULE, WIDE_BROAD_RULE,
    TRIO_ROI_RULE, TRIO_BALANCED_RULE, TRIO_HIT_RATE_RULE,
    DEFAULT_WIN_PREDICTIONS, DEFAULT_PLACE_PREDICTIONS, DEFAULT_EXOTIC_PREDICTIONS,
    DB_PATH,
    _load_rule_predictions,
    _load_trio_predictions,
    _single_selections,
    _wide_selections,
    _trio_selections,
    _summary,
    _format_pct
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--win-predictions", type=Path, default=DEFAULT_WIN_PREDICTIONS)
    parser.add_argument("--place-predictions", type=Path, default=DEFAULT_PLACE_PREDICTIONS)
    parser.add_argument("--exotic-predictions", type=Path, default=DEFAULT_EXOTIC_PREDICTIONS)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--engine", choices=["auto", "pyarrow", "fastparquet"], default="auto")
    parser.add_argument("--stake", type=float, default=100.0)
    parser.add_argument("--min-buy-rate", type=float, default=20.0)
    parser.add_argument("--max-buy-rate", type=float, default=100.0)
    parser.add_argument("--min-fold-return", type=float, default=100.0)
    parser.add_argument("--top-n", type=int, default=50)
    args = parser.parse_args()

    # Load predictions
    win_predictions = _load_rule_predictions(args.win_predictions, args.engine)
    place_predictions = _load_rule_predictions(args.place_predictions, args.engine)
    exotic_predictions = _load_trio_predictions(args.exotic_predictions, args.engine)

    strategy_races = {
        "win_stable": set(win_predictions["race_id"].unique()),
        "win_roi": set(win_predictions["race_id"].unique()),
        "win_broad": set(win_predictions["race_id"].unique()),
        "place_latest": set(place_predictions["race_id"].unique()),
        "place_broad": set(place_predictions["race_id"].unique()),
        "place_stable": set(place_predictions["race_id"].unique()),
        "wide_roi": set(exotic_predictions["race_id"].unique()),
        "wide_broad": set(exotic_predictions["race_id"].unique()),
        "trio_roi": set(exotic_predictions["race_id"].unique()),
        "trio_balanced": set(exotic_predictions["race_id"].unique()),
        "trio_hit_rate": set(exotic_predictions["race_id"].unique()),
    }

    selections_by_strategy = {
        "win_stable": _single_selections(win_predictions, WIN_STABLE_RULE, "win", "win_stable", args.stake),
        "win_roi": _single_selections(win_predictions, WIN_ROI_RULE, "win", "win_roi", args.stake),
        "win_broad": _single_selections(win_predictions, WIN_BROAD_RULE, "win", "win_broad", args.stake),
        "place_latest": _single_selections(place_predictions, PLACE_RULE, "place", "place_latest", args.stake),
        "place_broad": _single_selections(place_predictions, PLACE_BROAD_RULE, "place", "place_broad", args.stake),
        "place_stable": _single_selections(place_predictions, PLACE_STABLE_RULE, "place", "place_stable", args.stake),
        "wide_roi": _wide_selections(args.exotic_predictions, args.db, args.engine, args.stake, WIDE_ROI_RULE, "wide_roi"),
        "wide_broad": _wide_selections(args.exotic_predictions, args.db, args.engine, args.stake, WIDE_BROAD_RULE, "wide_broad"),
        "trio_roi": _trio_selections(args.exotic_predictions, args.db, args.engine, args.stake, TRIO_ROI_RULE, "trio_roi"),
        "trio_balanced": _trio_selections(args.exotic_predictions, args.db, args.engine, args.stake, TRIO_BALANCED_RULE, "trio_balanced"),
        "trio_hit_rate": _trio_selections(args.exotic_predictions, args.db, args.engine, args.stake, TRIO_HIT_RATE_RULE, "trio_hit_rate"),
    }

    strategy_names = list(selections_by_strategy.keys())
    results = []

    # Generate all subsets of sizes from 1 to len(strategy_names)
    print("Evaluating all portfolio combinations...")
    for k in range(1, len(strategy_names) + 1):
        for combo in combinations(strategy_names, k):
            selected = pd.concat([selections_by_strategy[name] for name in combo], ignore_index=True)
            total_races = len(set().union(*(strategy_races[name] for name in combo)))
            
            row = _summary(selected, total_races, "+".join(combo))
            buy_rate = row["buy_rate_pct"]
            if buy_rate is None or buy_rate < args.min_buy_rate or buy_rate > args.max_buy_rate:
                continue
            
            min_fold_return = row["min_fold_return_pct"]
            if args.min_fold_return is not None and (min_fold_return is None or min_fold_return < args.min_fold_return):
                continue
                
            results.append(row)

    results = sorted(
        results,
        key=lambda row: (row["return_pct"] or 0, row["min_fold_return_pct"] or 0),
        reverse=True
    )

    print(f"\nTop {args.top_n} Combinations (Min Buy Rate: {args.min_buy_rate}%, Min Fold Return: {args.min_fold_return}%):")
    print("name                 buy_rate  races  selections  hits  hit_rate  ROI     min_fold")
    for row in results[:args.top_n]:
        print(
            f"{row['name'][:20]:<20} "
            f"{_format_pct(row['buy_rate_pct']):>8}  "
            f"{row['races']:>5,}  {row['selections']:>10,}  {row['hits']:>4,}  "
            f"{_format_pct(row['hit_rate_pct']):>8}  {_format_pct(row['return_pct']):>7}  "
            f"{_format_pct(row['min_fold_return_pct']):>8}"
        )

if __name__ == "__main__":
    main()
