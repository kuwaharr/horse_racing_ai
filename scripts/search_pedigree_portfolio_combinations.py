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
from scripts.evaluate_prediction_consensus_rules import (
    _load_consensus_predictions,
    _apply_consensus_rule
)

PEDIGREE_MID_20_STABLE_RULE = {
    "name": "pedigree_mid_20_stable_value",
    "mode": "intersection",
    "base_pred_min": 0.34,
    "secondary_pred_min": 0.36,
    "avg_pred_min": None,
    "odds_min": 3.2,
    "odds_max": 6.0,
    "distance_min": None,
    "distance_max": None,
    "include_track_ids": [4, 5, 6, 8, 9],
    "exclude_track_ids": None,
    "surface_id": None,
}

TRIO_HIGH_ROI_RULE = {
    "score_topn": 10,
    "rank_max": 4,
    "trio_score_min": 0.0,
    "trio_min_pred_min": 0.0,
    "ev_mid_min": 0.0,
    "odds_min": 20.0,
    "odds_max": 200.0,
    "surface_label": "turf",
    "surface_id": 0,
    "distance_label": "1800_2600",
    "distance_min": 1800,
    "distance_max": 2600,
    "track_label": "exclude_1_2_3_7_10",
    "exclude_track_ids": (1, 2, 3, 7, 10),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--win-predictions", type=Path, default=DEFAULT_WIN_PREDICTIONS)
    parser.add_argument("--place-predictions", type=Path, default=DEFAULT_PLACE_PREDICTIONS)
    parser.add_argument("--exotic-predictions", type=Path, default=DEFAULT_EXOTIC_PREDICTIONS)
    parser.add_argument("--pedigree-predictions", type=Path, default=ROOT_DIR / "local_models" / "catboost_place_top3_model.cbm" or Path("D:/horse_racing_ai/data/model/catboost_place_top3_predictions_pedigree.parquet"))
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--engine", choices=["auto", "pyarrow", "fastparquet"], default="auto")
    parser.add_argument("--stake", type=float, default=100.0)
    parser.add_argument("--min-buy-rate", type=float, default=20.0)
    parser.add_argument("--max-buy-rate", type=float, default=100.0)
    parser.add_argument("--min-fold-return", type=float, default=90.0)
    parser.add_argument("--top-n", type=int, default=50)
    args = parser.parse_args()

    # Actual path check
    ped_pred_path = Path("D:/horse_racing_ai/data/model/catboost_place_top3_predictions_pedigree.parquet")
    noped_pred_path = Path("D:/horse_racing_ai/data/model/catboost_place_top3_predictions_no_pedigree.parquet")

    # Load predictions
    win_predictions = _load_rule_predictions(args.win_predictions, args.engine)
    place_predictions = _load_rule_predictions(args.place_predictions, args.engine)
    exotic_predictions = _load_trio_predictions(args.exotic_predictions, args.engine)
    
    # Load pedigree consensus predictions
    pedigree_consensus_df = _load_consensus_predictions(ped_pred_path, noped_pred_path, args.engine)
    
    # Generate selections for pedigree_mid_20_stable
    ped_selected = _apply_consensus_rule(pedigree_consensus_df, PEDIGREE_MID_20_STABLE_RULE)
    ped_selected["bet_type"] = "place"
    ped_selected["strategy"] = "pedigree_mid_20_stable"
    ped_selected["stake"] = args.stake
    ped_selected["hit"] = ped_selected["target_top3"].astype("int8")
    ped_selected["odds"] = ped_selected["place_odds_mid"]
    ped_selected["payout"] = ped_selected["hit"] * ped_selected["odds"] * args.stake
    ped_selected["selection_key"] = ped_selected["horse_number"].astype(str)
    
    pedigree_selections = ped_selected[
        ["race_id", "fold", "bet_type", "strategy", "selection_key", "stake", "hit", "odds", "payout"]
    ]

    strategy_races = {
        "win_stable": set(win_predictions["race_id"].unique()),
        "win_roi": set(win_predictions["race_id"].unique()),
        "win_broad": set(win_predictions["race_id"].unique()),
        "place_latest": set(place_predictions["race_id"].unique()),
        "place_broad": set(place_predictions["race_id"].unique()),
        "place_stable": set(place_predictions["race_id"].unique()),
        "pedigree_mid_20_stable": set(pedigree_consensus_df["race_id"].unique()),
        "wide_roi": set(exotic_predictions["race_id"].unique()),
        "wide_broad": set(exotic_predictions["race_id"].unique()),
        "trio_roi": set(exotic_predictions["race_id"].unique()),
        "trio_balanced": set(exotic_predictions["race_id"].unique()),
        "trio_hit_rate": set(exotic_predictions["race_id"].unique()),
        "trio_high_roi": set(exotic_predictions["race_id"].unique()),
    }

    selections_by_strategy = {
        "win_stable": _single_selections(win_predictions, WIN_STABLE_RULE, "win", "win_stable", args.stake),
        "win_roi": _single_selections(win_predictions, WIN_ROI_RULE, "win", "win_roi", args.stake),
        "win_broad": _single_selections(win_predictions, WIN_BROAD_RULE, "win", "win_broad", args.stake),
        "place_latest": _single_selections(place_predictions, PLACE_RULE, "place", "place_latest", args.stake),
        "place_broad": _single_selections(place_predictions, PLACE_BROAD_RULE, "place", "place_broad", args.stake),
        "place_stable": _single_selections(place_predictions, PLACE_STABLE_RULE, "place", "place_stable", args.stake),
        "pedigree_mid_20_stable": pedigree_selections,
        "wide_roi": _wide_selections(args.exotic_predictions, args.db, args.engine, args.stake, WIDE_ROI_RULE, "wide_roi"),
        "wide_broad": _wide_selections(args.exotic_predictions, args.db, args.engine, args.stake, WIDE_BROAD_RULE, "wide_broad"),
        "trio_roi": _trio_selections(args.exotic_predictions, args.db, args.engine, args.stake, TRIO_ROI_RULE, "trio_roi"),
        "trio_balanced": _trio_selections(args.exotic_predictions, args.db, args.engine, args.stake, TRIO_BALANCED_RULE, "trio_balanced"),
        "trio_hit_rate": _trio_selections(args.exotic_predictions, args.db, args.engine, args.stake, TRIO_HIT_RATE_RULE, "trio_hit_rate"),
        "trio_high_roi": _trio_selections(args.exotic_predictions, args.db, args.engine, args.stake, TRIO_HIGH_ROI_RULE, "trio_high_roi"),
    }

    strategy_names = list(selections_by_strategy.keys())
    results = []

    print("Evaluating all portfolio combinations (including pedigree)...")
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

    print(f"\nTop {args.top_n} Combinations (including Pedigree):")
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
