import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.search_trio_rules_from_predictions import (  # noqa: E402
    _build_context as _build_trio_context,
    _build_trio_combinations,
    _load_predictions as _load_trio_predictions,
    _load_trio_odds,
    _mask_for_rule as _mask_for_trio_rule,
)
from scripts.search_wide_rules_from_predictions import (  # noqa: E402
    _build_context as _build_wide_context,
    _build_wide_pairs,
    _load_predictions as _load_wide_predictions,
    _load_wide_odds,
    _mask_for_rule as _mask_for_wide_rule,
)
from src.data.paths import DB_PATH, MODEL_DIR  # noqa: E402
from scripts.evaluate_prediction_consensus_rules import (  # noqa: E402
    _load_consensus_predictions,
    _apply_consensus_rule
)


DEFAULT_WIN_PREDICTIONS = MODEL_DIR / "catboost_win_top1_predictions.parquet"
DEFAULT_PLACE_PREDICTIONS = MODEL_DIR / "catboost_place_top3_predictions.parquet"
DEFAULT_EXOTIC_PREDICTIONS = MODEL_DIR / "catboost_place_top3_predictions_affinity_lift_no_horse_id.parquet"
DEFAULT_OUTPUT = MODEL_DIR / "bet_portfolio_results.csv"

WIN_STABLE_RULE = {
    "pred_min": 0.15,
    "odds_min": 1.2,
    "odds_max": 3.5,
    "distance_min": 1600,
    "distance_max": None,
    "include_track_ids": [4, 5, 6, 8, 9],
    "exclude_track_ids": None,
    "surface_id": 0,
    "pred_rank_max": 3,
    "ev_mid_min": None,
}
WIN_ROI_RULE = {
    **WIN_STABLE_RULE,
    "odds_min": 3.0,
    "odds_max": 10.0,
}
WIN_BROAD_RULE = {
    "pred_min": 0.12,
    "odds_min": 1.0,
    "odds_max": 4.0,
    "distance_min": 1400,
    "distance_max": None,
    "include_track_ids": [4, 5, 6, 9],
    "exclude_track_ids": None,
    "surface_id": 0,
    "pred_rank_max": None,
    "ev_mid_min": None,
}
PLACE_RULE = {
    "pred_min": 0.37,
    "odds_min": 3.2,
    "odds_max": 6.0,
    "distance_min": 1200,
    "distance_max": None,
    "include_track_ids": None,
    "exclude_track_ids": None,
    "surface_id": None,
    "pred_rank_max": 5,
    "ev_mid_min": 1.5,
}
PLACE_BROAD_RULE = {
    "pred_min": 0.37,
    "odds_min": 3.0,
    "odds_max": 6.0,
    "distance_min": 1200,
    "distance_max": None,
    "include_track_ids": None,
    "exclude_track_ids": None,
    "surface_id": None,
    "pred_rank_max": 5,
    "ev_mid_min": 1.5,
}
PLACE_STABLE_RULE = {
    "pred_min": 0.34,
    "odds_min": 3.2,
    "odds_max": 6.0,
    "distance_min": 1200,
    "distance_max": None,
    "include_track_ids": None,
    "exclude_track_ids": [3, 7, 10],
    "surface_id": None,
    "pred_rank_max": 5,
    "ev_mid_min": 1.4,
}
WIDE_ROI_RULE = {
    "score_topn": 5,
    "rank_max": 3,
    "pair_score_min": 0.10,
    "pair_min_pred_min": 0.25,
    "ev_mid_min": 0.0,
    "odds_min": 10.0,
    "odds_max": 100.0,
    "surface_label": "all",
    "surface_id": None,
    "distance_label": "all",
    "distance_min": None,
    "distance_max": None,
    "track_label": "exclude_1_2_3_7_10",
    "exclude_track_ids": (1, 2, 3, 7, 10),
}
WIDE_BROAD_RULE = {
    **WIDE_ROI_RULE,
    "pair_score_min": 0.07,
    "pair_min_pred_min": 0.22,
    "ev_mid_min": 1.0,
}
TRIO_ROI_RULE = {
    "score_topn": 10,
    "rank_max": 6,
    "trio_score_min": 0.0,
    "trio_min_pred_min": 0.0,
    "ev_mid_min": 0.0,
    "odds_min": 100.0,
    "odds_max": 1000.0,
    "surface_label": "turf",
    "surface_id": 0,
    "distance_label": "all",
    "distance_min": None,
    "distance_max": None,
    "track_label": "exclude_1_2_3_7_10",
    "exclude_track_ids": (1, 2, 3, 7, 10),
}
TRIO_BALANCED_RULE = {
    "score_topn": 10,
    "rank_max": 4,
    "trio_score_min": 0.0,
    "trio_min_pred_min": 0.0,
    "ev_mid_min": 0.0,
    "odds_min": 1.0,
    "odds_max": 200.0,
    "surface_label": "turf",
    "surface_id": 0,
    "distance_label": "1800_2600",
    "distance_min": 1800,
    "distance_max": 2600,
    "track_label": "exclude_1_2_3_7_10",
    "exclude_track_ids": (1, 2, 3, 7, 10),
}
TRIO_HIT_RATE_RULE = {
    "score_topn": 3,
    "rank_max": 4,
    "trio_score_min": 0.0,
    "trio_min_pred_min": 0.0,
    "ev_mid_min": 0.0,
    "odds_min": 1.0,
    "odds_max": 100.0,
    "surface_label": "turf",
    "surface_id": 0,
    "distance_label": "1800_2600",
    "distance_min": 1800,
    "distance_max": 2600,
    "track_label": "exclude_3_7_10",
    "exclude_track_ids": (3, 7, 10),
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

STRATEGY_DESCRIPTIONS = {
    "win_stable": "win pred>=0.15 odds[1.2,3.5) turf 1600m+ tracks 4,5,6,8,9 rank<=3",
    "win_roi": "win pred>=0.15 odds[3.0,10.0) turf 1600m+ tracks 4,5,6,8,9 rank<=3",
    "win_broad": "win pred>=0.12 odds[1.0,4.0) turf 1400m+ tracks 4,5,6,9",
    "place_latest": "place pred>=0.37 odds[3.2,6.0) 1200m+ rank<=5 ev>=1.5",
    "place_broad": "place pred>=0.37 odds[3.0,6.0) 1200m+ rank<=5 ev>=1.5",
    "place_stable": "place pred>=0.34 odds[3.2,6.0) 1200m+ exclude 3,7,10 rank<=5 ev>=1.4",
    "pedigree_mid_20_stable": "consensus place base>=0.34 sec>=0.36 odds[3.2,6.0) tracks 4,5,6,8,9",
    "wide_roi": "wide score top5 rank<=3 score>=0.10 min_pred>=0.25 odds[10,100) exclude 1,2,3,7,10",
    "wide_broad": "wide score top5 rank<=3 score>=0.07 min_pred>=0.22 ev>=1.0 odds[10,100) exclude 1,2,3,7,10",
    "trio_roi": "trio score top10 rank<=6 odds[100,1000) turf exclude 1,2,3,7,10",
    "trio_balanced": "trio score top10 rank<=4 odds[1,200) turf 1800-2600 exclude 1,2,3,7,10",
    "trio_hit_rate": "trio score top3 rank<=4 odds[1,100) turf 1800-2600 exclude 3,7,10",
    "trio_high_roi": "trio score top10 rank<=4 odds[20,200) turf 1800-2600 exclude 1,2,3,7,10",
}

PORTFOLIOS = [
    {
        "name": "best_mixed_width",
        "description": "best current mix: stable win, wider place, narrow wide/trio ROI",
        "strategies": ["win_stable", "place_broad", "wide_roi", "trio_roi"],
    },
    {
        "name": "balanced_all",
        "description": "all bet types, balanced trio hit-rate/ROI",
        "strategies": ["win_stable", "place_latest", "wide_roi", "trio_balanced"],
    },
    {
        "name": "roi_all",
        "description": "all bet types, higher ROI trio and win variants",
        "strategies": ["win_roi", "place_latest", "wide_roi", "trio_roi"],
    },
    {
        "name": "volume_plus_roi",
        "description": "buy more often by using both win variants and balanced trio",
        "strategies": ["win_stable", "win_roi", "place_latest", "wide_roi", "trio_balanced"],
    },
    {
        "name": "exotic_roi",
        "description": "wide and high-odds trio only",
        "strategies": ["wide_roi", "trio_roi"],
    },
    {
        "name": "exotic_balanced",
        "description": "wide plus balanced trio for roughly 30 percent coverage",
        "strategies": ["wide_roi", "trio_balanced"],
    },
    {
        "name": "place_wide",
        "description": "place broad plus narrow high-ROI wide",
        "strategies": ["place_broad", "wide_roi"],
    },
    {
        "name": "place_trio_roi",
        "description": "place broad plus high-ROI trio",
        "strategies": ["place_broad", "trio_roi"],
    },
    {
        "name": "pedigree_trio_roi",
        "description": "pedigree place consensus plus high-ROI trio (best overall ROI)",
        "strategies": ["pedigree_mid_20_stable", "trio_high_roi"],
    },
    {
        "name": "trio_double_roi",
        "description": "combination of two high-ROI trio rules",
        "strategies": ["trio_roi", "trio_high_roi"],
    },
    {
        "name": "base_low_variance",
        "description": "win stable plus place only",
        "strategies": ["win_stable", "place_latest"],
    },
]

COMBINATION_GROUPS = {
    "win": ["win_stable", "win_roi", "win_broad"],
    "place": ["place_latest", "place_broad", "place_stable", "pedigree_mid_20_stable"],
    "wide": ["wide_roi", "wide_broad"],
    "trio": ["trio_roi", "trio_balanced", "trio_hit_rate", "trio_high_roi"],
}


def _format_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}%"


def _load_rule_predictions(path: Path, engine: str) -> pd.DataFrame:
    predictions = pd.read_parquet(path, engine=engine)
    predictions = predictions.dropna(subset=["pred_top3", "target_top3", "place_odds_mid"]).copy()
    predictions["pred_rank"] = (
        predictions.groupby("race_id")["pred_top3"]
        .rank(method="first", ascending=False)
        .astype("int16")
    )
    predictions["ev_mid"] = predictions["pred_top3"] * predictions["place_odds_mid"]
    return predictions


def _apply_single_rule(predictions: pd.DataFrame, rule: dict[str, Any]) -> pd.DataFrame:
    mask = (
        (predictions["pred_top3"] >= rule["pred_min"])
        & (predictions["place_odds_mid"] >= rule["odds_min"])
        & (predictions["place_odds_mid"] < rule["odds_max"])
    )
    if rule["distance_min"] is not None:
        mask &= predictions["distance"] >= rule["distance_min"]
    if rule["distance_max"] is not None:
        mask &= predictions["distance"] < rule["distance_max"]
    if rule["include_track_ids"] is not None:
        mask &= predictions["track_id"].isin(rule["include_track_ids"])
    if rule["exclude_track_ids"] is not None:
        mask &= ~predictions["track_id"].isin(rule["exclude_track_ids"])
    if rule["surface_id"] is not None:
        mask &= predictions["surface_id"] == rule["surface_id"]
    if rule["pred_rank_max"] is not None:
        mask &= predictions["pred_rank"] <= rule["pred_rank_max"]
    if rule["ev_mid_min"] is not None:
        mask &= predictions["ev_mid"] >= rule["ev_mid_min"]
    return predictions.loc[mask].copy()


def _single_selections(predictions: pd.DataFrame, rule: dict[str, Any], bet_type: str, strategy: str, stake: float):
    selected = _apply_single_rule(predictions, rule)
    selected["bet_type"] = bet_type
    selected["strategy"] = strategy
    selected["stake"] = stake
    selected["hit"] = selected["target_top3"].astype("int8")
    selected["odds"] = selected["place_odds_mid"]
    selected["payout"] = selected["hit"] * selected["odds"] * stake
    selected["selection_key"] = selected["horse_number"].astype(str)
    return selected[
        ["race_id", "fold", "bet_type", "strategy", "selection_key", "stake", "hit", "odds", "payout"]
    ]


def _wide_selections(
    predictions_path: Path,
    db_path: Path,
    engine: str,
    stake: float,
    rule: dict[str, Any],
    strategy: str,
) -> pd.DataFrame:
    predictions = _load_wide_predictions(predictions_path, engine)
    wide_odds = _load_wide_odds(db_path)
    pairs = _build_wide_pairs(predictions, wide_odds)
    context = _build_wide_context(pairs)
    selected = pairs.loc[_mask_for_wide_rule(context, rule)].copy()
    selected["bet_type"] = "wide"
    selected["strategy"] = strategy
    selected["stake"] = stake
    selected["odds"] = selected["wide_odds_mid"]
    selected["payout"] = selected["hit"] * selected["odds"] * stake
    selected["selection_key"] = (
        selected["horse_number_1"].astype(str) + "-" + selected["horse_number_2"].astype(str)
    )
    return selected[
        ["race_id", "fold", "bet_type", "strategy", "selection_key", "stake", "hit", "odds", "payout"]
    ]


def _trio_selections(
    predictions_path: Path,
    db_path: Path,
    engine: str,
    stake: float,
    rule: dict[str, Any],
    strategy: str,
) -> pd.DataFrame:
    predictions = _load_trio_predictions(predictions_path, engine)
    trio_odds = _load_trio_odds(db_path, predictions["race_id"])
    combos = _build_trio_combinations(predictions, trio_odds)
    max_score_topn = rule["score_topn"]
    combos = combos[combos["score_rank"] <= max_score_topn].copy()
    context = _build_trio_context(combos)
    selected = combos.loc[_mask_for_trio_rule(context, rule)].copy()
    selected["bet_type"] = "trio"
    selected["strategy"] = strategy
    selected["stake"] = stake
    selected["payout"] = selected["hit"] * selected["odds"] * stake
    selected["selection_key"] = (
        selected["horse_number_1"].astype(str)
        + "-"
        + selected["horse_number_2"].astype(str)
        + "-"
        + selected["horse_number_3"].astype(str)
    )
    return selected[
        ["race_id", "fold", "bet_type", "strategy", "selection_key", "stake", "hit", "odds", "payout"]
    ]


def _summary(selections: pd.DataFrame, total_races: int, label: str) -> dict[str, Any]:
    if selections.empty:
        return {
            "name": label,
            "races": 0,
            "buy_rate_pct": 0.0,
            "selections": 0,
            "stake": 0.0,
            "hits": 0,
            "hit_rate_pct": None,
            "return_pct": None,
            "min_fold_return_pct": None,
        }
    stake_total = float(selections["stake"].sum())
    fold_returns = []
    for _, group in selections.groupby("fold", observed=True):
        fold_stake = float(group["stake"].sum())
        if fold_stake > 0:
            fold_returns.append(float(group["payout"].sum()) / fold_stake * 100)
    hits = int(selections["hit"].sum())
    return {
        "name": label,
        "races": int(selections["race_id"].nunique()),
        "buy_rate_pct": selections["race_id"].nunique() / total_races * 100 if total_races else None,
        "selections": len(selections),
        "stake": stake_total,
        "hits": hits,
        "hit_rate_pct": hits / len(selections) * 100 if len(selections) else None,
        "return_pct": float(selections["payout"].sum()) / stake_total * 100 if stake_total else None,
        "min_fold_return_pct": min(fold_returns) if fold_returns else None,
    }


def _portfolio_rows(selections_by_strategy: dict[str, pd.DataFrame], total_races_by_strategy: dict[str, int]):
    rows = []
    for portfolio in PORTFOLIOS:
        selected = pd.concat([selections_by_strategy[name] for name in portfolio["strategies"]], ignore_index=True)
        total_races = len(
            set().union(*(set(total_races_by_strategy[name]) for name in portfolio["strategies"]))
        )
        row = _summary(selected, total_races, portfolio["name"])
        row["description"] = portfolio["description"]
        row["strategies"] = ",".join(portfolio["strategies"])
        rows.append(row)
    return rows


def _combination_rows(selections_by_strategy: dict[str, pd.DataFrame], total_races_by_strategy: dict[str, int]):
    rows = []
    for win_name in COMBINATION_GROUPS["win"]:
        for place_name in COMBINATION_GROUPS["place"]:
            for wide_name in COMBINATION_GROUPS["wide"]:
                for trio_name in COMBINATION_GROUPS["trio"]:
                    strategies = [win_name, place_name, wide_name, trio_name]
                    selected = pd.concat([selections_by_strategy[name] for name in strategies], ignore_index=True)
                    total_races = len(set().union(*(set(total_races_by_strategy[name]) for name in strategies)))
                    row = _summary(selected, total_races, "+".join(strategies))
                    row["description"] = " / ".join(strategies)
                    row["strategies"] = ",".join(strategies)
                    rows.append(row)
    return sorted(
        rows,
        key=lambda row: (
            row["return_pct"] or 0,
            row["buy_rate_pct"] or 0,
            row["min_fold_return_pct"] or 0,
        ),
        reverse=True,
    )


def _print_rows(title: str, rows: list[dict[str, Any]]) -> None:
    print("")
    print(title)
    print("name                 buy_rate  races  selections  hits  hit_rate  ROI     min_fold  description")
    for row in rows:
        print(
            f"{row['name']:<20} "
            f"{_format_pct(row['buy_rate_pct']):>8}  "
            f"{row['races']:>5,}  {row['selections']:>10,}  {row['hits']:>4,}  "
            f"{_format_pct(row['hit_rate_pct']):>8}  {_format_pct(row['return_pct']):>7}  "
            f"{_format_pct(row['min_fold_return_pct']):>8}  "
            f"{row.get('description', '')}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate mixed bet portfolios across win/place/wide/trio rules.")
    parser.add_argument("--win-predictions", type=Path, default=DEFAULT_WIN_PREDICTIONS)
    parser.add_argument("--place-predictions", type=Path, default=DEFAULT_PLACE_PREDICTIONS)
    parser.add_argument("--exotic-predictions", type=Path, default=DEFAULT_EXOTIC_PREDICTIONS)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--engine", choices=["auto", "pyarrow", "fastparquet"], default="auto")
    parser.add_argument("--stake", type=float, default=100.0)
    args = parser.parse_args()

    win_predictions = _load_rule_predictions(args.win_predictions, args.engine)
    place_predictions = _load_rule_predictions(args.place_predictions, args.engine)
    exotic_predictions = _load_trio_predictions(args.exotic_predictions, args.engine)

    ped_pred_path = MODEL_DIR / "catboost_place_top3_predictions_pedigree.parquet"
    noped_pred_path = MODEL_DIR / "catboost_place_top3_predictions_no_pedigree.parquet"
    pedigree_consensus_df = _load_consensus_predictions(ped_pred_path, noped_pred_path, args.engine)

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
        "wide_broad": _wide_selections(
            args.exotic_predictions,
            args.db,
            args.engine,
            args.stake,
            WIDE_BROAD_RULE,
            "wide_broad",
        ),
        "trio_roi": _trio_selections(
            args.exotic_predictions, args.db, args.engine, args.stake, TRIO_ROI_RULE, "trio_roi"
        ),
        "trio_balanced": _trio_selections(
            args.exotic_predictions,
            args.db,
            args.engine,
            args.stake,
            TRIO_BALANCED_RULE,
            "trio_balanced",
        ),
        "trio_hit_rate": _trio_selections(
            args.exotic_predictions,
            args.db,
            args.engine,
            args.stake,
            TRIO_HIT_RATE_RULE,
            "trio_hit_rate",
        ),
        "trio_high_roi": _trio_selections(
            args.exotic_predictions,
            args.db,
            args.engine,
            args.stake,
            TRIO_HIGH_ROI_RULE,
            "trio_high_roi",
        ),
    }

    strategy_rows = []
    for name, selections in selections_by_strategy.items():
        row = _summary(selections, len(strategy_races[name]), name)
        row["description"] = STRATEGY_DESCRIPTIONS[name]
        row["strategies"] = name
        strategy_rows.append(row)
    portfolio_rows = _portfolio_rows(selections_by_strategy, strategy_races)
    combination_rows = _combination_rows(selections_by_strategy, strategy_races)

    output_rows = []
    for group, rows in [
        ("strategy", strategy_rows),
        ("portfolio", portfolio_rows),
        ("combination", combination_rows),
    ]:
        for row in rows:
            output_rows.append({"group": group, **row})
    output = pd.DataFrame(output_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"Output: {args.output}")
    print(f"Stake per selection: {args.stake:.0f}")
    print(f"Win races: {len(strategy_races['win_stable']):,}")
    print(f"Place races: {len(strategy_races['place_latest']):,}")
    print(f"Exotic races: {len(strategy_races['wide_roi']):,}")
    _print_rows("Strategy summaries", strategy_rows)
    _print_rows("Portfolio summaries", portfolio_rows)
    _print_rows("Top combination summaries", combination_rows[:20])


if __name__ == "__main__":
    main()
