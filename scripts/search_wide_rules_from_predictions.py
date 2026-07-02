import argparse
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.paths import DB_PATH, MODEL_DIR


DEFAULT_PREDICTIONS = MODEL_DIR / "catboost_place_top3_predictions.parquet"
DEFAULT_OUTPUT = MODEL_DIR / "wide_rule_search_results.csv"


@dataclass(frozen=True)
class TrackFilter:
    label: str
    exclude_ids: tuple[int, ...] | None


@dataclass(frozen=True)
class DistanceFilter:
    label: str
    min_value: int | None
    max_value: int | None


@dataclass(frozen=True)
class SurfaceFilter:
    label: str
    value: int | None


def _format_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}%"


def _load_predictions(path: Path, engine: str) -> pd.DataFrame:
    columns = [
        "race_id",
        "date",
        "horse_number",
        "target_top3",
        "track_id",
        "surface_id",
        "distance",
        "pred_top3",
        "fold",
    ]
    predictions = pd.read_parquet(path, columns=columns, engine=engine)
    predictions = predictions.dropna(subset=["pred_top3", "target_top3"])
    predictions["pred_rank"] = (
        predictions.groupby("race_id")["pred_top3"]
        .rank(method="first", ascending=False)
        .astype("int16")
    )
    return predictions


def _load_wide_odds(db_path: Path) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        wide_odds = pd.read_sql_query(
            """
            SELECT race_id, horse_number_1, horse_number_2, odds_min, odds_max
            FROM wide_odds
            """,
            conn,
        )
    wide_odds["wide_odds_mid"] = (wide_odds["odds_min"] + wide_odds["odds_max"]) / 2
    return wide_odds.dropna(subset=["wide_odds_mid"])


def _build_wide_pairs(predictions: pd.DataFrame, wide_odds: pd.DataFrame) -> pd.DataFrame:
    left = predictions.rename(
        columns={
            "horse_number": "horse_number_1",
            "target_top3": "target_top3_1",
            "pred_top3": "pred_top3_1",
            "pred_rank": "pred_rank_1",
        }
    )
    right = predictions.rename(
        columns={
            "horse_number": "horse_number_2",
            "target_top3": "target_top3_2",
            "pred_top3": "pred_top3_2",
            "pred_rank": "pred_rank_2",
        }
    )[["race_id", "horse_number_2", "target_top3_2", "pred_top3_2", "pred_rank_2"]]

    pairs = left.merge(right, on="race_id")
    pairs = pairs[pairs["horse_number_1"] < pairs["horse_number_2"]].copy()
    pairs = pairs.merge(wide_odds, on=["race_id", "horse_number_1", "horse_number_2"], how="inner")
    pairs["hit"] = ((pairs["target_top3_1"] == 1) & (pairs["target_top3_2"] == 1)).astype("int8")
    pairs["pair_score"] = pairs["pred_top3_1"] * pairs["pred_top3_2"]
    pairs["rank_max"] = pairs[["pred_rank_1", "pred_rank_2"]].max(axis=1)
    pairs["score_rank"] = (
        pairs.groupby("race_id")["pair_score"]
        .rank(method="first", ascending=False)
        .astype("int16")
    )
    return pairs


def _track_filters() -> list[TrackFilter]:
    return [
        TrackFilter("all", None),
        TrackFilter("exclude_3_7_10", (3, 7, 10)),
        TrackFilter("exclude_3_7_8_10", (3, 7, 8, 10)),
        TrackFilter("exclude_1_2_3_7_10", (1, 2, 3, 7, 10)),
    ]


def _distance_filters() -> list[DistanceFilter]:
    return [
        DistanceFilter("all", None, None),
        DistanceFilter("1200_1800", 1200, 1800),
        DistanceFilter("1600_2400", 1600, 2400),
        DistanceFilter("1800_2600", 1800, 2600),
        DistanceFilter("2000_up", 2000, None),
    ]


def _surface_filters() -> list[SurfaceFilter]:
    return [
        SurfaceFilter("all", None),
        SurfaceFilter("turf", 0),
        SurfaceFilter("dirt", 1),
    ]


def _candidate_rules() -> list[dict[str, Any]]:
    odds_ranges = [
        (1.0, 5.0),
        (1.0, 8.0),
        (1.0, 10.0),
        (1.0, 15.0),
        (1.0, 20.0),
        (1.0, 30.0),
        (2.0, 15.0),
        (2.0, 30.0),
        (3.0, 30.0),
        (5.0, 50.0),
        (10.0, 100.0),
    ]
    rules = []
    for score_topn in [1, 2, 3, 5, 8, 10, 15, 20, 30]:
        for rank_max in [2, 3, 4, 5, 6, 8, 10, 12, 16]:
            for odds_min, odds_max in odds_ranges:
                for surface in _surface_filters():
                    for distance in _distance_filters():
                        for track in _track_filters():
                            rules.append(
                                {
                                    "score_topn": score_topn,
                                    "rank_max": rank_max,
                                    "odds_min": odds_min,
                                    "odds_max": odds_max,
                                    "surface_label": surface.label,
                                    "surface_id": surface.value,
                                    "distance_label": distance.label,
                                    "distance_min": distance.min_value,
                                    "distance_max": distance.max_value,
                                    "track_label": track.label,
                                    "exclude_track_ids": track.exclude_ids,
                                }
                            )
    return rules


def _build_context(pairs: pd.DataFrame) -> dict[str, Any]:
    race_codes, _ = pd.factorize(pairs["race_id"], sort=False)
    return {
        "race": race_codes,
        "total_races": int(race_codes.max() + 1),
        "fold_values": np.sort(pairs["fold"].dropna().unique()),
        "fold": pairs["fold"].to_numpy(),
        "hit": pairs["hit"].to_numpy(),
        "wide_odds_mid": pairs["wide_odds_mid"].to_numpy(),
        "score_rank": pairs["score_rank"].to_numpy(),
        "rank_max": pairs["rank_max"].to_numpy(),
        "track_id": pairs["track_id"].to_numpy(),
        "surface_id": pairs["surface_id"].to_numpy(),
        "distance": pairs["distance"].to_numpy(),
    }


def _race_count(context: dict[str, Any], mask: np.ndarray) -> int:
    counts = np.bincount(context["race"][mask], minlength=context["total_races"])
    return int(np.count_nonzero(counts))


def _mask_for_rule(context: dict[str, Any], rule: dict[str, Any]) -> np.ndarray:
    mask = (
        (context["score_rank"] <= rule["score_topn"])
        & (context["rank_max"] <= rule["rank_max"])
        & (context["wide_odds_mid"] >= rule["odds_min"])
        & (context["wide_odds_mid"] < rule["odds_max"])
    )
    if rule["surface_id"] is not None:
        mask &= context["surface_id"] == rule["surface_id"]
    if rule["distance_min"] is not None:
        mask &= context["distance"] >= rule["distance_min"]
    if rule["distance_max"] is not None:
        mask &= context["distance"] < rule["distance_max"]
    if rule["exclude_track_ids"] is not None:
        mask &= ~np.isin(context["track_id"], rule["exclude_track_ids"])
    return mask


def _evaluate_rule(
    context: dict[str, Any],
    rule: dict[str, Any],
    min_buy_rate: float,
    max_buy_rate: float,
    min_selections: int,
    min_fold_selections: int,
    min_fold_return_mid: float | None,
    stake: float,
) -> dict[str, Any] | None:
    mask = _mask_for_rule(context, rule)
    selections = int(mask.sum())
    if selections < min_selections:
        return None

    races = _race_count(context, mask)
    buy_rate = races / context["total_races"] * 100
    if buy_rate < min_buy_rate or buy_rate > max_buy_rate:
        return None

    payout = float((context["hit"][mask] * context["wide_odds_mid"][mask] * stake).sum())
    return_pct = payout / (selections * stake) * 100
    hits = int(context["hit"][mask].sum())

    fold_returns = []
    fold_selections = []
    for fold in context["fold_values"]:
        fold_mask = mask & (context["fold"] == fold)
        fold_n = int(fold_mask.sum())
        if fold_n == 0:
            return None
        if fold_n < min_fold_selections:
            return None
        fold_payout = float((context["hit"][fold_mask] * context["wide_odds_mid"][fold_mask] * stake).sum())
        fold_returns.append(fold_payout / (fold_n * stake) * 100)
        fold_selections.append(fold_n)
    if min_fold_return_mid is not None and min(fold_returns) < min_fold_return_mid:
        return None

    result = dict(rule)
    result["races"] = races
    result["buy_rate_pct"] = buy_rate
    result["selections"] = selections
    result["hits"] = hits
    result["hit_rate_pct"] = hits / selections * 100
    result["return_mid_pct"] = return_pct
    result["min_fold_return_mid_pct"] = min(fold_returns)
    result["max_fold_return_mid_pct"] = max(fold_returns)
    result["min_fold_selections"] = min(fold_selections)
    result["exclude_track_ids"] = (
        "" if result["exclude_track_ids"] is None else ",".join(str(v) for v in result["exclude_track_ids"])
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Search wide betting rules from place-top3 predictions.")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--engine", choices=["auto", "pyarrow", "fastparquet"], default="auto")
    parser.add_argument("--min-buy-rate", type=float, default=18.0)
    parser.add_argument("--max-buy-rate", type=float, default=22.0)
    parser.add_argument("--min-selections", type=int, default=50)
    parser.add_argument("--min-fold-selections", type=int, default=10)
    parser.add_argument("--min-fold-return-mid", type=float, default=None)
    parser.add_argument("--stake", type=float, default=100.0)
    args = parser.parse_args()

    predictions = _load_predictions(args.predictions, args.engine)
    wide_odds = _load_wide_odds(args.db)
    pairs = _build_wide_pairs(predictions, wide_odds)
    context = _build_context(pairs)

    results = []
    for rule in _candidate_rules():
        result = _evaluate_rule(
            context,
            rule,
            min_buy_rate=args.min_buy_rate,
            max_buy_rate=args.max_buy_rate,
            min_selections=args.min_selections,
            min_fold_selections=args.min_fold_selections,
            min_fold_return_mid=args.min_fold_return_mid,
            stake=args.stake,
        )
        if result is not None:
            results.append(result)

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values(
            ["return_mid_pct", "min_fold_return_mid_pct", "buy_rate_pct"],
            ascending=[False, False, True],
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"Predictions: {args.predictions}")
    print(f"DB: {args.db}")
    print(f"Pair rows: {len(pairs):,}")
    print(f"Prediction races: {context['total_races']:,}")
    print(f"Results: {len(results_df):,}")
    print(f"Output: {args.output}")
    if results_df.empty:
        return
    print("")
    print("Top rules")
    print(
        "return_mid  min_fold  buy_rate  races  selections  hits  hit_rate  "
        "score_topn  rank_max  odds_range  surface  distance  track"
    )
    for row in results_df.head(20).itertuples(index=False):
        print(
            f"{_format_pct(row.return_mid_pct):>10}  "
            f"{_format_pct(row.min_fold_return_mid_pct):>8}  "
            f"{_format_pct(row.buy_rate_pct):>8}  "
            f"{row.races:>5}  {row.selections:>10}  {row.hits:>4}  "
            f"{_format_pct(row.hit_rate_pct):>8}  "
            f"{row.score_topn:>10}  {row.rank_max:>8}  "
            f"[{row.odds_min:.1f},{row.odds_max:.1f})  "
            f"{row.surface_label:<7}  {row.distance_label:<9}  {row.track_label}"
        )


if __name__ == "__main__":
    main()
