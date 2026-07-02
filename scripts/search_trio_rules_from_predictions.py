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
DEFAULT_OUTPUT = MODEL_DIR / "trio_rule_search_results.csv"


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


def _load_trio_odds(db_path: Path, race_ids: pd.Series) -> pd.DataFrame:
    race_id_values = [str(value) for value in race_ids.dropna().unique()]
    if not race_id_values:
        return pd.DataFrame(columns=["race_id", "horse_number_1", "horse_number_2", "horse_number_3", "odds"])

    chunks = []
    with sqlite3.connect(db_path) as conn:
        for start in range(0, len(race_id_values), 900):
            chunk = race_id_values[start : start + 900]
            placeholders = ",".join("?" for _ in chunk)
            chunks.append(
                pd.read_sql_query(
                    f"""
                    SELECT race_id, horse_number_1, horse_number_2, horse_number_3, odds
                    FROM trio_odds
                    WHERE race_id IN ({placeholders})
                    """,
                    conn,
                    params=chunk,
                )
            )
    trio_odds = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    return trio_odds.dropna(subset=["odds"])


def _build_trio_combinations(predictions: pd.DataFrame, trio_odds: pd.DataFrame) -> pd.DataFrame:
    base_columns = [
        "race_id",
        "horse_number",
        "target_top3",
        "pred_top3",
        "pred_rank",
        "date",
        "track_id",
        "surface_id",
        "distance",
        "fold",
    ]
    base = predictions[base_columns]
    combos = trio_odds.merge(
        base.rename(
            columns={
                "horse_number": "horse_number_1",
                "target_top3": "target_top3_1",
                "pred_top3": "pred_top3_1",
                "pred_rank": "pred_rank_1",
            }
        ),
        on=["race_id", "horse_number_1"],
        how="inner",
    )
    combos = combos.merge(
        base[["race_id", "horse_number", "target_top3", "pred_top3", "pred_rank"]].rename(
            columns={
                "horse_number": "horse_number_2",
                "target_top3": "target_top3_2",
                "pred_top3": "pred_top3_2",
                "pred_rank": "pred_rank_2",
            }
        ),
        on=["race_id", "horse_number_2"],
        how="inner",
    )
    combos = combos.merge(
        base[["race_id", "horse_number", "target_top3", "pred_top3", "pred_rank"]].rename(
            columns={
                "horse_number": "horse_number_3",
                "target_top3": "target_top3_3",
                "pred_top3": "pred_top3_3",
                "pred_rank": "pred_rank_3",
            }
        ),
        on=["race_id", "horse_number_3"],
        how="inner",
    )
    combos["hit"] = (
        (combos["target_top3_1"] == 1)
        & (combos["target_top3_2"] == 1)
        & (combos["target_top3_3"] == 1)
    ).astype("int8")
    combos["trio_score"] = combos["pred_top3_1"] * combos["pred_top3_2"] * combos["pred_top3_3"]
    combos["trio_min_pred"] = combos[["pred_top3_1", "pred_top3_2", "pred_top3_3"]].min(axis=1)
    combos["rank_max"] = combos[["pred_rank_1", "pred_rank_2", "pred_rank_3"]].max(axis=1)
    combos["ev_mid"] = combos["trio_score"] * combos["odds"]
    combos["score_rank"] = (
        combos.groupby("race_id")["trio_score"]
        .rank(method="first", ascending=False)
        .astype("int16")
    )
    return combos


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
        (1.0, 50.0),
        (1.0, 100.0),
        (1.0, 200.0),
        (5.0, 100.0),
        (5.0, 200.0),
        (10.0, 100.0),
        (10.0, 200.0),
        (20.0, 200.0),
        (20.0, 300.0),
        (30.0, 300.0),
        (50.0, 500.0),
        (100.0, 1000.0),
    ]
    rules = []
    for score_topn in [1, 2, 3, 5, 8, 10, 15, 20, 30]:
        for rank_max in [3, 4, 5, 6, 8]:
            for odds_min, odds_max in odds_ranges:
                for surface in _surface_filters():
                    for distance in _distance_filters():
                        for track in _track_filters():
                            rules.append(
                                {
                                    "score_topn": score_topn,
                                    "rank_max": rank_max,
                                    "trio_score_min": 0.0,
                                    "trio_min_pred_min": 0.0,
                                    "ev_mid_min": 0.0,
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
    for score_topn in [5, 8, 10, 15]:
        for rank_max in [4, 5, 6]:
            for trio_score_min in [0.010, 0.015, 0.020, 0.025]:
                for trio_min_pred_min in [0.18, 0.20, 0.22, 0.25]:
                    for ev_mid_min in [0.0, 1.0, 1.5, 2.0]:
                        rules.append(
                            {
                                "score_topn": score_topn,
                                "rank_max": rank_max,
                                "trio_score_min": trio_score_min,
                                "trio_min_pred_min": trio_min_pred_min,
                                "ev_mid_min": ev_mid_min,
                                "odds_min": 20.0,
                                "odds_max": 300.0,
                                "surface_label": "all",
                                "surface_id": None,
                                "distance_label": "all",
                                "distance_min": None,
                                "distance_max": None,
                                "track_label": "exclude_1_2_3_7_10",
                                "exclude_track_ids": (1, 2, 3, 7, 10),
                            }
                        )
    return rules


def _build_context(combos: pd.DataFrame) -> dict[str, Any]:
    race_codes, _ = pd.factorize(combos["race_id"], sort=False)
    return {
        "race": race_codes,
        "total_races": int(race_codes.max() + 1),
        "fold_values": np.sort(combos["fold"].dropna().unique()),
        "fold": combos["fold"].to_numpy(),
        "hit": combos["hit"].to_numpy(),
        "trio_odds": combos["odds"].to_numpy(),
        "trio_score": combos["trio_score"].to_numpy(),
        "trio_min_pred": combos["trio_min_pred"].to_numpy(),
        "ev_mid": combos["ev_mid"].to_numpy(),
        "score_rank": combos["score_rank"].to_numpy(),
        "rank_max": combos["rank_max"].to_numpy(),
        "track_id": combos["track_id"].to_numpy(),
        "surface_id": combos["surface_id"].to_numpy(),
        "distance": combos["distance"].to_numpy(),
    }


def _race_count(context: dict[str, Any], mask: np.ndarray) -> int:
    counts = np.bincount(context["race"][mask], minlength=context["total_races"])
    return int(np.count_nonzero(counts))


def _mask_for_rule(context: dict[str, Any], rule: dict[str, Any]) -> np.ndarray:
    mask = (
        (context["score_rank"] <= rule["score_topn"])
        & (context["rank_max"] <= rule["rank_max"])
        & (context["trio_score"] >= rule["trio_score_min"])
        & (context["trio_min_pred"] >= rule["trio_min_pred_min"])
        & (context["ev_mid"] >= rule["ev_mid_min"])
        & (context["trio_odds"] >= rule["odds_min"])
        & (context["trio_odds"] < rule["odds_max"])
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

    payout = float((context["hit"][mask] * context["trio_odds"][mask] * stake).sum())
    return_pct = payout / (selections * stake) * 100
    hits = int(context["hit"][mask].sum())

    fold_returns = []
    fold_selections = []
    for fold in context["fold_values"]:
        fold_mask = mask & (context["fold"] == fold)
        fold_n = int(fold_mask.sum())
        if fold_n == 0 or fold_n < min_fold_selections:
            return None
        fold_payout = float((context["hit"][fold_mask] * context["trio_odds"][fold_mask] * stake).sum())
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
    return result


def _selection_columns(combos: pd.DataFrame, mask: np.ndarray, stake: float) -> pd.DataFrame:
    selections = combos.loc[mask].copy()
    selections["trio_payout"] = selections["hit"] * selections["odds"] * stake
    columns = [
        "race_id",
        "date",
        "track_id",
        "surface_id",
        "distance",
        "horse_number_1",
        "horse_number_2",
        "horse_number_3",
        "pred_top3_1",
        "pred_top3_2",
        "pred_top3_3",
        "pred_rank_1",
        "pred_rank_2",
        "pred_rank_3",
        "trio_score",
        "trio_min_pred",
        "ev_mid",
        "score_rank",
        "odds",
        "hit",
        "trio_payout",
        "fold",
    ]
    return selections[[col for col in columns if col in selections.columns]].sort_values(
        ["race_id", "score_rank", "horse_number_1", "horse_number_2", "horse_number_3"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Search trio betting rules from place-top3 predictions.")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--selections-output", type=Path, default=None)
    parser.add_argument("--engine", choices=["auto", "pyarrow", "fastparquet"], default="auto")
    parser.add_argument("--min-buy-rate", type=float, default=18.0)
    parser.add_argument("--max-buy-rate", type=float, default=22.0)
    parser.add_argument("--min-selections", type=int, default=50)
    parser.add_argument("--min-fold-selections", type=int, default=10)
    parser.add_argument("--min-fold-return-mid", type=float, default=None)
    parser.add_argument("--stake", type=float, default=100.0)
    args = parser.parse_args()

    predictions = _load_predictions(args.predictions, args.engine)
    rules = _candidate_rules()
    trio_odds = _load_trio_odds(args.db, predictions["race_id"])
    combos = _build_trio_combinations(predictions, trio_odds)
    max_score_topn = max(rule["score_topn"] for rule in rules)
    combos = combos[combos["score_rank"] <= max_score_topn].copy()
    context = _build_context(combos)

    results = []
    for rule in rules:
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
        results_df["exclude_track_ids"] = results_df["exclude_track_ids"].map(
            lambda value: "" if value is None else ",".join(str(v) for v in value)
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.output, index=False, encoding="utf-8-sig")

    if args.selections_output is not None and results:
        best_rule = max(results, key=lambda row: (row["return_mid_pct"], row["min_fold_return_mid_pct"]))
        best_mask = _mask_for_rule(context, best_rule)
        selections = _selection_columns(combos, best_mask, args.stake)
        args.selections_output.parent.mkdir(parents=True, exist_ok=True)
        selections.to_csv(args.selections_output, index=False, encoding="utf-8-sig")

    print(f"Predictions: {args.predictions}")
    print(f"DB: {args.db}")
    print(f"Trio rows: {len(combos):,}")
    print(f"Prediction races: {context['total_races']:,}")
    print(f"Results: {len(results_df):,}")
    print(f"Output: {args.output}")
    if args.selections_output is not None:
        print(f"Selections output: {args.selections_output}")
    if results_df.empty:
        return
    print("")
    print("Top rules")
    print(
        "return_mid  min_fold  buy_rate  races  selections  hits  hit_rate  "
        "score_topn  rank_max  score_min  min_pred  ev_min  odds_range  surface  distance  track"
    )
    for row in results_df.head(20).itertuples(index=False):
        print(
            f"{_format_pct(row.return_mid_pct):>10}  "
            f"{_format_pct(row.min_fold_return_mid_pct):>8}  "
            f"{_format_pct(row.buy_rate_pct):>8}  "
            f"{row.races:>5}  {row.selections:>10}  {row.hits:>4}  "
            f"{_format_pct(row.hit_rate_pct):>8}  "
            f"{row.score_topn:>10}  {row.rank_max:>8}  "
            f"{row.trio_score_min:>9.3f}  {row.trio_min_pred_min:>8.2f}  {row.ev_mid_min:>6.1f}  "
            f"[{row.odds_min:.1f},{row.odds_max:.1f})  "
            f"{row.surface_label:<7}  {row.distance_label:<9}  {row.track_label}"
        )


if __name__ == "__main__":
    main()
