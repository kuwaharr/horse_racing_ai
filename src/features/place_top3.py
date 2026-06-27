import sqlite3
from collections import defaultdict
from datetime import date
from pathlib import Path


DEFAULT_DATASET_NAMES = {
    "early": "place_top3_early_dataset.parquet",
    "late": "place_top3_late_dataset.parquet",
}
DEFAULT_HISTORY_DATASET_NAMES = {
    "early": "place_top3_early_history_dataset.parquet",
    "late": "place_top3_late_history_dataset.parquet",
}


LATE_FEATURE_COLUMNS = [
    "race_id",
    "date",
    "track_id",
    "race_number",
    "post_time_min",
    "surface_id",
    "distance",
    "course_direction_id",
    "course_layout_id",
    "course_variant_id",
    "weather_id",
    "track_condition_id",
    "race_size",
    "gate",
    "horse_number",
    "horse_id",
    "horse_name",
    "sex_id",
    "age",
    "jockey_id",
    "weight",
    "popularity",
    "stable_id",
    "trainer_id",
    "horse_weight",
    "horse_weight_diff",
    "place_odds_min",
    "place_odds_max",
    "target_top3",
]


EARLY_FEATURE_COLUMNS = [
    "race_id",
    "date",
    "track_id",
    "race_number",
    "post_time_min",
    "surface_id",
    "distance",
    "course_direction_id",
    "course_layout_id",
    "course_variant_id",
    "weather_id",
    "track_condition_id",
    "race_size",
    "gate",
    "horse_number",
    "horse_id",
    "horse_name",
    "sex_id",
    "age",
    "jockey_id",
    "weight",
    "stable_id",
    "trainer_id",
    "target_top3",
]


FEATURE_COLUMNS_BY_MODE = {
    "early": EARLY_FEATURE_COLUMNS,
    "late": LATE_FEATURE_COLUMNS,
}

HISTORY_FEATURE_COLUMNS = [
    "horse_past_starts",
    "horse_past_top3",
    "horse_past_top3_rate",
    "horse_past_avg_finish",
    "horse_past_avg_distance",
    "horse_distance_diff_avg",
    "horse_past_min_distance",
    "horse_past_max_distance",
    "horse_distance_above_max",
    "horse_distance_below_min",
    "horse_past_avg_weight",
    "horse_weight_diff_avg",
    "horse_recent3_top3_rate",
    "horse_recent3_avg_finish",
    "horse_recent5_top3_rate",
    "horse_recent5_avg_finish",
    "horse_days_since_last",
    "horse_prev_distance",
    "horse_distance_diff_prev",
    "horse_track_past_starts",
    "horse_track_past_top3",
    "horse_track_past_top3_rate",
    "horse_surface_past_starts",
    "horse_surface_past_top3",
    "horse_surface_past_top3_rate",
    "horse_distance_band_past_starts",
    "horse_distance_band_past_top3",
    "horse_distance_band_past_top3_rate",
    "jockey_past_starts",
    "jockey_past_top3",
    "jockey_past_top3_rate",
    "jockey_past_avg_finish",
    "jockey_recent20_top3_rate",
    "jockey_recent20_avg_finish",
    "jockey_track_past_starts",
    "jockey_track_past_top3",
    "jockey_track_past_top3_rate",
    "jockey_surface_past_starts",
    "jockey_surface_past_top3",
    "jockey_surface_past_top3_rate",
    "trainer_past_starts",
    "trainer_past_top3",
    "trainer_past_top3_rate",
    "trainer_past_avg_finish",
    "trainer_recent20_top3_rate",
    "trainer_recent20_avg_finish",
    "trainer_track_past_starts",
    "trainer_track_past_top3",
    "trainer_track_past_top3_rate",
    "trainer_surface_past_starts",
    "trainer_surface_past_top3",
    "trainer_surface_past_top3_rate",
    "race_horse_past_top3_rate_rank",
    "race_horse_past_top3_rate_diff",
    "race_horse_past_avg_finish_rank",
    "race_horse_past_avg_finish_diff",
    "race_horse_distance_diff_avg_rank",
    "race_horse_distance_diff_avg_diff",
    "race_horse_distance_above_max_rank",
    "race_horse_distance_above_max_diff",
    "race_horse_distance_below_min_rank",
    "race_horse_distance_below_min_diff",
    "race_horse_weight_diff_avg_rank",
    "race_horse_weight_diff_avg_diff",
    "race_horse_recent3_top3_rate_rank",
    "race_horse_recent3_top3_rate_diff",
    "race_horse_recent3_avg_finish_rank",
    "race_horse_recent3_avg_finish_diff",
    "race_horse_days_since_last_rank",
    "race_horse_days_since_last_diff",
    "race_horse_distance_diff_prev_rank",
    "race_horse_distance_diff_prev_diff",
    "race_horse_track_past_top3_rate_rank",
    "race_horse_track_past_top3_rate_diff",
    "race_horse_surface_past_top3_rate_rank",
    "race_horse_surface_past_top3_rate_diff",
    "race_horse_distance_band_past_top3_rate_rank",
    "race_horse_distance_band_past_top3_rate_diff",
    "race_jockey_past_top3_rate_rank",
    "race_jockey_past_top3_rate_diff",
    "race_jockey_past_avg_finish_rank",
    "race_jockey_past_avg_finish_diff",
    "race_jockey_recent20_top3_rate_rank",
    "race_jockey_recent20_top3_rate_diff",
    "race_jockey_recent20_avg_finish_rank",
    "race_jockey_recent20_avg_finish_diff",
    "race_jockey_track_past_top3_rate_rank",
    "race_jockey_track_past_top3_rate_diff",
    "race_jockey_surface_past_top3_rate_rank",
    "race_jockey_surface_past_top3_rate_diff",
    "race_trainer_past_top3_rate_rank",
    "race_trainer_past_top3_rate_diff",
    "race_trainer_past_avg_finish_rank",
    "race_trainer_past_avg_finish_diff",
    "race_trainer_recent20_top3_rate_rank",
    "race_trainer_recent20_top3_rate_diff",
    "race_trainer_recent20_avg_finish_rank",
    "race_trainer_recent20_avg_finish_diff",
    "race_trainer_track_past_top3_rate_rank",
    "race_trainer_track_past_top3_rate_diff",
    "race_trainer_surface_past_top3_rate_rank",
    "race_trainer_surface_past_top3_rate_diff",
    "race_gate_rank",
    "race_gate_diff",
    "race_horse_number_rank",
    "race_horse_number_diff",
    "race_weight_rank",
    "race_weight_diff",
    "race_age_rank",
    "race_age_diff",
]


PLACE_TOP3_BASE_QUERY = """
SELECT
    ra.race_id,
    ra.date,
    ra.track_id,
    ra.race_number,
    ra.post_time_min,
    ra.surface_id,
    ra.distance,
    ra.course_direction_id,
    ra.course_layout_id,
    ra.course_variant_id,
    ra.weather_id,
    ra.track_condition_id,
    ra.race_size,
    ru.gate,
    ru.horse_number,
    ru.horse_id,
    ru.horse_name,
    ru.sex_id,
    ru.age,
    ru.jockey_id,
    ru.weight,
    ru.popularity,
    ru.stable_id,
    ru.trainer_id,
    ru.horse_weight,
    ru.horse_weight_diff,
    ru.finish AS finish_position,
    po.odds_min AS place_odds_min,
    po.odds_max AS place_odds_max,
    CASE WHEN ru.finish <= 3 THEN 1 ELSE 0 END AS target_top3
FROM runner ru
INNER JOIN race ra
    ON ra.race_id = ru.race_id
LEFT JOIN place_odds po
    ON po.race_id = ru.race_id
    AND po.horse_number = ru.horse_number
WHERE ru.status_id = 0
  AND ru.finish IS NOT NULL
ORDER BY ra.date, ra.race_id, ru.horse_number
"""


def default_dataset_name(mode: str, history_features: bool = False) -> str:
    names = DEFAULT_HISTORY_DATASET_NAMES if history_features else DEFAULT_DATASET_NAMES
    if mode not in names:
        raise ValueError(f"Unknown dataset mode: {mode}")
    return names[mode]


def _append_history_features(df):
    entity_specs = [
        ("horse", "horse_id"),
        ("jockey", "jockey_id"),
        ("trainer", "trainer_id"),
    ]
    stats = {
        name: defaultdict(_empty_entity_stat)
        for name, _ in entity_specs
    }
    horse_track_stats = defaultdict(_empty_history_stat)
    horse_surface_stats = defaultdict(_empty_history_stat)
    horse_distance_band_stats = defaultdict(_empty_history_stat)
    jockey_track_stats = defaultdict(_empty_history_stat)
    jockey_surface_stats = defaultdict(_empty_history_stat)
    trainer_track_stats = defaultdict(_empty_history_stat)
    trainer_surface_stats = defaultdict(_empty_history_stat)
    recent_history = {
        name: defaultdict(list)
        for name, _ in entity_specs
    }
    horse_last_seen = {}

    df = df.sort_values(["date", "race_id", "horse_number"]).copy()
    df["distance_band"] = df["distance"].map(_distance_band)
    df["date_obj"] = df["date"].map(date.fromisoformat)
    history_rows = []

    for _, day_df in df.groupby("date", sort=True):
        for idx, row in day_df.iterrows():
            history_row = {"__idx": idx}
            for name, id_col in entity_specs:
                value = row[id_col]
                item = stats[name][value]
                starts = item["starts"]
                top3 = item["top3"]
                finish_sum = item["finish_sum"]
                history_row[f"{name}_past_starts"] = starts
                history_row[f"{name}_past_top3"] = top3
                history_row[f"{name}_past_top3_rate"] = None if starts == 0 else top3 / starts
                history_row[f"{name}_past_avg_finish"] = None if starts == 0 else finish_sum / starts
                recent = recent_history[name][value]
                if name == "horse":
                    avg_distance = None if starts == 0 else item["distance_sum"] / starts
                    avg_weight = None if starts == 0 else item["weight_sum"] / starts
                    min_distance = item["min_distance"]
                    max_distance = item["max_distance"]
                    history_row["horse_past_avg_distance"] = avg_distance
                    history_row["horse_distance_diff_avg"] = None if avg_distance is None else row["distance"] - avg_distance
                    history_row["horse_past_min_distance"] = min_distance
                    history_row["horse_past_max_distance"] = max_distance
                    history_row["horse_distance_above_max"] = (
                        None if max_distance is None else max(0, row["distance"] - max_distance)
                    )
                    history_row["horse_distance_below_min"] = (
                        None if min_distance is None else max(0, min_distance - row["distance"])
                    )
                    history_row["horse_past_avg_weight"] = avg_weight
                    history_row["horse_weight_diff_avg"] = None if avg_weight is None else row["weight"] - avg_weight
                    history_row["horse_recent3_top3_rate"] = _recent_top3_rate(recent, 3)
                    history_row["horse_recent3_avg_finish"] = _recent_avg_finish(recent, 3)
                    history_row["horse_recent5_top3_rate"] = _recent_top3_rate(recent, 5)
                    history_row["horse_recent5_avg_finish"] = _recent_avg_finish(recent, 5)
                    last_seen = horse_last_seen.get(value)
                    if last_seen is None:
                        history_row["horse_days_since_last"] = None
                        history_row["horse_prev_distance"] = None
                        history_row["horse_distance_diff_prev"] = None
                    else:
                        last_date, last_distance = last_seen
                        history_row["horse_days_since_last"] = (row["date_obj"] - last_date).days
                        history_row["horse_prev_distance"] = last_distance
                        history_row["horse_distance_diff_prev"] = row["distance"] - last_distance
                elif name == "jockey":
                    history_row["jockey_recent20_top3_rate"] = _recent_top3_rate(recent, 20)
                    history_row["jockey_recent20_avg_finish"] = _recent_avg_finish(recent, 20)
                elif name == "trainer":
                    history_row["trainer_recent20_top3_rate"] = _recent_top3_rate(recent, 20)
                    history_row["trainer_recent20_avg_finish"] = _recent_avg_finish(recent, 20)
            for feature_name, item in [
                ("horse_track", horse_track_stats[(row["horse_id"], row["track_id"])]),
                ("horse_surface", horse_surface_stats[(row["horse_id"], row["surface_id"])]),
                (
                    "horse_distance_band",
                    horse_distance_band_stats[(row["horse_id"], row["distance_band"])],
                ),
                ("jockey_track", jockey_track_stats[(row["jockey_id"], row["track_id"])]),
                ("jockey_surface", jockey_surface_stats[(row["jockey_id"], row["surface_id"])]),
                ("trainer_track", trainer_track_stats[(row["trainer_id"], row["track_id"])]),
                ("trainer_surface", trainer_surface_stats[(row["trainer_id"], row["surface_id"])]),
            ]:
                starts = item["starts"]
                top3 = item["top3"]
                history_row[f"{feature_name}_past_starts"] = starts
                history_row[f"{feature_name}_past_top3"] = top3
                history_row[f"{feature_name}_past_top3_rate"] = None if starts == 0 else top3 / starts
            history_rows.append(history_row)

        for _, row in day_df.iterrows():
            target = int(row["target_top3"])
            finish = float(row["finish_position"])
            for name, id_col in entity_specs:
                item = stats[name][row[id_col]]
                item["starts"] += 1
                item["top3"] += target
                item["finish_sum"] += finish
                if name == "horse":
                    item["distance_sum"] += float(row["distance"])
                    item["weight_sum"] += float(row["weight"])
                    item["min_distance"] = (
                        int(row["distance"])
                        if item["min_distance"] is None
                        else min(item["min_distance"], int(row["distance"]))
                    )
                    item["max_distance"] = (
                        int(row["distance"])
                        if item["max_distance"] is None
                        else max(item["max_distance"], int(row["distance"]))
                    )
                recent_history[name][row[id_col]].append({"top3": target, "finish": finish})
            horse_last_seen[row["horse_id"]] = (row["date_obj"], int(row["distance"]))
            for item in [
                horse_track_stats[(row["horse_id"], row["track_id"])],
                horse_surface_stats[(row["horse_id"], row["surface_id"])],
                horse_distance_band_stats[(row["horse_id"], row["distance_band"])],
                jockey_track_stats[(row["jockey_id"], row["track_id"])],
                jockey_surface_stats[(row["jockey_id"], row["surface_id"])],
                trainer_track_stats[(row["trainer_id"], row["track_id"])],
                trainer_surface_stats[(row["trainer_id"], row["surface_id"])],
            ]:
                item["starts"] += 1
                item["top3"] += target

    try:
        import pandas as pd
    except ModuleNotFoundError as e:
        raise RuntimeError("履歴特徴量の作成には pandas が必要です。") from e

    history_df = pd.DataFrame(history_rows).set_index("__idx")
    df = df.join(history_df).sort_values(["date", "race_id", "horse_number"])
    df = _append_race_relative_features(df)
    return df.drop(columns=["distance_band", "date_obj"])


def _empty_history_stat() -> dict:
    return {"starts": 0, "top3": 0}


def _empty_entity_stat() -> dict:
    return {
        "starts": 0,
        "top3": 0,
        "finish_sum": 0.0,
        "distance_sum": 0.0,
        "weight_sum": 0.0,
        "min_distance": None,
        "max_distance": None,
    }


def _recent_top3_rate(rows: list[dict], n: int) -> float | None:
    recent = rows[-n:]
    if not recent:
        return None
    return sum(row["top3"] for row in recent) / len(recent)


def _recent_avg_finish(rows: list[dict], n: int) -> float | None:
    recent = rows[-n:]
    if not recent:
        return None
    return sum(row["finish"] for row in recent) / len(recent)


def _distance_band(distance: int) -> str:
    if distance < 1400:
        return "under1400"
    if distance < 1800:
        return "1400_1799"
    if distance < 2200:
        return "1800_2199"
    return "2200plus"


def _append_race_relative_features(df):
    relative_specs = [
        ("horse_past_top3_rate", False),
        ("horse_past_avg_finish", True),
        ("horse_distance_diff_avg", True),
        ("horse_distance_above_max", True),
        ("horse_distance_below_min", True),
        ("horse_weight_diff_avg", True),
        ("horse_recent3_top3_rate", False),
        ("horse_recent3_avg_finish", True),
        ("horse_days_since_last", True),
        ("horse_distance_diff_prev", True),
        ("horse_track_past_top3_rate", False),
        ("horse_surface_past_top3_rate", False),
        ("horse_distance_band_past_top3_rate", False),
        ("jockey_past_top3_rate", False),
        ("jockey_past_avg_finish", True),
        ("jockey_recent20_top3_rate", False),
        ("jockey_recent20_avg_finish", True),
        ("jockey_track_past_top3_rate", False),
        ("jockey_surface_past_top3_rate", False),
        ("trainer_past_top3_rate", False),
        ("trainer_past_avg_finish", True),
        ("trainer_recent20_top3_rate", False),
        ("trainer_recent20_avg_finish", True),
        ("trainer_track_past_top3_rate", False),
        ("trainer_surface_past_top3_rate", False),
        ("gate", True),
        ("horse_number", True),
        ("weight", False),
        ("age", True),
    ]
    for col, ascending in relative_specs:
        rank_col = f"race_{col}_rank"
        diff_col = f"race_{col}_diff"
        filled = df[col].fillna(-1)
        df[rank_col] = filled.groupby(df["race_id"]).rank(method="average", ascending=ascending)
        df[diff_col] = df[col] - df.groupby("race_id")[col].transform("mean")
    return df


def build_place_top3_dataset(
    db_path: Path,
    output_path: Path,
    mode: str = "late",
    engine: str = "auto",
    history_features: bool = False,
) -> int:
    if mode not in FEATURE_COLUMNS_BY_MODE:
        raise ValueError(f"Unknown dataset mode: {mode}")

    try:
        import pandas as pd
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "Parquet出力には pandas と pyarrow または fastparquet が必要です。"
            " PowerShellで `pip install pandas pyarrow` を実行してください。"
        ) from e

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(PLACE_TOP3_BASE_QUERY, conn)

    if history_features:
        df = _append_history_features(df)

    columns = FEATURE_COLUMNS_BY_MODE[mode]
    if history_features:
        columns = columns[:-1] + HISTORY_FEATURE_COLUMNS + columns[-1:]
    df = df[columns]

    try:
        df.to_parquet(output_path, index=False, engine=engine)
    except ImportError as e:
        raise RuntimeError(
            "Parquet出力エンジンが見つかりません。"
            " PowerShellで `pip install pyarrow` を実行するか、"
            " `--engine fastparquet` を指定してください。"
        ) from e

    return len(df)
