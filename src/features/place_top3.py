import sqlite3
from collections import defaultdict
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
    "jockey_past_starts",
    "jockey_past_top3",
    "jockey_past_top3_rate",
    "trainer_past_starts",
    "trainer_past_top3",
    "trainer_past_top3_rate",
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
        name: defaultdict(lambda: {"starts": 0, "top3": 0})
        for name, _ in entity_specs
    }

    df = df.sort_values(["date", "race_id", "horse_number"]).copy()
    history_rows = []

    for _, day_df in df.groupby("date", sort=True):
        for idx, row in day_df.iterrows():
            history_row = {"__idx": idx}
            for name, id_col in entity_specs:
                value = row[id_col]
                item = stats[name][value]
                starts = item["starts"]
                top3 = item["top3"]
                history_row[f"{name}_past_starts"] = starts
                history_row[f"{name}_past_top3"] = top3
                history_row[f"{name}_past_top3_rate"] = None if starts == 0 else top3 / starts
            history_rows.append(history_row)

        for _, row in day_df.iterrows():
            target = int(row["target_top3"])
            for name, id_col in entity_specs:
                item = stats[name][row[id_col]]
                item["starts"] += 1
                item["top3"] += target

    try:
        import pandas as pd
    except ModuleNotFoundError as e:
        raise RuntimeError("履歴特徴量の作成には pandas が必要です。") from e

    history_df = pd.DataFrame(history_rows).set_index("__idx")
    return df.join(history_df).sort_values(["date", "race_id", "horse_number"])


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
