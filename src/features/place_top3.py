import sqlite3
from pathlib import Path


DEFAULT_DATASET_NAMES = {
    "early": "place_top3_early_dataset.parquet",
    "late": "place_top3_late_dataset.parquet",
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


def default_dataset_name(mode: str) -> str:
    if mode not in DEFAULT_DATASET_NAMES:
        raise ValueError(f"Unknown dataset mode: {mode}")
    return DEFAULT_DATASET_NAMES[mode]


def build_place_top3_dataset(db_path: Path, output_path: Path, mode: str = "late", engine: str = "auto") -> int:
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

    df = df[FEATURE_COLUMNS_BY_MODE[mode]]

    try:
        df.to_parquet(output_path, index=False, engine=engine)
    except ImportError as e:
        raise RuntimeError(
            "Parquet出力エンジンが見つかりません。"
            " PowerShellで `pip install pyarrow` を実行するか、"
            " `--engine fastparquet` を指定してください。"
        ) from e

    return len(df)
