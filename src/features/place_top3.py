import sqlite3
from collections import defaultdict
from datetime import date
from math import isnan
from pathlib import Path


DEFAULT_TRAINING_DATASET_NAME = "place_top3_dataset.parquet"
DEFAULT_EVAL_ODDS_DATASET_NAME = "place_top3_eval_odds.parquet"


PEDIGREE_FEATURE_COLUMNS = [
    "sire_id",
    "dam_id",
    "broodmare_sire_id",
    "pedigree_available",
]


PEDIGREE_HISTORY_FEATURE_COLUMNS = [
    "sire_past_starts",
    "sire_past_top3",
    "sire_past_top3_rate",
    "sire_past_avg_finish",
    "sire_track_past_starts",
    "sire_track_past_top3",
    "sire_track_past_top3_rate",
    "sire_surface_past_starts",
    "sire_surface_past_top3",
    "sire_surface_past_top3_rate",
    "sire_distance_band_past_starts",
    "sire_distance_band_past_top3",
    "sire_distance_band_past_top3_rate",
    "dam_past_starts",
    "dam_past_top3",
    "dam_past_top3_rate",
    "dam_past_avg_finish",
    "broodmare_sire_past_starts",
    "broodmare_sire_past_top3",
    "broodmare_sire_past_top3_rate",
    "broodmare_sire_past_avg_finish",
    "broodmare_sire_track_past_starts",
    "broodmare_sire_track_past_top3",
    "broodmare_sire_track_past_top3_rate",
    "broodmare_sire_surface_past_starts",
    "broodmare_sire_surface_past_top3",
    "broodmare_sire_surface_past_top3_rate",
    "broodmare_sire_distance_band_past_starts",
    "broodmare_sire_distance_band_past_top3",
    "broodmare_sire_distance_band_past_top3_rate",
    "race_sire_past_top3_rate_rank",
    "race_sire_past_top3_rate_diff",
    "race_sire_past_avg_finish_rank",
    "race_sire_past_avg_finish_diff",
    "race_broodmare_sire_past_top3_rate_rank",
    "race_broodmare_sire_past_top3_rate_diff",
    "race_broodmare_sire_past_avg_finish_rank",
    "race_broodmare_sire_past_avg_finish_diff",
]


EVAL_ODDS_COLUMNS = [
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


TRAINING_FEATURE_COLUMNS = [
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

HISTORY_FEATURE_COLUMNS = [
    "horse_past_starts",
    "horse_past_top3",
    "horse_past_top3_rate",
    "horse_past_avg_finish",
    "horse_past_avg_time_per_200",
    "horse_past_avg_finish_diff",
    "horse_past_avg_finish_3f",
    "horse_past_avg_corner4",
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
    "horse_recent3_avg_time_per_200",
    "horse_recent3_avg_finish_diff",
    "horse_recent3_avg_finish_3f",
    "horse_recent3_avg_corner4",
    "horse_recent5_top3_rate",
    "horse_recent5_avg_finish",
    "horse_recent5_avg_time_per_200",
    "horse_recent5_avg_finish_diff",
    "horse_recent5_avg_finish_3f",
    "horse_recent5_avg_corner4",
    "horse_days_since_last",
    "horse_prev_distance",
    "horse_distance_diff_prev",
    "horse_track_past_starts",
    "horse_track_past_top3",
    "horse_track_past_top3_rate",
    "horse_track_past_start_share",
    "horse_track_past_top3_rate_lift",
    "horse_surface_past_starts",
    "horse_surface_past_top3",
    "horse_surface_past_top3_rate",
    "horse_surface_past_start_share",
    "horse_surface_past_top3_rate_lift",
    "horse_distance_band_past_starts",
    "horse_distance_band_past_top3",
    "horse_distance_band_past_top3_rate",
    "horse_distance_band_past_start_share",
    "horse_distance_band_past_top3_rate_lift",
    "jockey_past_starts",
    "jockey_past_top3",
    "jockey_past_top3_rate",
    "jockey_past_avg_finish",
    "jockey_recent20_top3_rate",
    "jockey_recent20_avg_finish",
    "jockey_track_past_starts",
    "jockey_track_past_top3",
    "jockey_track_past_top3_rate",
    "jockey_track_past_start_share",
    "jockey_track_past_top3_rate_lift",
    "jockey_surface_past_starts",
    "jockey_surface_past_top3",
    "jockey_surface_past_top3_rate",
    "jockey_surface_past_start_share",
    "jockey_surface_past_top3_rate_lift",
    "jockey_course_past_starts",
    "jockey_course_past_top3",
    "jockey_course_past_top3_rate",
    "jockey_course_past_start_share",
    "jockey_course_past_top3_rate_lift",
    "trainer_past_starts",
    "trainer_past_top3",
    "trainer_past_top3_rate",
    "trainer_past_avg_finish",
    "trainer_recent20_top3_rate",
    "trainer_recent20_avg_finish",
    "trainer_track_past_starts",
    "trainer_track_past_top3",
    "trainer_track_past_top3_rate",
    "trainer_track_past_start_share",
    "trainer_track_past_top3_rate_lift",
    "trainer_surface_past_starts",
    "trainer_surface_past_top3",
    "trainer_surface_past_top3_rate",
    "trainer_surface_past_start_share",
    "trainer_surface_past_top3_rate_lift",
    "trainer_course_past_starts",
    "trainer_course_past_top3",
    "trainer_course_past_top3_rate",
    "trainer_course_past_start_share",
    "trainer_course_past_top3_rate_lift",
    "horse_jockey_past_starts",
    "horse_jockey_past_top3",
    "horse_jockey_past_top3_rate",
    "jockey_trainer_past_starts",
    "jockey_trainer_past_top3",
    "jockey_trainer_past_top3_rate",
    "horse_trainer_past_starts",
    "horse_trainer_past_top3",
    "horse_trainer_past_top3_rate",
    "race_horse_past_top3_rate_rank",
    "race_horse_past_top3_rate_diff",
    "race_horse_past_avg_finish_rank",
    "race_horse_past_avg_finish_diff",
    "race_horse_past_avg_time_per_200_rank",
    "race_horse_past_avg_time_per_200_diff",
    "race_horse_past_avg_finish_diff_rank",
    "race_horse_past_avg_finish_diff_diff",
    "race_horse_past_avg_finish_3f_rank",
    "race_horse_past_avg_finish_3f_diff",
    "race_horse_past_avg_corner4_rank",
    "race_horse_past_avg_corner4_diff",
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
    "race_horse_recent3_avg_time_per_200_rank",
    "race_horse_recent3_avg_time_per_200_diff",
    "race_horse_recent3_avg_finish_diff_rank",
    "race_horse_recent3_avg_finish_diff_diff",
    "race_horse_recent3_avg_finish_3f_rank",
    "race_horse_recent3_avg_finish_3f_diff",
    "race_horse_recent3_avg_corner4_rank",
    "race_horse_recent3_avg_corner4_diff",
    "race_horse_recent5_avg_time_per_200_rank",
    "race_horse_recent5_avg_time_per_200_diff",
    "race_horse_recent5_avg_finish_diff_rank",
    "race_horse_recent5_avg_finish_diff_diff",
    "race_horse_recent5_avg_finish_3f_rank",
    "race_horse_recent5_avg_finish_3f_diff",
    "race_horse_recent5_avg_corner4_rank",
    "race_horse_recent5_avg_corner4_diff",
    "race_horse_days_since_last_rank",
    "race_horse_days_since_last_diff",
    "race_horse_distance_diff_prev_rank",
    "race_horse_distance_diff_prev_diff",
    "race_horse_track_past_top3_rate_rank",
    "race_horse_track_past_top3_rate_diff",
    "race_horse_track_past_start_share_rank",
    "race_horse_track_past_start_share_diff",
    "race_horse_track_past_top3_rate_lift_rank",
    "race_horse_track_past_top3_rate_lift_diff",
    "race_horse_surface_past_top3_rate_rank",
    "race_horse_surface_past_top3_rate_diff",
    "race_horse_surface_past_start_share_rank",
    "race_horse_surface_past_start_share_diff",
    "race_horse_surface_past_top3_rate_lift_rank",
    "race_horse_surface_past_top3_rate_lift_diff",
    "race_horse_distance_band_past_top3_rate_rank",
    "race_horse_distance_band_past_top3_rate_diff",
    "race_horse_distance_band_past_start_share_rank",
    "race_horse_distance_band_past_start_share_diff",
    "race_horse_distance_band_past_top3_rate_lift_rank",
    "race_horse_distance_band_past_top3_rate_lift_diff",
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
    "race_jockey_track_past_start_share_rank",
    "race_jockey_track_past_start_share_diff",
    "race_jockey_track_past_top3_rate_lift_rank",
    "race_jockey_track_past_top3_rate_lift_diff",
    "race_jockey_surface_past_top3_rate_rank",
    "race_jockey_surface_past_top3_rate_diff",
    "race_jockey_surface_past_start_share_rank",
    "race_jockey_surface_past_start_share_diff",
    "race_jockey_surface_past_top3_rate_lift_rank",
    "race_jockey_surface_past_top3_rate_lift_diff",
    "race_jockey_course_past_top3_rate_rank",
    "race_jockey_course_past_top3_rate_diff",
    "race_jockey_course_past_start_share_rank",
    "race_jockey_course_past_start_share_diff",
    "race_jockey_course_past_top3_rate_lift_rank",
    "race_jockey_course_past_top3_rate_lift_diff",
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
    "race_trainer_track_past_start_share_rank",
    "race_trainer_track_past_start_share_diff",
    "race_trainer_track_past_top3_rate_lift_rank",
    "race_trainer_track_past_top3_rate_lift_diff",
    "race_trainer_surface_past_top3_rate_rank",
    "race_trainer_surface_past_top3_rate_diff",
    "race_trainer_surface_past_start_share_rank",
    "race_trainer_surface_past_start_share_diff",
    "race_trainer_surface_past_top3_rate_lift_rank",
    "race_trainer_surface_past_top3_rate_lift_diff",
    "race_trainer_course_past_top3_rate_rank",
    "race_trainer_course_past_top3_rate_diff",
    "race_trainer_course_past_start_share_rank",
    "race_trainer_course_past_start_share_diff",
    "race_trainer_course_past_top3_rate_lift_rank",
    "race_trainer_course_past_top3_rate_lift_diff",
    "race_horse_jockey_past_top3_rate_rank",
    "race_horse_jockey_past_top3_rate_diff",
    "race_jockey_trainer_past_top3_rate_rank",
    "race_jockey_trainer_past_top3_rate_diff",
    "race_horse_trainer_past_top3_rate_rank",
    "race_horse_trainer_past_top3_rate_diff",
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
    ho.sire_id,
    ho.dam_id,
    ho.broodmare_sire_id,
    CASE WHEN ho.pedigree_fetch_status = 'fetched' THEN 1 ELSE 0 END AS pedigree_available,
    ru.sex_id,
    ru.age,
    ru.jockey_id,
    ru.weight,
    ru.time_sec,
    ru.finish_diff,
    ru.popularity,
    ru.finish_3f,
    ru.corner_4,
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
LEFT JOIN horse ho
    ON ho.horse_id = ru.horse_id
WHERE ru.status_id = 0
  AND ru.finish IS NOT NULL
ORDER BY ra.date, ra.race_id, ru.horse_number
"""


def default_training_dataset_name() -> str:
    return DEFAULT_TRAINING_DATASET_NAME


def default_eval_odds_dataset_name() -> str:
    return DEFAULT_EVAL_ODDS_DATASET_NAME


def _append_history_features(df):
    entity_specs = [
        ("horse", "horse_id"),
        ("jockey", "jockey_id"),
        ("trainer", "trainer_id"),
    ]
    pedigree_entity_specs = [
        ("sire", "sire_id"),
        ("dam", "dam_id"),
        ("broodmare_sire", "broodmare_sire_id"),
    ]
    pedigree_entity_specs = [
        (name, id_col)
        for name, id_col in pedigree_entity_specs
        if id_col in df.columns
    ]
    stats = {
        name: defaultdict(_empty_entity_stat)
        for name, _ in entity_specs + pedigree_entity_specs
    }
    horse_track_stats = defaultdict(_empty_history_stat)
    horse_surface_stats = defaultdict(_empty_history_stat)
    horse_distance_band_stats = defaultdict(_empty_history_stat)
    jockey_track_stats = defaultdict(_empty_history_stat)
    jockey_surface_stats = defaultdict(_empty_history_stat)
    jockey_course_stats = defaultdict(_empty_history_stat)
    trainer_track_stats = defaultdict(_empty_history_stat)
    trainer_surface_stats = defaultdict(_empty_history_stat)
    trainer_course_stats = defaultdict(_empty_history_stat)
    horse_jockey_stats = defaultdict(_empty_history_stat)
    jockey_trainer_stats = defaultdict(_empty_history_stat)
    horse_trainer_stats = defaultdict(_empty_history_stat)
    sire_track_stats = defaultdict(_empty_history_stat)
    sire_surface_stats = defaultdict(_empty_history_stat)
    sire_distance_band_stats = defaultdict(_empty_history_stat)
    broodmare_sire_track_stats = defaultdict(_empty_history_stat)
    broodmare_sire_surface_stats = defaultdict(_empty_history_stat)
    broodmare_sire_distance_band_stats = defaultdict(_empty_history_stat)
    recent_history = {
        name: defaultdict(list)
        for name, _ in entity_specs + pedigree_entity_specs
    }
    horse_last_seen = {}

    df = df.sort_values(["date", "race_id", "horse_number"]).copy()
    df["distance_band"] = df["distance"].map(_distance_band)
    df["date_obj"] = df["date"].map(date.fromisoformat)
    history_rows = []

    for _, day_df in df.groupby("date", sort=True):
        for idx, row in day_df.iterrows():
            history_row = {"__idx": idx}
            course_key = (row["track_id"], row["surface_id"], row["distance_band"])
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
                    time_count = item["time_per_200_count"]
                    finish_diff_count = item["finish_diff_count"]
                    finish_3f_count = item["finish_3f_count"]
                    corner4_count = item["corner4_count"]
                    history_row["horse_past_avg_time_per_200"] = (
                        None if time_count == 0 else item["time_per_200_sum"] / time_count
                    )
                    history_row["horse_past_avg_finish_diff"] = (
                        None if finish_diff_count == 0 else item["finish_diff_sum"] / finish_diff_count
                    )
                    history_row["horse_past_avg_finish_3f"] = (
                        None if finish_3f_count == 0 else item["finish_3f_sum"] / finish_3f_count
                    )
                    history_row["horse_past_avg_corner4"] = (
                        None if corner4_count == 0 else item["corner4_sum"] / corner4_count
                    )
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
                    history_row["horse_recent3_avg_time_per_200"] = _recent_avg_value(recent, "time_per_200", 3)
                    history_row["horse_recent3_avg_finish_diff"] = _recent_avg_value(recent, "finish_diff", 3)
                    history_row["horse_recent3_avg_finish_3f"] = _recent_avg_value(recent, "finish_3f", 3)
                    history_row["horse_recent3_avg_corner4"] = _recent_avg_value(recent, "corner4", 3)
                    history_row["horse_recent5_top3_rate"] = _recent_top3_rate(recent, 5)
                    history_row["horse_recent5_avg_finish"] = _recent_avg_finish(recent, 5)
                    history_row["horse_recent5_avg_time_per_200"] = _recent_avg_value(recent, "time_per_200", 5)
                    history_row["horse_recent5_avg_finish_diff"] = _recent_avg_value(recent, "finish_diff", 5)
                    history_row["horse_recent5_avg_finish_3f"] = _recent_avg_value(recent, "finish_3f", 5)
                    history_row["horse_recent5_avg_corner4"] = _recent_avg_value(recent, "corner4", 5)
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
            for name, id_col in pedigree_entity_specs:
                value = row[id_col]
                if _missing_key(value):
                    starts = 0
                    top3 = 0
                    finish_sum = 0.0
                else:
                    item = stats[name][value]
                    starts = item["starts"]
                    top3 = item["top3"]
                    finish_sum = item["finish_sum"]
                history_row[f"{name}_past_starts"] = starts
                history_row[f"{name}_past_top3"] = top3
                history_row[f"{name}_past_top3_rate"] = None if starts == 0 else top3 / starts
                history_row[f"{name}_past_avg_finish"] = None if starts == 0 else finish_sum / starts
            for feature_name, item in [
                ("horse_track", horse_track_stats[(row["horse_id"], row["track_id"])]),
                ("horse_surface", horse_surface_stats[(row["horse_id"], row["surface_id"])]),
                (
                    "horse_distance_band",
                    horse_distance_band_stats[(row["horse_id"], row["distance_band"])],
                ),
                ("jockey_track", jockey_track_stats[(row["jockey_id"], row["track_id"])]),
                ("jockey_surface", jockey_surface_stats[(row["jockey_id"], row["surface_id"])]),
                ("jockey_course", jockey_course_stats[(row["jockey_id"], course_key)]),
                ("trainer_track", trainer_track_stats[(row["trainer_id"], row["track_id"])]),
                ("trainer_surface", trainer_surface_stats[(row["trainer_id"], row["surface_id"])]),
                ("trainer_course", trainer_course_stats[(row["trainer_id"], course_key)]),
                ("horse_jockey", horse_jockey_stats[(row["horse_id"], row["jockey_id"])]),
                ("jockey_trainer", jockey_trainer_stats[(row["jockey_id"], row["trainer_id"])]),
                ("horse_trainer", horse_trainer_stats[(row["horse_id"], row["trainer_id"])]),
            ]:
                starts = item["starts"]
                top3 = item["top3"]
                top3_rate = None if starts == 0 else top3 / starts
                history_row[f"{feature_name}_past_starts"] = starts
                history_row[f"{feature_name}_past_top3"] = top3
                history_row[f"{feature_name}_past_top3_rate"] = top3_rate
                owner = _affinity_owner(feature_name)
                if owner is not None:
                    owner_starts = history_row[f"{owner}_past_starts"]
                    owner_top3_rate = history_row[f"{owner}_past_top3_rate"]
                    history_row[f"{feature_name}_past_start_share"] = (
                        None if owner_starts == 0 else starts / owner_starts
                    )
                    history_row[f"{feature_name}_past_top3_rate_lift"] = (
                        None if top3_rate is None or owner_top3_rate is None else top3_rate - owner_top3_rate
                    )
            for feature_name, id_col, stat_map, key_value in [
                ("sire_track", "sire_id", sire_track_stats, row["track_id"]),
                ("sire_surface", "sire_id", sire_surface_stats, row["surface_id"]),
                ("sire_distance_band", "sire_id", sire_distance_band_stats, row["distance_band"]),
                (
                    "broodmare_sire_track",
                    "broodmare_sire_id",
                    broodmare_sire_track_stats,
                    row["track_id"],
                ),
                (
                    "broodmare_sire_surface",
                    "broodmare_sire_id",
                    broodmare_sire_surface_stats,
                    row["surface_id"],
                ),
                (
                    "broodmare_sire_distance_band",
                    "broodmare_sire_id",
                    broodmare_sire_distance_band_stats,
                    row["distance_band"],
                ),
            ]:
                if id_col not in df.columns or _missing_key(row[id_col]):
                    item = _empty_history_stat()
                else:
                    item = stat_map[(row[id_col], key_value)]
                starts = item["starts"]
                top3 = item["top3"]
                history_row[f"{feature_name}_past_starts"] = starts
                history_row[f"{feature_name}_past_top3"] = top3
                history_row[f"{feature_name}_past_top3_rate"] = None if starts == 0 else top3 / starts
            history_rows.append(history_row)

        for _, row in day_df.iterrows():
            target = int(row["target_top3"])
            finish = float(row["finish_position"])
            course_key = (row["track_id"], row["surface_id"], row["distance_band"])
            for name, id_col in entity_specs:
                item = stats[name][row[id_col]]
                item["starts"] += 1
                item["top3"] += target
                item["finish_sum"] += finish
                if name == "horse":
                    distance = _valid_float(row["distance"])
                    time_sec = _valid_float(row["time_sec"])
                    if time_sec is not None and distance:
                        time_per_200 = time_sec / distance * 200.0
                        item["time_per_200_sum"] += time_per_200
                        item["time_per_200_count"] += 1
                    else:
                        time_per_200 = None
                    finish_diff = _valid_float(row["finish_diff"])
                    if finish_diff is not None:
                        item["finish_diff_sum"] += finish_diff
                        item["finish_diff_count"] += 1
                    finish_3f = _valid_float(row["finish_3f"])
                    if finish_3f is not None:
                        item["finish_3f_sum"] += finish_3f
                        item["finish_3f_count"] += 1
                    corner4 = _valid_float(row["corner_4"])
                    if corner4 is not None:
                        item["corner4_sum"] += corner4
                        item["corner4_count"] += 1
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
                recent_item = {"top3": target, "finish": finish}
                if name == "horse":
                    recent_item.update(
                        {
                            "time_per_200": time_per_200,
                            "finish_diff": finish_diff,
                            "finish_3f": finish_3f,
                            "corner4": corner4,
                        }
                    )
                recent_history[name][row[id_col]].append(recent_item)
            for name, id_col in pedigree_entity_specs:
                value = row[id_col]
                if _missing_key(value):
                    continue
                item = stats[name][value]
                item["starts"] += 1
                item["top3"] += target
                item["finish_sum"] += finish
                recent_history[name][value].append({"top3": target, "finish": finish})
            horse_last_seen[row["horse_id"]] = (row["date_obj"], int(row["distance"]))
            for item in [
                horse_track_stats[(row["horse_id"], row["track_id"])],
                horse_surface_stats[(row["horse_id"], row["surface_id"])],
                horse_distance_band_stats[(row["horse_id"], row["distance_band"])],
                jockey_track_stats[(row["jockey_id"], row["track_id"])],
                jockey_surface_stats[(row["jockey_id"], row["surface_id"])],
                jockey_course_stats[(row["jockey_id"], course_key)],
                trainer_track_stats[(row["trainer_id"], row["track_id"])],
                trainer_surface_stats[(row["trainer_id"], row["surface_id"])],
                trainer_course_stats[(row["trainer_id"], course_key)],
                horse_jockey_stats[(row["horse_id"], row["jockey_id"])],
                jockey_trainer_stats[(row["jockey_id"], row["trainer_id"])],
                horse_trainer_stats[(row["horse_id"], row["trainer_id"])],
            ]:
                item["starts"] += 1
                item["top3"] += target
            if "sire_id" in df.columns and not _missing_key(row["sire_id"]):
                for item in [
                    sire_track_stats[(row["sire_id"], row["track_id"])],
                    sire_surface_stats[(row["sire_id"], row["surface_id"])],
                    sire_distance_band_stats[(row["sire_id"], row["distance_band"])],
                ]:
                    item["starts"] += 1
                    item["top3"] += target
            if "broodmare_sire_id" in df.columns and not _missing_key(row["broodmare_sire_id"]):
                for item in [
                    broodmare_sire_track_stats[(row["broodmare_sire_id"], row["track_id"])],
                    broodmare_sire_surface_stats[(row["broodmare_sire_id"], row["surface_id"])],
                    broodmare_sire_distance_band_stats[(row["broodmare_sire_id"], row["distance_band"])],
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
        "time_per_200_sum": 0.0,
        "time_per_200_count": 0,
        "finish_diff_sum": 0.0,
        "finish_diff_count": 0,
        "finish_3f_sum": 0.0,
        "finish_3f_count": 0,
        "corner4_sum": 0.0,
        "corner4_count": 0,
        "min_distance": None,
        "max_distance": None,
    }


def _affinity_owner(feature_name: str) -> str | None:
    owner_by_feature = {
        "horse_track": "horse",
        "horse_surface": "horse",
        "horse_distance_band": "horse",
        "jockey_track": "jockey",
        "jockey_surface": "jockey",
        "jockey_course": "jockey",
        "trainer_track": "trainer",
        "trainer_surface": "trainer",
        "trainer_course": "trainer",
    }
    return owner_by_feature.get(feature_name)


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


def _recent_avg_value(rows: list[dict], key: str, n: int) -> float | None:
    values = [row[key] for row in rows[-n:] if row.get(key) is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _valid_float(value) -> float | None:
    if value is None:
        return None
    value = float(value)
    if isnan(value):
        return None
    return value


def _missing_key(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and isnan(value):
        return True
    return value == ""


def _distance_band(distance: int) -> str:
    if distance < 1400:
        return "under1400"
    if distance < 1800:
        return "1400_1799"
    if distance < 2200:
        return "1800_2199"
    return "2200plus"


def _append_race_relative_features(df):
    import pandas as pd

    relative_specs = [
        ("horse_past_top3_rate", False),
        ("horse_past_avg_finish", True),
        ("horse_past_avg_time_per_200", True),
        ("horse_past_avg_finish_diff", True),
        ("horse_past_avg_finish_3f", True),
        ("horse_past_avg_corner4", True),
        ("horse_distance_diff_avg", True),
        ("horse_distance_above_max", True),
        ("horse_distance_below_min", True),
        ("horse_weight_diff_avg", True),
        ("horse_recent3_top3_rate", False),
        ("horse_recent3_avg_finish", True),
        ("horse_recent3_avg_time_per_200", True),
        ("horse_recent3_avg_finish_diff", True),
        ("horse_recent3_avg_finish_3f", True),
        ("horse_recent3_avg_corner4", True),
        ("horse_recent5_avg_time_per_200", True),
        ("horse_recent5_avg_finish_diff", True),
        ("horse_recent5_avg_finish_3f", True),
        ("horse_recent5_avg_corner4", True),
        ("horse_days_since_last", True),
        ("horse_distance_diff_prev", True),
        ("horse_track_past_top3_rate", False),
        ("horse_track_past_start_share", False),
        ("horse_track_past_top3_rate_lift", False),
        ("horse_surface_past_top3_rate", False),
        ("horse_surface_past_start_share", False),
        ("horse_surface_past_top3_rate_lift", False),
        ("horse_distance_band_past_top3_rate", False),
        ("horse_distance_band_past_start_share", False),
        ("horse_distance_band_past_top3_rate_lift", False),
        ("jockey_past_top3_rate", False),
        ("jockey_past_avg_finish", True),
        ("jockey_recent20_top3_rate", False),
        ("jockey_recent20_avg_finish", True),
        ("jockey_track_past_top3_rate", False),
        ("jockey_track_past_start_share", False),
        ("jockey_track_past_top3_rate_lift", False),
        ("jockey_surface_past_top3_rate", False),
        ("jockey_surface_past_start_share", False),
        ("jockey_surface_past_top3_rate_lift", False),
        ("jockey_course_past_top3_rate", False),
        ("jockey_course_past_start_share", False),
        ("jockey_course_past_top3_rate_lift", False),
        ("trainer_past_top3_rate", False),
        ("trainer_past_avg_finish", True),
        ("trainer_recent20_top3_rate", False),
        ("trainer_recent20_avg_finish", True),
        ("trainer_track_past_top3_rate", False),
        ("trainer_track_past_start_share", False),
        ("trainer_track_past_top3_rate_lift", False),
        ("trainer_surface_past_top3_rate", False),
        ("trainer_surface_past_start_share", False),
        ("trainer_surface_past_top3_rate_lift", False),
        ("trainer_course_past_top3_rate", False),
        ("trainer_course_past_start_share", False),
        ("trainer_course_past_top3_rate_lift", False),
        ("horse_jockey_past_top3_rate", False),
        ("jockey_trainer_past_top3_rate", False),
        ("horse_trainer_past_top3_rate", False),
        ("sire_past_top3_rate", False),
        ("sire_past_avg_finish", True),
        ("broodmare_sire_past_top3_rate", False),
        ("broodmare_sire_past_avg_finish", True),
        ("gate", True),
        ("horse_number", True),
        ("weight", False),
        ("age", True),
    ]
    relative_columns = {}
    for col, ascending in relative_specs:
        rank_col = f"race_{col}_rank"
        diff_col = f"race_{col}_diff"
        filled = df[col].fillna(-1)
        relative_columns[rank_col] = filled.groupby(df["race_id"]).rank(method="average", ascending=ascending)
        relative_columns[diff_col] = df[col] - df.groupby("race_id")[col].transform("mean")
    return pd.concat([df, pd.DataFrame(relative_columns, index=df.index)], axis=1)


def build_place_top3_dataset(
    db_path: Path,
    output_path: Path,
    engine: str = "auto",
    history_features: bool = True,
    pedigree_features: bool = True,
) -> int:
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

    columns = TRAINING_FEATURE_COLUMNS
    if pedigree_features:
        horse_name_index = columns.index("horse_name")
        columns = (
            columns[: horse_name_index + 1]
            + PEDIGREE_FEATURE_COLUMNS
            + columns[horse_name_index + 1 :]
        )
    if history_features:
        history_columns = HISTORY_FEATURE_COLUMNS
        if pedigree_features:
            history_columns = history_columns + PEDIGREE_HISTORY_FEATURE_COLUMNS
        columns = columns[:-1] + history_columns + columns[-1:]
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


def build_place_top3_eval_odds_dataset(
    db_path: Path,
    output_path: Path,
    engine: str = "auto",
) -> int:
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

    df = df[EVAL_ODDS_COLUMNS]

    try:
        df.to_parquet(output_path, index=False, engine=engine)
    except ImportError as e:
        raise RuntimeError(
            "Parquet出力エンジンが見つかりません。"
            " PowerShellで `pip install pyarrow` を実行するか、"
            " `--engine fastparquet` を指定してください。"
        ) from e

    return len(df)
