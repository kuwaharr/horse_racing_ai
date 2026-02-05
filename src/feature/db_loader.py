import pandas as pd

from ..data.paths import DB_PATH
from ..data.database import connect


def load_train_df() -> pd.DataFrame:
    sql = """
    SELECT
        r.race_id,
        r.date,
        r.track_id,
        r.race_number,
        r.post_time_min,
        r.surface_id,
        r.distance,
        r.course_direction_id,
        r.course_layout_id,
        r.course_variant_id,
        r.weather_id,
        r.track_condition_id,
        r.race_size,

        ru.gate,
        ru.horse_number,
        ru.horse_name,
        ru.horse_id,
        ru.sex_id,
        ru.age,
        ru.jockey_id,
        ru.weight,
        ru.stable_id,
        ru.trainer_id,
        ru.horse_weight,
        ru.horse_weight_diff,

        ru.finish

    FROM race r
    JOIN runner ru USING (race_id);
    """

    with connect(DB_PATH) as conn:
        return pd.read_sql(sql, conn)

def load_eval_df() -> pd.DataFrame:
    sql = """
    SELECT
        r.race_id,
        ru.horse_number,
        po.odds_min,
        po.odds_max

    FROM race r
    JOIN runner ru USING (race_id)
    LEFT JOIN place_odds po
        ON po.race_id = ru.race_id AND po.horse_number = ru.horse_number;
    """

    with connect(DB_PATH) as conn:
        return pd.read_sql(sql, conn)