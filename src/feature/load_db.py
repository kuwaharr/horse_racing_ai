import pandas as pd

from ..data.data_path import DB_PATH
from ..data.db import connect


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
        ru.sex_id,
        ru.age,
        ru.weight,
        ru.stable_id,
        ru.horse_weight,
        ru.horse_weight_diff,

        ru.finish

    FROM race r
    JOIN runner ru USING (race_id);
    """

    with connect(DB_PATH) as conn:
        return pd.read_sql(sql, conn)