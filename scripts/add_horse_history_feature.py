import pandas as pd

from src.common.config import DB_PATH
from src.common.db import connect


def is_place(finish: int, race_size: int) -> int:
    if race_size <= 7:
        return int(finish <= 2)
    if race_size >= 8:
        return int(finish <= 3)


def main():
    sql = """
    SELECT
        r.race_id,
        r.date,
        r.race_size,
        ru.horse_id,
        ru.finish

    FROM race r
    JOIN runner ru USING (race_id)
    """

    with connect(DB_PATH) as conn:
        df = pd.read_sql(sql, conn)


if __name__ == "__main__":
    main()