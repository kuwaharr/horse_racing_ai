import pandas as pd

from src.data.data_path import DB_PATH, FEAT_DIR
from src.data.db import connect


def main():
    sql = """
    SELECT
        r.race_id,
        r.date,
        r.race_size,
        ru.horse_number,
        ru.horse_id,
        ru.finish

    FROM race r
    JOIN runner ru USING (race_id)
    """

    with connect(DB_PATH) as conn:
        df = pd.read_sql(sql, conn)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna() & df["finish"].notna() & df["race_size"].notna()].copy()
    df["finish"] = df["finish"].astype(int)
    df["race_size"] = df["race_size"].astype(int)

    df["is_place"] = df.apply(
        lambda r: is_place(r["finish"], r["race_size"]),
        axis=1
    ).astype(int)

    df = df.sort_values(["horse_id", "date", "race_id"]).reset_index(drop=True)

    df["place_rate_last3"] = (
        df.groupby("horse_id")["is_place"]
        .apply(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )

    df["n_runs_past"] = (
        df.groupby("horse_id").cumcount()
    )

    out = df[["race_id", "horse_number", "place_rate_last3", "n_runs_past"]].copy()

    horse_hist_path = FEAT_DIR / "horse_hist.parquet"
    out.to_parquet(horse_hist_path, index=False)
    print("OK")
    print(out.head(10))


if __name__ == "__main__":
    main()