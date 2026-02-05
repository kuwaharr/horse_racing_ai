import pandas as pd


def is_place(finish: int, race_size: int) -> int:
    if race_size <= 7:
        return int(finish <= 2)
    return int(finish <= 3)


def add_is_place_column(df: pd.DataFrame) -> pd.DataFrame:
    # finish and race_size column required and cannot be none
    df["finish"] = df["finish"].astype(int)
    df["race_size"] = df["race_size"].astype(int)

    df["is_place"] = df.apply(
        lambda r: is_place(r["finish"], r["race_size"]),
        axis=1
    ).astype(int)

    return df


def convert_date_column(df: pd.DataFrame) -> pd.DataFrame:
    # date column required
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def add_n_horse_races_past_column(df: pd.DataFrame) -> pd.DataFrame:
    df["n_horse_races_past"] = (
        df.groupby("horse_id").cumcount()
    )
    return df


def add_n_jockey_races_past_column(df: pd.DataFrame) -> pd.DataFrame:
    df["n_jockey_races_past"] = (
        df.groupby("jockey_id").cumcount()
    )
    return df


def add_n_trainer_races_past_column(df: pd.DataFrame) -> pd.DataFrame:
    df["n_trainer_races_past"] = (
        df.groupby("trainer_id").cumcount()
    )
    return df


def add_n_trainer_races_past_column(df: pd.DataFrame) -> pd.DataFrame:
    df["n_trainer_races_past_column"] = (
        df.groupby("trainer_id").cumcount()
    )
    return df


def add_horse_finish_last3_mean_column(df: pd.DataFrame) -> pd.DataFrame:
    # horse_id, date(converted), and finish column required and cannot be none
    df = df.sort_values(["horse_id", "date", "race_id"]).reset_index(drop=True)

    df["horse_finish_last3_mean"] = (
        df.groupby("horse_id")["finish"]
        .apply(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )
    return df


def add_horse_finish_last3_best_column(df: pd.DataFrame) -> pd.DataFrame:
    # horse_id, date(converted), and finish column required and cannot be none
    df = df.sort_values(["horse_id", "date", "race_id"]).reset_index(drop=True)

    df["horse_finish_last3_best"] = (
        df.groupby("horse_id")["finish"]
        .apply(lambda s: s.shift(1).rolling(3, min_periods=1).min())
        .reset_index(level=0, drop=True)
    ).astype(int)


def add_horse_place_last3_mean_column(df: pd.DataFrame) -> pd.DataFrame:
    # horse_id, date(converted), and is_place column required and cannot be none
    df = df.sort_values(["horse_id", "date", "race_id"]).reset_index(drop=True)

    df["horse_place_last3_mean"] = (
        df.groupby("horse_id")["is_place"]
        .apply(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )
    return df


def add_jockey_finish_last3_mean_column(df: pd.DataFrame) -> pd.DataFrame:
    # jockey_id, date(converted), and finish column required and cannot be none
    df = df.sort_values(["jockey_id", "date", "race_id"]).reset_index(drop=True)

    df["jockey_finish_last3_mean"] = (
        df.groupby("jockey_id")["finish"]
        .apply(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )
    return df


def add_jockey_finish_last3_best_column(df: pd.DataFrame) -> pd.DataFrame:
    # jockey_id, date(converted), and finish column required and cannot be none
    df = df.sort_values(["jockey_id", "date", "race_id"]).reset_index(drop=True)

    df["jockey_finish_last3_best"] = (
        df.groupby("jockey_id")["finish"]
        .apply(lambda s: s.shift(1).rolling(3, min_periods=1).min())
        .reset_index(level=0, drop=True)
    )
    return df


def add_trainer_finish_last3_mean_column(df: pd.DataFrame) -> pd.DataFrame:
    # trainer_id, date(converted), and finish column required and cannot be none
    df = df.sort_values(["trainer_id", "date", "race_id"]).reset_index(drop=True)

    df["trainer_finish_last3_mean"] = (
        df.groupby("trainer_id")["finish"]
        .apply(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )
    return df


def add_trainer_finish_last3_best_column(df: pd.DataFrame) -> pd.DataFrame:
    # trainer_id, date(converted), and finish column required and cannot be none
    df = df.sort_values(["trainer_id", "date", "race_id"]).reset_index(drop=True)

    df["trainer_finish_last3_best"] = (
        df.groupby("trainer_id")["finish"]
        .apply(lambda s: s.shift(1).rolling(3, min_periods=1).min())
        .reset_index(level=0, drop=True)
    )
    return df


def add_onehot_columns(df: pd.DataFrame) -> pd.DataFrame:
    # encode columns where its name ends with _id
    cat_cols = [c for c in df.columns if c.endswith("_id")]

    df = pd.get_dummies(df, columns=cat_cols, dummy_na=True)
    return df