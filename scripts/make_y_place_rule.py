import pandas as pd

from src.data.data_path import FEAT_DIR
from src.feature.load_db import load_train_df


def main():
    df = load_train_df()

    df = df[df["finish"].notna() & df["race_size"].notna()].copy()
    df["finish"] = df["finish"].astype(int)
    df["race_size"] = df["race_size"].astype(int)

    y = df.apply(
        lambda r: is_place(r["finish"], r["race_size"]),
        axis=1
    ).astype(int)

    meta = df[["race_id", "horse_number", "horse_name", "date"]].copy()

    y_place_rule_path = FEAT_DIR / "y_place_rule.parquet"
    meta_path = FEAT_DIR / "meta.parquet"
    y.to_frame("y").to_parquet(y_place_rule_path, index=False)
    meta.to_parquet(meta_path, index=False)

    print("OK:", y.shape, "positive_rate:", float(y.mean()))


if __name__ == "__main__":
    main()