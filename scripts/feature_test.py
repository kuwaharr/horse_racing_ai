from __future__ import annotations

import pandas as pd

from src.common.config import FEAT_DIR
from src.feature.load_db import load_predict_df


def main():
    df = load_predict_df()
    print(df.head(10))

    df = df[df["finish"].notna()].copy()
    df["finish"] = df["finish"].astype(int)
    y = (df["finish"] <= 3).astype(int)

    meta = df[["race_id", "horse_number", "horse_name", "date"]].copy()

    X = df.drop(columns=["finish", "horse_name", "race_id", "date"], errors="ignore")

    cat_cols = [c for c in X.columns if c.endswith("_id")]
    X = pd.get_dummies(X, columns=cat_cols, dummy_na=True)

    num_cols = X.select_dtypes(include=["number"]).columns
    X[num_cols] = X[num_cols].fillna(0)

    x_path = FEAT_DIR / "x.parquet"
    y_path = FEAT_DIR / "y.parquet"
    meta_path = FEAT_DIR / "meta.parquet"

    X.to_parquet(x_path, index=False)
    y.to_frame("y").to_parquet(y_path, index=False)
    meta.to_parquet(meta_path, index=False)

    print("X: ", X.shape, "y: ", y.shape, "meta: ", meta.shape)
    print("example columns:", list(X.columns)[:20])


if __name__ == "__main__":
    main()