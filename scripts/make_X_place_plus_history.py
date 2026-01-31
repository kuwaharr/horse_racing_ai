import pandas as pd

from src.common.config import FEAT_DIR


def main():
    X = pd.read_parquet(FEAT_DIR / "x.parquet")
    meta = pd.read_parquet(FEAT_DIR / "meta.parquet")

    hist = pd.read_parquet(FEAT_DIR / "horse_hist.parquet")

    m = meta[["race_id", "horse_number"]].copy()
    feat = m.merge(hist, on=["race_id", "horse_number"], how="left")

    feat["place_rate_last3"] = feat["place_rate_last3"].fillna(0).astype("float32")
    feat["n_runs_past"] = feat["n_runs_past"].fillna(0).astype("float32")

    X2 = X.copy()
    X2["place_rate_last3"] = feat["place_rate_last3"].values
    X2["n_runs_past"] = feat["n_runs_past"].values

    X2.to_parquet(FEAT_DIR / "x_plus_hist.parquet", index=False)
    print("OK:", X2.shape)


if __name__ == "__main__":
    main()