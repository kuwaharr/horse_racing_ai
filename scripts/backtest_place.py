import numpy as np
import pandas as pd

from src.evaluation.place import eval_ev
from src.feature.dataset_builders import (
    add_is_place_column,
    add_horse_finish_last3_best_column,
    add_horse_finish_last3_mean_column,
    add_horse_place_last3_mean_column,
    add_jockey_finish_last3_best_column,
    add_jockey_finish_last3_mean_column,
    add_onehot_columns,
    add_trainer_finish_last3_best_column,
    add_trainer_finish_last3_mean_column,
    calc_odds_mid,
    convert_date_column,
)
from src.feature.db_loader import load_eval_df, load_train_df
from src.feature.parquet_exporter import export_parquet
from src.model.calibrators import calibrate_isotonic, calibrate_platt
from src.model.trainer import make_oof_predictions


def main():
    train_df = load_train_df()

    train_df = train_df.dropna(subset=["finish"])
    train_df = add_is_place_column(train_df)

    train_df = convert_date_column(train_df)

    train_df = train_df.dropna(subset=["horse_id"])
    train_df = add_horse_finish_last3_best_column(train_df)
    train_df = add_horse_finish_last3_mean_column(train_df)
    train_df = add_horse_place_last3_mean_column(train_df)

    train_df = train_df.dropna(subset=["jockey_id"])
    train_df = add_jockey_finish_last3_best_column(train_df)
    train_df = add_jockey_finish_last3_mean_column(train_df)

    train_df = train_df.dropna(subset=["trainer_id"])
    train_df = add_trainer_finish_last3_best_column(train_df)
    train_df = add_trainer_finish_last3_mean_column(train_df)

    train_keys = train_df[["race_id", "horse_number", "is_place"]].reset_index(drop=True)
    groups = train_df["race_id"].reset_index(drop=True)

    train_df = train_df.drop(columns=["race_id", "horse_id", "jockey_id", "trainer_id"])
    train_df = add_onehot_columns(train_df)

    print(train_df.shape)
    print(train_df.columns)
    print(train_df.head(10))
    print("")

    X = train_df.copy().drop(columns=["date", "race_number", "horse_number", "horse_name", "finish", "is_place"])
    y = train_df["is_place"].copy()

    export_parquet("X.parquet", X)
    export_parquet("y.parquet", pd.DataFrame(y))

    pred = make_oof_predictions(X, y, groups, n_splits=5)
    print("")

    eval_df = load_eval_df()
    eval_df = train_keys.merge(eval_df, on=["race_id", "horse_number"], how="left")

    eval_df["odds_mid"] = eval_df.apply(
        lambda r: calc_odds_mid(r["odds_min"], r["odds_max"]),
        axis=1
    )

    eval_df["pred"] = pred

    pred_np = eval_df["pred"].values
    is_place_np = eval_df["is_place"].values

    eval_df["pred_platt"] = calibrate_platt(pred_np, is_place_np)
    eval_df["pred_iso"] = calibrate_isotonic(pred_np, is_place_np)

    print(eval_df.shape)
    print(eval_df.columns)
    print(eval_df.head(10))

    for name in ["pred_platt", "pred_iso"]:
        eval_df["ev"] = eval_df[name] * eval_df["odds_mid"]

        print(f"\n=== {name} ===")
        for ev_th in [1.00, 1.05, 1.10, 1.20]:
            res = eval_ev(eval_df, ev_th, max_k=3)
            print(res)
    print("")

    print(eval_df["ev"].describe())
    print(eval_df["ev"].quantile([0.9, 0.95, 0.99]))


if __name__ == "__main__":
    main()