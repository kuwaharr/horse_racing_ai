from __future__ import annotations

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from src.common.config import FEAT_DIR, DATA_ROOT


def main():
    x_path = FEAT_DIR / "x.parquet"
    y_path = FEAT_DIR / "y_place_rule.parquet"
    meta_path = FEAT_DIR / "meta.parquet"

    X = pd.read_parquet(x_path)
    y = pd.read_parquet(y_path)["y"]
    meta = pd.read_parquet(meta_path)

    groups = meta["race_id"]

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, valid_idx = next(gss.split(X, y, groups=groups))

    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain)

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "verbosity": -1,
        "seed": 42,
    }

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=500,
        valid_sets=[dvalid],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=50),
        ]
    )

    pred = model.predict(X_valid)
    auc = roc_auc_score(y_valid, pred)
    print(f"AUC(LightGBM, group by race_id): {auc:.4f}")

    model_path = DATA_ROOT / "model" / "lgbm_place_rule.txt"
    model.save_model(model_path)


if __name__ == "__main__":
    main()