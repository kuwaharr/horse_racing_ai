import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def make_oof_predictions(X: pd.DataFrame, y: pd.Series, groups: pd.Series, n_splits: int = 5) -> np.ndarray:
    oof = np.zeros(len(X), dtype=float)

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.9,
        "baggin_fraction": 0.9,
        "verbosity": -1,
        "seed": 42,
    }

    gkf = GroupKFold(n_splits=n_splits)

    for fold, (train_idx, valid_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

        dtrain = lgb.Dataset(X_train, label=y_train)
        dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain)

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=2000,
            valid_sets=[dvalid],
            callbacks=[
                lgb.early_stopping(50),
                lgb.log_evaluation(200)
            ]
        )

        oof[valid_idx] = model.predict(X_valid)

        print(f"[fold {fold}] done. best_iter={model.best_iteration}")

    return oof