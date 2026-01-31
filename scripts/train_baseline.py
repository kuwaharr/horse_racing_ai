import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.data.data_path import FEAT_DIR


def main():
    x_path = FEAT_DIR / "x.parquet"
    y_path = FEAT_DIR / "y.parquet"
    meta_path = FEAT_DIR / "meta.parquet"

    X = pd.read_parquet(x_path)
    y = pd.read_parquet(y_path)["y"]
    meta = pd.read_parquet(meta_path)

    if "horse_weight_diff" in X.columns:
        X = X.drop(columns=["horse_weight_diff"])

    bad = [c for c in X.columns if any(k in c.lower() for k in ["finish", "time_sec", "finish_3f", "corner", "diff", "popularity", "odds", "status"])]
    print("suspicious columns:", bad[:50], "count=", len(bad))

    groups = meta["race_id"]

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, valid_idx = next(gss.split(X, y, groups=groups))

    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    model = LogisticRegression(max_iter=3000)
    model.fit(X_train, y_train)

    pred = model.predict_proba(X_valid)[:, 1]
    auc = roc_auc_score(y_valid, pred)
    print(f"AUC(group by race_id): {auc:.4f}")


if __name__ == "__main__":
    main()