import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from src.common.config import DB_PATH, FEAT_DIR
from src.common.db import connect


def is_place(finish: int, race_size: int) -> int:
    if race_size <= 7:
        return int(finish <= 2)
    return int(finish <= 3)


def odds_mid(odds_min, odds_max):
    if pd.isna(odds_min) and pd.isna(odds_max):
        return np.nan
    return float((odds_min + odds_max) / 2.0)


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

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]

        dtr = lgb.Dataset(X_tr, label=y_tr)
        dva = lgb.Dataset(X_va, label=y_va, reference=dtr)

        model = lgb.train(
            params,
            dtr,
            num_boost_round=2000,
            valid_sets=[dva],
            callbacks=[
                lgb.early_stopping(50),
                lgb.log_evaluation(200)
            ]
        )

        oof[va_idx] = model.predict(X_va)

        print(f"[fold {fold}] done. best_iter={model.best_iteration}")

    return oof


def load_labels_and_odds_from_db() -> pd.DataFrame:
    sql = """
    SELECT
        r.race_id,
        r.race_size,
        ru.horse_number,
        ru.finish,
        po.odds_min,
        po.odds_max

    FROM race r
    JOIN runner ru USING (race_id)
    LEFT JOIN place_odds po
        ON po.race_id = ru.race_id AND po.horse_number = ru.horse_number
    """

    with connect(DB_PATH) as conn:
        df = pd.read_sql(sql, conn)

    df = df[df["finish"].notna() & df["race_size"].notna()].copy()
    df["finish"] = df["finish"].astype(int)
    df["race_size"] = df["race_size"].astype(int)

    df["y_true"] = df.apply(
        lambda r: is_place(r["finish"], r["race_size"]),
        axis=1
    ).astype(int)
    df["odds_mid"] = df.apply(
        lambda r: odds_mid(r["odds_min"], r["odds_max"]),
        axis=1
    )

    return df[["race_id", "horse_number", "y_true", "odds_mid", "race_size"]]


def calibrate_platt(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    lr = LogisticRegression(solver="lbfgs", max_iter=1000)
    lr.fit(pred.reshape(-1, 1), y)
    return lr.predict_proba(pred.reshape(-1, 1))[:, 1]


def calibrate_isotonic(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(pred, y)
    return iso.predict(pred)


def eval_topk(df: pd.DataFrame, k: int) -> dict:
    picked = (
        df.sort_values(["race_id", "pred"], ascending=[True, False])
        .groupby("race_id")
        .head(k)
        .copy()
    )

    bets = len(picked)
    hits = int(picked["y_true"].sum())
    hit_rate = hits / bets if bets else 0.0

    roi_df = picked[picked["odds_mid"].notna()].copy()
    bet_roi = len(roi_df)
    ret = float((roi_df["y_true"] * roi_df["odds_mid"]).sum())
    roi = ret / bet_roi if bet_roi else np.nan

    return {
        "k": k,
        "bets": bets,
        "hits": hits,
        "hit_rate": hit_rate,
        "roi_bets_used": bet_roi,
        "roi": roi,
    }


def eval_ev(df: pd.DataFrame, ev_th: float, max_k: int = 3) -> dict:
    df2 = df[df["odds_mid"].notna() & (df["ev"] >= ev_th)].copy()

    picked = (
        df2.sort_values(["race_id", "ev"], ascending=[True, False])
        .groupby("race_id")
        .head(max_k)
    )

    bets = len(picked)
    hits = int(picked["y_true"].sum())
    hit_rate = hits / bets if bets else 0.0

    ret = float((picked["y_true"] * picked["odds_mid"]).sum())
    roi = ret / bets if bets else float("nan")

    return {
        "ev_th": ev_th,
        "bets": bets,
        "hits": hits,
        "hit_rate": hit_rate,
        "roi": roi,
    }


def main():
    X = pd.read_parquet(FEAT_DIR / "x_plus_hist.parquet")
    meta = pd.read_parquet(FEAT_DIR / "meta.parquet")[["race_id", "horse_number"]]

    truth = load_labels_and_odds_from_db()

    base = meta.merge(truth, on=["race_id", "horse_number"], how="inner")

    X_join = meta.merge(base[["race_id", "horse_number"]], on=["race_id", "horse_number"], how="inner")

    key = meta["race_id"].astype(str) + "_" + meta["horse_number"].astype(str)
    key2 = X_join["race_id"].astype(str) + "_" + X_join["horse_number"].astype(str)
    pos = pd.Series(np.arange(len(meta)), index=key.values)
    X_aligned = X.iloc[pos.loc[key2.values].values].reset_index(drop=True)

    y = base["y_true"].reset_index(drop=True)
    groups = base["race_id"].reset_index(drop=True)

    print("rows used:", len(base), "races:", base["race_id"].nunique(), "positive_rate:", float(y.mean()))

    pred = make_oof_predictions(X_aligned, y, groups, n_splits=5)

    df = base.copy().reset_index(drop=True)
    df["pred"] = pred

    #for k in [1, 2, 3]:
    #    res = eval_topk(df, k)
    #    print(res)

    pred_np = df["pred"].values
    y_np = df["y_true"].values

    df["pred_platt"] = calibrate_platt(pred_np, y_np)
    df["pred_iso"] = calibrate_isotonic(pred_np, y_np)

    for name in ["pred_platt", "pred_iso"]:
        df["ev"] = df[name] * df["odds_mid"]

        print(f"\n=== {name} ===")
        for ev_th in [1.00, 1.05, 1.10, 1.20]:
            res = eval_ev(df, ev_th, max_k=3)
            print(res)

    miss = df["odds_mid"].isna().mean()
    print("odds_mid missing rate:", float(miss))

    print(df["ev"].describe())
    print(df["ev"].quantile([0.9, 0.95, 0.99]))


if __name__ == "__main__":
    main()