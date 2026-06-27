from pathlib import Path
from typing import Any


ID_COLUMNS = ["horse_id", "jockey_id", "trainer_id"]
EXCLUDE_FEATURE_COLUMNS = {"race_id", "date", "horse_name", "target_top3"}


def _read_parquet(path: Path, engine: str):
    try:
        import pandas as pd
    except ModuleNotFoundError as e:
        raise RuntimeError("LightGBM評価には pandas が必要です。") from e

    return pd.read_parquet(path, engine=engine)


def _split_by_date(df, test_ratio: float):
    dates = sorted(df["date"].dropna().unique())
    if len(dates) < 2:
        raise ValueError("Dataset needs at least two dates for time split")

    split_idx = int(len(dates) * (1 - test_ratio))
    split_idx = min(max(split_idx, 1), len(dates) - 1)
    split_date = dates[split_idx]

    train_df = df[df["date"] < split_date].copy()
    test_df = df[df["date"] >= split_date].copy()
    return train_df, test_df, split_date


def _prepare_features(train_df, test_df):
    feature_cols = [c for c in train_df.columns if c not in EXCLUDE_FEATURE_COLUMNS]

    train_x = train_df[feature_cols].copy()
    test_x = test_df[feature_cols].copy()

    categorical_cols = [c for c in ID_COLUMNS if c in feature_cols]
    for col in categorical_cols:
        categories = train_x[col].fillna("__MISSING__").astype("string").unique()
        train_x[col] = train_x[col].fillna("__MISSING__").astype("category").cat.set_categories(categories)
        test_x[col] = test_x[col].fillna("__MISSING__").astype("category").cat.set_categories(categories)

    return train_x, test_x, feature_cols, categorical_cols


def _strategy_report(df, score_col: str, ascending: bool, stake: float, top_k_values: list[int]) -> list[dict[str, Any]]:
    results = []
    scored = df.dropna(subset=[score_col, "place_odds_min", "place_odds_max"]).copy()
    scored["place_odds_mid"] = (scored["place_odds_min"] + scored["place_odds_max"]) / 2

    for top_k in top_k_values:
        selected = (
            scored.sort_values(["race_id", score_col, "horse_number"], ascending=[True, ascending, True])
            .groupby("race_id", as_index=False)
            .head(top_k)
        )
        selections = len(selected)
        hits = int(selected["target_top3"].sum())
        stake_total = selections * stake
        return_min = float((selected["target_top3"] * selected["place_odds_min"] * stake).sum())
        return_mid = float((selected["target_top3"] * selected["place_odds_mid"] * stake).sum())
        return_max = float((selected["target_top3"] * selected["place_odds_max"] * stake).sum())
        results.append(
            {
                "strategy": score_col,
                "top_k": top_k,
                "races": int(selected["race_id"].nunique()),
                "selections": selections,
                "hits": hits,
                "hit_rate_pct": None if selections == 0 else hits / selections * 100,
                "return_min_pct": None if stake_total == 0 else return_min / stake_total * 100,
                "return_mid_pct": None if stake_total == 0 else return_mid / stake_total * 100,
                "return_max_pct": None if stake_total == 0 else return_max / stake_total * 100,
            }
        )

    return results


def evaluate_place_top3_lgbm(
    early_dataset_path: Path,
    late_dataset_path: Path,
    engine: str = "auto",
    test_ratio: float = 0.2,
    stake: float = 100.0,
) -> dict[str, Any]:
    try:
        from lightgbm import LGBMClassifier
        from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    except ModuleNotFoundError as e:
        raise RuntimeError("LightGBM評価には lightgbm と scikit-learn が必要です。") from e

    early_df = _read_parquet(early_dataset_path, engine)
    late_df = _read_parquet(late_dataset_path, engine)

    train_df, test_df, split_date = _split_by_date(early_df, test_ratio)
    train_x, test_x, feature_cols, categorical_cols = _prepare_features(train_df, test_df)
    train_y = train_df["target_top3"]
    test_y = test_df["target_top3"]

    model = LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=80,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(train_x, train_y, categorical_feature=categorical_cols)

    pred = model.predict_proba(test_x)[:, 1]

    eval_df = test_df[["race_id", "date", "horse_number", "target_top3"]].copy()
    eval_df["pred_top3"] = pred

    odds_cols = ["race_id", "horse_number", "place_odds_min", "place_odds_max"]
    eval_df = eval_df.merge(late_df[odds_cols], on=["race_id", "horse_number"], how="left")
    eval_df["expected_value_min"] = eval_df["pred_top3"] * eval_df["place_odds_min"]
    eval_df["expected_value_mid"] = eval_df["pred_top3"] * (
        (eval_df["place_odds_min"] + eval_df["place_odds_max"]) / 2
    )

    metric_report = {
        "auc": float(roc_auc_score(test_y, pred)),
        "logloss": float(log_loss(test_y, pred)),
        "brier": float(brier_score_loss(test_y, pred)),
    }

    strategy_results = []
    strategy_results.extend(_strategy_report(eval_df, "pred_top3", ascending=False, stake=stake, top_k_values=[1, 2, 3]))
    strategy_results.extend(
        _strategy_report(eval_df, "expected_value_min", ascending=False, stake=stake, top_k_values=[1, 2, 3])
    )
    strategy_results.extend(
        _strategy_report(eval_df, "expected_value_mid", ascending=False, stake=stake, top_k_values=[1, 2, 3])
    )

    importance = sorted(
        zip(feature_cols, model.feature_importances_),
        key=lambda x: int(x[1]),
        reverse=True,
    )

    return {
        "early_dataset_path": str(early_dataset_path),
        "late_dataset_path": str(late_dataset_path),
        "rows": len(early_df),
        "races": int(early_df["race_id"].nunique()),
        "train_rows": len(train_df),
        "train_races": int(train_df["race_id"].nunique()),
        "test_rows": len(test_df),
        "test_races": int(test_df["race_id"].nunique()),
        "split_date": split_date,
        "features": feature_cols,
        "categorical_features": categorical_cols,
        "metrics": metric_report,
        "strategies": strategy_results,
        "feature_importance": importance[:15],
    }


def format_lgbm_report(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    lines = [
        f"Early dataset: {report['early_dataset_path']}",
        f"Late odds dataset: {report['late_dataset_path']}",
        f"Rows: {report['rows']:,}",
        f"Races: {report['races']:,}",
        f"Split date: {report['split_date']}",
        f"Train: {report['train_rows']:,} rows / {report['train_races']:,} races",
        f"Test: {report['test_rows']:,} rows / {report['test_races']:,} races",
        f"Features: {len(report['features'])} ({', '.join(report['features'])})",
        "",
        "Metrics",
        f"  AUC: {metrics['auc']:.5f}",
        f"  logloss: {metrics['logloss']:.5f}",
        f"  brier: {metrics['brier']:.5f}",
        "",
        "Strategy results on test period",
        "strategy            top_k  races  selections  hits  hit_rate  return_min  return_mid  return_max",
    ]

    for row in report["strategies"]:
        lines.append(
            f"{row['strategy']:<19} {row['top_k']:>5}  {row['races']:>5,}  "
            f"{row['selections']:>10,}  {row['hits']:>4,}  "
            f"{row['hit_rate_pct']:>7.2f}%  {row['return_min_pct']:>9.2f}%  "
            f"{row['return_mid_pct']:>9.2f}%  {row['return_max_pct']:>9.2f}%"
        )

    lines.extend(["", "Top feature importance"])
    for name, value in report["feature_importance"]:
        lines.append(f"  {name}: {value}")

    return "\n".join(lines)
