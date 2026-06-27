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


def _split_by_start_date(df, test_start_date: str, test_end_date: str | None = None):
    train_df = df[df["date"] < test_start_date].copy()
    if test_end_date is None:
        test_df = df[df["date"] >= test_start_date].copy()
    else:
        test_df = df[(df["date"] >= test_start_date) & (df["date"] < test_end_date)].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(f"Invalid split: start={test_start_date}, end={test_end_date}")

    return train_df, test_df


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


def _threshold_report(
    df,
    score_col: str,
    thresholds: list[float],
    stake: float,
) -> list[dict[str, Any]]:
    results = []
    scored = df.dropna(subset=[score_col, "place_odds_min", "place_odds_max"]).copy()
    scored["place_odds_mid"] = (scored["place_odds_min"] + scored["place_odds_max"]) / 2

    for threshold in thresholds:
        selected = scored[scored[score_col] >= threshold].copy()
        selections = len(selected)
        hits = int(selected["target_top3"].sum())
        stake_total = selections * stake
        return_min = float((selected["target_top3"] * selected["place_odds_min"] * stake).sum())
        return_mid = float((selected["target_top3"] * selected["place_odds_mid"] * stake).sum())
        return_max = float((selected["target_top3"] * selected["place_odds_max"] * stake).sum())
        results.append(
            {
                "score": score_col,
                "threshold": threshold,
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


def _band_report(df, band_col: str, bins: list[float], stake: float) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except ModuleNotFoundError as e:
        raise RuntimeError("帯別分析には pandas が必要です。") from e

    scored = df.dropna(subset=[band_col, "place_odds_min", "place_odds_max"]).copy()
    scored["place_odds_mid"] = (scored["place_odds_min"] + scored["place_odds_max"]) / 2
    scored["band"] = pd.cut(scored[band_col], bins=bins, include_lowest=True, right=False)

    results = []
    for band, selected in scored.groupby("band", observed=True):
        selections = len(selected)
        hits = int(selected["target_top3"].sum())
        stake_total = selections * stake
        return_min = float((selected["target_top3"] * selected["place_odds_min"] * stake).sum())
        return_mid = float((selected["target_top3"] * selected["place_odds_mid"] * stake).sum())
        return_max = float((selected["target_top3"] * selected["place_odds_max"] * stake).sum())
        results.append(
            {
                "band_col": band_col,
                "band": str(band),
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


def _selection_summary(selected, stake: float) -> dict[str, Any]:
    selections = len(selected)
    hits = int(selected["target_top3"].sum())
    stake_total = selections * stake
    return_min = float((selected["target_top3"] * selected["place_odds_min"] * stake).sum())
    return_mid = float((selected["target_top3"] * selected["place_odds_mid"] * stake).sum())
    return_max = float((selected["target_top3"] * selected["place_odds_max"] * stake).sum())
    return {
        "races": int(selected["race_id"].nunique()),
        "selections": selections,
        "hits": hits,
        "hit_rate_pct": None if selections == 0 else hits / selections * 100,
        "return_min_pct": None if stake_total == 0 else return_min / stake_total * 100,
        "return_mid_pct": None if stake_total == 0 else return_mid / stake_total * 100,
        "return_max_pct": None if stake_total == 0 else return_max / stake_total * 100,
    }


def _apply_fixed_rule(
    df,
    pred_min: float,
    odds_min: float,
    odds_max: float,
    distance_min: int | None,
    distance_max: int | None,
    track_id: int | None,
    include_track_ids: list[int] | None,
    exclude_track_ids: list[int] | None,
    surface_id: int | None,
):
    selected = df[
        (df["pred_top3"] >= pred_min)
        & (df["place_odds_mid"] >= odds_min)
        & (df["place_odds_mid"] < odds_max)
    ].copy()
    if distance_min is not None:
        selected = selected[selected["distance"] >= distance_min]
    if distance_max is not None:
        selected = selected[selected["distance"] < distance_max]
    if track_id is not None:
        selected = selected[selected["track_id"] == track_id]
    if include_track_ids is not None:
        selected = selected[selected["track_id"].isin(include_track_ids)]
    if exclude_track_ids is not None:
        selected = selected[~selected["track_id"].isin(exclude_track_ids)]
    if surface_id is not None:
        selected = selected[selected["surface_id"] == surface_id]
    return selected


def _rule_grid_report(
    df,
    pred_thresholds: list[float],
    odds_ranges: list[tuple[float, float]],
    stake: float,
    min_selections: int,
) -> list[dict[str, Any]]:
    scored = df.dropna(subset=["pred_top3", "place_odds_min", "place_odds_max"]).copy()
    scored["place_odds_mid"] = (scored["place_odds_min"] + scored["place_odds_max"]) / 2

    results = []
    for pred_min in pred_thresholds:
        for odds_min, odds_max in odds_ranges:
            selected = scored[
                (scored["pred_top3"] >= pred_min)
                & (scored["place_odds_mid"] >= odds_min)
                & (scored["place_odds_mid"] < odds_max)
            ]
            summary = _selection_summary(selected, stake)
            if summary["selections"] < min_selections:
                continue
            summary.update(
                {
                    "pred_min": pred_min,
                    "odds_min": odds_min,
                    "odds_max": odds_max,
                    "rule_key": f"pred>={pred_min:.2f}|odds=[{odds_min:.1f},{odds_max:.1f})",
                }
            )
            results.append(summary)

    return sorted(
        results,
        key=lambda row: (
            -1 if row["return_mid_pct"] is None else row["return_mid_pct"],
            row["selections"],
        ),
        reverse=True,
    )


def _fixed_rule_breakdown(selected, group_col: str, stake: float) -> list[dict[str, Any]]:
    results = []
    if selected.empty:
        return results
    for value, group in selected.groupby(group_col, observed=True):
        summary = _selection_summary(group, stake)
        summary.update({"group_col": group_col, "group_value": str(value)})
        results.append(summary)
    return sorted(results, key=lambda row: row["return_mid_pct"], reverse=True)


def _rule_condition_report(
    df,
    pred_thresholds: list[float],
    odds_ranges: list[tuple[float, float]],
    stake: float,
    min_selections: int,
) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except ModuleNotFoundError as e:
        raise RuntimeError("条件別ルール分析には pandas が必要です。") from e

    scored = df.dropna(subset=["pred_top3", "place_odds_min", "place_odds_max"]).copy()
    scored["place_odds_mid"] = (scored["place_odds_min"] + scored["place_odds_max"]) / 2
    scored["distance_band"] = pd.cut(
        scored["distance"],
        bins=[0, 1400, 1800, 2200, 10000],
        labels=["under1400", "1400-1799", "1800-2199", "2200plus"],
        include_lowest=True,
        right=False,
    )
    scored["race_size_band"] = pd.cut(
        scored["race_size"],
        bins=[0, 10, 14, 19],
        labels=["small", "medium", "large"],
        include_lowest=True,
        right=False,
    )

    condition_cols = ["track_id", "surface_id", "distance_band", "race_size_band"]
    results = []
    for pred_min in pred_thresholds:
        for odds_min, odds_max in odds_ranges:
            selected = scored[
                (scored["pred_top3"] >= pred_min)
                & (scored["place_odds_mid"] >= odds_min)
                & (scored["place_odds_mid"] < odds_max)
            ]
            rule_key = f"pred>={pred_min:.2f}|odds=[{odds_min:.1f},{odds_max:.1f})"
            for condition_col in condition_cols:
                for condition_value, group in selected.groupby(condition_col, observed=True):
                    summary = _selection_summary(group, stake)
                    if summary["selections"] < min_selections:
                        continue
                    summary.update(
                        {
                            "rule_key": rule_key,
                            "condition_col": condition_col,
                            "condition_value": str(condition_value),
                        }
                    )
                    results.append(summary)

    return results


def _fit_and_evaluate_split(
    train_df,
    test_df,
    late_df,
    stake: float,
    pred_thresholds: list[float] | None,
    expected_value_thresholds: list[float] | None,
    min_rule_selections: int,
) -> dict[str, Any]:
    try:
        from lightgbm import LGBMClassifier
        from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    except ModuleNotFoundError as e:
        raise RuntimeError("LightGBM評価には lightgbm と scikit-learn が必要です。") from e

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

    eval_df = test_df[
        [
            "race_id",
            "date",
            "horse_number",
            "target_top3",
            "track_id",
            "surface_id",
            "distance",
            "race_size",
        ]
    ].copy()
    eval_df["pred_top3"] = pred

    odds_cols = ["race_id", "horse_number", "place_odds_min", "place_odds_max"]
    eval_df = eval_df.merge(late_df[odds_cols], on=["race_id", "horse_number"], how="left")
    eval_df["expected_value_min"] = eval_df["pred_top3"] * eval_df["place_odds_min"]
    eval_df["expected_value_mid"] = eval_df["pred_top3"] * (
        (eval_df["place_odds_min"] + eval_df["place_odds_max"]) / 2
    )
    eval_df["place_odds_mid"] = (eval_df["place_odds_min"] + eval_df["place_odds_max"]) / 2

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

    if pred_thresholds is None:
        pred_thresholds = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    if expected_value_thresholds is None:
        expected_value_thresholds = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3]

    threshold_results = []
    threshold_results.extend(_threshold_report(eval_df, "pred_top3", pred_thresholds, stake=stake))
    threshold_results.extend(_threshold_report(eval_df, "expected_value_min", expected_value_thresholds, stake=stake))
    threshold_results.extend(_threshold_report(eval_df, "expected_value_mid", expected_value_thresholds, stake=stake))

    band_results = []
    band_results.extend(_band_report(eval_df, "pred_top3", [0, 0.2, 0.3, 0.4, 0.5, 0.6, 1.01], stake=stake))
    band_results.extend(_band_report(eval_df, "place_odds_mid", [0, 1.5, 2.0, 3.0, 5.0, 10.0, 1000.0], stake=stake))
    band_results.extend(_band_report(eval_df, "expected_value_mid", [0, 0.7, 0.9, 1.0, 1.1, 1.3, 1000.0], stake=stake))

    rule_grid_results = _rule_grid_report(
        eval_df,
        pred_thresholds=[0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6],
        odds_ranges=[(1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 10.0), (10.0, 1000.0)],
        stake=stake,
        min_selections=min_rule_selections,
    )
    rule_condition_results = _rule_condition_report(
        eval_df,
        pred_thresholds=[0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6],
        odds_ranges=[(1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 10.0), (10.0, 1000.0)],
        stake=stake,
        min_selections=min_rule_selections,
    )

    importance = sorted(
        zip(feature_cols, model.feature_importances_),
        key=lambda x: int(x[1]),
        reverse=True,
    )

    return {
        "features": feature_cols,
        "categorical_features": categorical_cols,
        "metrics": metric_report,
        "eval_df": eval_df,
        "strategies": strategy_results,
        "thresholds": threshold_results,
        "bands": band_results,
        "rule_grid": rule_grid_results,
        "rule_conditions": rule_condition_results,
        "feature_importance": importance[:15],
    }


def evaluate_place_top3_lgbm(
    early_dataset_path: Path,
    late_dataset_path: Path,
    engine: str = "auto",
    test_ratio: float = 0.2,
    stake: float = 100.0,
    pred_thresholds: list[float] | None = None,
    expected_value_thresholds: list[float] | None = None,
    min_rule_selections: int = 100,
) -> dict[str, Any]:
    try:
        from sklearn.metrics import roc_auc_score
    except ModuleNotFoundError as e:
        raise RuntimeError("LightGBM評価には scikit-learn が必要です。") from e

    early_df = _read_parquet(early_dataset_path, engine)
    late_df = _read_parquet(late_dataset_path, engine)

    train_df, test_df, split_date = _split_by_date(early_df, test_ratio)
    split_report = _fit_and_evaluate_split(
        train_df=train_df,
        test_df=test_df,
        late_df=late_df,
        stake=stake,
        pred_thresholds=pred_thresholds,
        expected_value_thresholds=expected_value_thresholds,
        min_rule_selections=min_rule_selections,
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
        "features": split_report["features"],
        "categorical_features": split_report["categorical_features"],
        "metrics": split_report["metrics"],
        "strategies": split_report["strategies"],
        "thresholds": split_report["thresholds"],
        "bands": split_report["bands"],
        "rule_grid": split_report["rule_grid"][:20],
        "min_rule_selections": min_rule_selections,
        "feature_importance": split_report["feature_importance"],
    }


def _make_walk_forward_splits(dates: list[str], n_splits: int, min_train_ratio: float) -> list[tuple[str, str | None]]:
    if n_splits < 1:
        raise ValueError("n_splits must be at least 1")
    if not 0 < min_train_ratio < 1:
        raise ValueError("min_train_ratio must be between 0 and 1")

    min_idx = int(len(dates) * min_train_ratio)
    if min_idx >= len(dates) - 1:
        raise ValueError("Not enough dates after min_train_ratio")

    remaining = len(dates) - min_idx
    step = max(1, remaining // n_splits)

    splits = []
    for i in range(n_splits):
        start_idx = min_idx + i * step
        if start_idx >= len(dates) - 1:
            break
        end_idx = min_idx + (i + 1) * step
        test_start = dates[start_idx]
        test_end = dates[end_idx] if i < n_splits - 1 and end_idx < len(dates) else None
        splits.append((test_start, test_end))

    return splits


def evaluate_place_top3_lgbm_walk_forward(
    early_dataset_path: Path,
    late_dataset_path: Path,
    engine: str = "auto",
    n_splits: int = 4,
    min_train_ratio: float = 0.5,
    stake: float = 100.0,
    min_rule_selections: int = 30,
) -> dict[str, Any]:
    early_df = _read_parquet(early_dataset_path, engine)
    late_df = _read_parquet(late_dataset_path, engine)

    dates = sorted(early_df["date"].dropna().unique())
    splits = _make_walk_forward_splits(dates, n_splits=n_splits, min_train_ratio=min_train_ratio)

    fold_reports = []
    rule_rows = []
    for fold_idx, (test_start, test_end) in enumerate(splits, start=1):
        train_df, test_df = _split_by_start_date(early_df, test_start, test_end)
        split_report = _fit_and_evaluate_split(
            train_df=train_df,
            test_df=test_df,
            late_df=late_df,
            stake=stake,
            pred_thresholds=None,
            expected_value_thresholds=None,
            min_rule_selections=min_rule_selections,
        )

        fold_summary = {
            "fold": fold_idx,
            "test_start": test_start,
            "test_end": test_end,
            "train_rows": len(train_df),
            "train_races": int(train_df["race_id"].nunique()),
            "test_rows": len(test_df),
            "test_races": int(test_df["race_id"].nunique()),
            "auc": split_report["metrics"]["auc"],
            "logloss": split_report["metrics"]["logloss"],
            "brier": split_report["metrics"]["brier"],
        }
        fold_reports.append(fold_summary)

        for row in split_report["rule_grid"]:
            rule_row = dict(row)
            rule_row["fold"] = fold_idx
            rule_row["test_start"] = test_start
            rule_row["test_end"] = test_end
            rule_rows.append(rule_row)

        for row in split_report["rule_conditions"]:
            condition_row = dict(row)
            condition_row["fold"] = fold_idx
            condition_row["test_start"] = test_start
            condition_row["test_end"] = test_end
            rule_rows.append(condition_row)

    rule_summary = _summarize_rules(rule_rows, n_folds=len(fold_reports))
    condition_summary = _summarize_condition_rules(rule_rows, n_folds=len(fold_reports))

    return {
        "early_dataset_path": str(early_dataset_path),
        "late_dataset_path": str(late_dataset_path),
        "rows": len(early_df),
        "races": int(early_df["race_id"].nunique()),
        "n_splits": len(fold_reports),
        "min_train_ratio": min_train_ratio,
        "min_rule_selections": min_rule_selections,
        "folds": fold_reports,
        "rule_summary": rule_summary[:20],
        "condition_summary": condition_summary[:20],
    }


def evaluate_fixed_place_top3_rule_walk_forward(
    early_dataset_path: Path,
    late_dataset_path: Path,
    engine: str = "auto",
    n_splits: int = 4,
    min_train_ratio: float = 0.5,
    stake: float = 100.0,
    pred_min: float = 0.35,
    odds_min: float = 3.0,
    odds_max: float = 5.0,
    distance_min: int | None = 1800,
    distance_max: int | None = 2200,
    track_id: int | None = None,
    include_track_ids: list[int] | None = None,
    exclude_track_ids: list[int] | None = None,
    surface_id: int | None = None,
) -> dict[str, Any]:
    early_df = _read_parquet(early_dataset_path, engine)
    late_df = _read_parquet(late_dataset_path, engine)

    dates = sorted(early_df["date"].dropna().unique())
    splits = _make_walk_forward_splits(dates, n_splits=n_splits, min_train_ratio=min_train_ratio)

    fold_reports = []
    selected_rows = []
    for fold_idx, (test_start, test_end) in enumerate(splits, start=1):
        train_df, test_df = _split_by_start_date(early_df, test_start, test_end)
        split_report = _fit_and_evaluate_split(
            train_df=train_df,
            test_df=test_df,
            late_df=late_df,
            stake=stake,
            pred_thresholds=None,
            expected_value_thresholds=None,
            min_rule_selections=1,
        )
        selected = _apply_fixed_rule(
            split_report["eval_df"],
            pred_min=pred_min,
            odds_min=odds_min,
            odds_max=odds_max,
            distance_min=distance_min,
            distance_max=distance_max,
            track_id=track_id,
            include_track_ids=include_track_ids,
            exclude_track_ids=exclude_track_ids,
            surface_id=surface_id,
        )
        selected = selected.copy()
        selected["fold"] = fold_idx
        selected["month"] = selected["date"].str.slice(0, 7)
        selected_rows.append(selected)

        summary = _selection_summary(selected, stake)
        summary.update(
            {
                "fold": fold_idx,
                "test_start": test_start,
                "test_end": test_end,
                "auc": split_report["metrics"]["auc"],
                "logloss": split_report["metrics"]["logloss"],
                "brier": split_report["metrics"]["brier"],
            }
        )
        fold_reports.append(summary)

    try:
        import pandas as pd
    except ModuleNotFoundError as e:
        raise RuntimeError("固定ルール評価には pandas が必要です。") from e

    all_selected = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()
    overall = _selection_summary(all_selected, stake)

    return {
        "early_dataset_path": str(early_dataset_path),
        "late_dataset_path": str(late_dataset_path),
        "rows": len(early_df),
        "races": int(early_df["race_id"].nunique()),
        "n_splits": len(fold_reports),
        "min_train_ratio": min_train_ratio,
        "rule": {
            "pred_min": pred_min,
            "odds_min": odds_min,
            "odds_max": odds_max,
            "distance_min": distance_min,
            "distance_max": distance_max,
            "track_id": track_id,
            "include_track_ids": include_track_ids,
            "exclude_track_ids": exclude_track_ids,
            "surface_id": surface_id,
        },
        "overall": overall,
        "folds": fold_reports,
        "by_month": _fixed_rule_breakdown(all_selected, "month", stake),
        "by_track": _fixed_rule_breakdown(all_selected, "track_id", stake),
        "by_surface": _fixed_rule_breakdown(all_selected, "surface_id", stake),
    }


def format_fixed_rule_report(report: dict[str, Any]) -> str:
    rule = report["rule"]
    overall = report["overall"]
    lines = [
        f"Early dataset: {report['early_dataset_path']}",
        f"Late odds dataset: {report['late_dataset_path']}",
        f"Rows: {report['rows']:,}",
        f"Races: {report['races']:,}",
        f"Folds: {report['n_splits']}",
        (
            "Rule: "
            f"pred_top3>={rule['pred_min']:.2f}, "
            f"odds_mid=[{rule['odds_min']:.1f},{rule['odds_max']:.1f}), "
            f"distance=[{rule['distance_min']},{rule['distance_max']}), "
            f"track_id={rule['track_id']}, "
            f"include_track_ids={rule['include_track_ids']}, "
            f"exclude_track_ids={rule['exclude_track_ids']}, "
            f"surface_id={rule['surface_id']}"
        ),
        "",
        "Overall",
        "selections  hits  hit_rate  return_min  return_mid  return_max",
        (
            f"{overall['selections']:>10,}  {overall['hits']:>4,}  "
            f"{overall['hit_rate_pct']:>7.2f}%  {overall['return_min_pct']:>9.2f}%  "
            f"{overall['return_mid_pct']:>9.2f}%  {overall['return_max_pct']:>9.2f}%"
        ),
        "",
        "Fold results",
        "fold  test_start  test_end    selections  hits  hit_rate  return_min  return_mid  return_max",
    ]
    for row in report["folds"]:
        test_end = row["test_end"] or "end"
        lines.append(
            f"{row['fold']:>4}  {row['test_start']}  {test_end:<10}  "
            f"{row['selections']:>10,}  {row['hits']:>4,}  "
            f"{row['hit_rate_pct']:>7.2f}%  {row['return_min_pct']:>9.2f}%  "
            f"{row['return_mid_pct']:>9.2f}%  {row['return_max_pct']:>9.2f}%"
        )

    for section_name, key in [("Monthly", "by_month"), ("Track", "by_track"), ("Surface", "by_surface")]:
        lines.extend(["", f"{section_name} breakdown", "value       selections  hits  hit_rate  return_min  return_mid  return_max"])
        for row in report[key]:
            lines.append(
                f"{row['group_value']:<10}  {row['selections']:>10,}  {row['hits']:>4,}  "
                f"{row['hit_rate_pct']:>7.2f}%  {row['return_min_pct']:>9.2f}%  "
                f"{row['return_mid_pct']:>9.2f}%  {row['return_max_pct']:>9.2f}%"
            )

    return "\n".join(lines)


def _summarize_rules(rule_rows: list[dict[str, Any]], n_folds: int) -> list[dict[str, Any]]:
    rule_rows = [row for row in rule_rows if "condition_col" not in row]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rule_rows:
        grouped.setdefault(row["rule_key"], []).append(row)

    summaries = []
    for rule_key, rows in grouped.items():
        selections = sum(row["selections"] for row in rows)
        hits = sum(row["hits"] for row in rows)
        stake_proxy = selections
        # Percent returns are weighted by selections, equivalent to aggregating stake-normalized returns.
        return_min = sum(row["return_min_pct"] * row["selections"] for row in rows) / stake_proxy
        return_mid = sum(row["return_mid_pct"] * row["selections"] for row in rows) / stake_proxy
        return_max = sum(row["return_max_pct"] * row["selections"] for row in rows) / stake_proxy
        summaries.append(
            {
                "rule_key": rule_key,
                "folds_hit": len(rows),
                "folds_total": n_folds,
                "selections": selections,
                "hits": hits,
                "hit_rate_pct": None if selections == 0 else hits / selections * 100,
                "return_min_pct": return_min,
                "return_mid_pct": return_mid,
                "return_max_pct": return_max,
                "min_fold_return_mid_pct": min(row["return_mid_pct"] for row in rows),
                "max_fold_return_mid_pct": max(row["return_mid_pct"] for row in rows),
            }
        )

    return sorted(
        summaries,
        key=lambda row: (
            row["folds_hit"] == row["folds_total"],
            row["return_mid_pct"],
            row["selections"],
        ),
        reverse=True,
    )


def _summarize_condition_rules(rule_rows: list[dict[str, Any]], n_folds: int) -> list[dict[str, Any]]:
    condition_rows = [row for row in rule_rows if "condition_col" in row]
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in condition_rows:
        key = (row["rule_key"], row["condition_col"], row["condition_value"])
        grouped.setdefault(key, []).append(row)

    summaries = []
    for (rule_key, condition_col, condition_value), rows in grouped.items():
        selections = sum(row["selections"] for row in rows)
        hits = sum(row["hits"] for row in rows)
        stake_proxy = selections
        return_min = sum(row["return_min_pct"] * row["selections"] for row in rows) / stake_proxy
        return_mid = sum(row["return_mid_pct"] * row["selections"] for row in rows) / stake_proxy
        return_max = sum(row["return_max_pct"] * row["selections"] for row in rows) / stake_proxy
        summaries.append(
            {
                "rule_key": rule_key,
                "condition_col": condition_col,
                "condition_value": condition_value,
                "folds_hit": len(rows),
                "folds_total": n_folds,
                "selections": selections,
                "hits": hits,
                "hit_rate_pct": None if selections == 0 else hits / selections * 100,
                "return_min_pct": return_min,
                "return_mid_pct": return_mid,
                "return_max_pct": return_max,
                "min_fold_return_mid_pct": min(row["return_mid_pct"] for row in rows),
                "max_fold_return_mid_pct": max(row["return_mid_pct"] for row in rows),
            }
        )

    return sorted(
        summaries,
        key=lambda row: (
            row["folds_hit"] == row["folds_total"],
            row["return_mid_pct"],
            row["selections"],
        ),
        reverse=True,
    )


def format_walk_forward_report(report: dict[str, Any]) -> str:
    lines = [
        f"Early dataset: {report['early_dataset_path']}",
        f"Late odds dataset: {report['late_dataset_path']}",
        f"Rows: {report['rows']:,}",
        f"Races: {report['races']:,}",
        f"Folds: {report['n_splits']}",
        f"Min train ratio: {report['min_train_ratio']:.2f}",
        "",
        "Fold metrics",
        "fold  test_start  test_end    train_rows  test_rows  test_races      AUC  logloss    brier",
    ]

    for row in report["folds"]:
        test_end = row["test_end"] or "end"
        lines.append(
            f"{row['fold']:>4}  {row['test_start']}  {test_end:<10}  "
            f"{row['train_rows']:>10,}  {row['test_rows']:>9,}  {row['test_races']:>10,}  "
            f"{row['auc']:>7.5f}  {row['logloss']:>7.5f}  {row['brier']:>7.5f}"
        )

    lines.extend(
        [
            "",
            f"Rule stability summary (min selections per fold: {report['min_rule_selections']})",
            "rule_key                       folds  selections  hits  hit_rate  return_min  return_mid  return_max  min_mid  max_mid",
        ]
    )
    for row in report["rule_summary"]:
        lines.append(
            f"{row['rule_key']:<30} {row['folds_hit']:>2}/{row['folds_total']:<2}  "
            f"{row['selections']:>10,}  {row['hits']:>4,}  "
            f"{row['hit_rate_pct']:>7.2f}%  {row['return_min_pct']:>9.2f}%  "
            f"{row['return_mid_pct']:>9.2f}%  {row['return_max_pct']:>9.2f}%  "
            f"{row['min_fold_return_mid_pct']:>7.2f}%  {row['max_fold_return_mid_pct']:>7.2f}%"
        )

    lines.extend(
        [
            "",
            f"Condition rule summary (min selections per fold: {report['min_rule_selections']})",
            "rule_key                       condition        folds  selections  hits  hit_rate  return_min  return_mid  return_max  min_mid  max_mid",
        ]
    )
    for row in report["condition_summary"]:
        condition = f"{row['condition_col']}={row['condition_value']}"
        lines.append(
            f"{row['rule_key']:<30} {condition:<16} {row['folds_hit']:>2}/{row['folds_total']:<2}  "
            f"{row['selections']:>10,}  {row['hits']:>4,}  "
            f"{row['hit_rate_pct']:>7.2f}%  {row['return_min_pct']:>9.2f}%  "
            f"{row['return_mid_pct']:>9.2f}%  {row['return_max_pct']:>9.2f}%  "
            f"{row['min_fold_return_mid_pct']:>7.2f}%  {row['max_fold_return_mid_pct']:>7.2f}%"
        )

    return "\n".join(lines)


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

    lines.extend(
        [
            "",
            "Threshold results on test period",
            "score              threshold  races  selections  hits  hit_rate  return_min  return_mid  return_max",
        ]
    )
    for row in report["thresholds"]:
        hit_rate = row["hit_rate_pct"]
        min_pct = row["return_min_pct"]
        mid_pct = row["return_mid_pct"]
        max_pct = row["return_max_pct"]
        lines.append(
            f"{row['score']:<19} {row['threshold']:>9.2f}  {row['races']:>5,}  "
            f"{row['selections']:>10,}  {row['hits']:>4,}  "
            f"{'n/a' if hit_rate is None else f'{hit_rate:.2f}%':>8}  "
            f"{'n/a' if min_pct is None else f'{min_pct:.2f}%':>9}  "
            f"{'n/a' if mid_pct is None else f'{mid_pct:.2f}%':>9}  "
            f"{'n/a' if max_pct is None else f'{max_pct:.2f}%':>9}"
        )

    lines.extend(
        [
            "",
            "Band results on test period",
            "band_col           band             races  selections  hits  hit_rate  return_min  return_mid  return_max",
        ]
    )
    for row in report["bands"]:
        hit_rate = row["hit_rate_pct"]
        min_pct = row["return_min_pct"]
        mid_pct = row["return_mid_pct"]
        max_pct = row["return_max_pct"]
        lines.append(
            f"{row['band_col']:<18} {row['band']:<16} {row['races']:>5,}  "
            f"{row['selections']:>10,}  {row['hits']:>4,}  "
            f"{'n/a' if hit_rate is None else f'{hit_rate:.2f}%':>8}  "
            f"{'n/a' if min_pct is None else f'{min_pct:.2f}%':>9}  "
            f"{'n/a' if mid_pct is None else f'{mid_pct:.2f}%':>9}  "
            f"{'n/a' if max_pct is None else f'{max_pct:.2f}%':>9}"
        )

    lines.extend(
        [
            "",
            f"Top rule grid results on test period (min selections: {report['min_rule_selections']})",
            "pred_min  odds_mid_range   races  selections  hits  hit_rate  return_min  return_mid  return_max",
        ]
    )
    for row in report["rule_grid"]:
        odds_range = f"[{row['odds_min']:.1f}, {row['odds_max']:.1f})"
        hit_rate = row["hit_rate_pct"]
        min_pct = row["return_min_pct"]
        mid_pct = row["return_mid_pct"]
        max_pct = row["return_max_pct"]
        lines.append(
            f"{row['pred_min']:>8.2f}  {odds_range:<15} {row['races']:>5,}  "
            f"{row['selections']:>10,}  {row['hits']:>4,}  "
            f"{'n/a' if hit_rate is None else f'{hit_rate:.2f}%':>8}  "
            f"{'n/a' if min_pct is None else f'{min_pct:.2f}%':>9}  "
            f"{'n/a' if mid_pct is None else f'{mid_pct:.2f}%':>9}  "
            f"{'n/a' if max_pct is None else f'{max_pct:.2f}%':>9}"
        )

    lines.extend(["", "Top feature importance"])
    for name, value in report["feature_importance"]:
        lines.append(f"  {name}: {value}")

    return "\n".join(lines)
