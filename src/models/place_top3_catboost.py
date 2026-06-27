from pathlib import Path
from typing import Any

from src.models.place_top3_lgbm import (
    EXCLUDE_FEATURE_COLUMNS,
    _apply_fixed_rule,
    _fixed_rule_breakdown,
    _make_walk_forward_splits,
    _read_parquet,
    _rule_condition_report,
    _rule_grid_report,
    _selection_summary,
    _split_by_start_date,
    format_fixed_rule_report,
    format_walk_forward_report,
    _summarize_condition_rules,
    _summarize_rules,
)


CATBOOST_CATEGORICAL_COLUMNS = [
    "track_id",
    "surface_id",
    "course_direction_id",
    "course_layout_id",
    "course_variant_id",
    "weather_id",
    "track_condition_id",
    "gate",
    "horse_number",
    "horse_id",
    "sex_id",
    "jockey_id",
    "stable_id",
    "trainer_id",
]


def _drop_feature_patterns(df, patterns: list[str] | None):
    if not patterns:
        return df, []
    protected = set(EXCLUDE_FEATURE_COLUMNS)
    drop_cols = [
        col
        for col in df.columns
        if col not in protected and any(pattern in col for pattern in patterns)
    ]
    return df.drop(columns=drop_cols), drop_cols


def _filter_by_surface(df, surface_id: int | None):
    if surface_id is None:
        return df
    return df[df["surface_id"] == surface_id]


def _prepare_catboost_features(train_df, test_df):
    feature_cols = [c for c in train_df.columns if c not in EXCLUDE_FEATURE_COLUMNS]
    train_x = train_df[feature_cols].copy()
    test_x = test_df[feature_cols].copy()

    categorical_cols = [col for col in CATBOOST_CATEGORICAL_COLUMNS if col in feature_cols]
    for col in categorical_cols:
        train_x[col] = train_x[col].fillna("__MISSING__").astype("string")
        test_x[col] = test_x[col].fillna("__MISSING__").astype("string")

    cat_feature_indices = [feature_cols.index(col) for col in categorical_cols]
    return train_x, test_x, feature_cols, categorical_cols, cat_feature_indices


def _fit_and_evaluate_catboost_split(
    train_df,
    test_df,
    odds_df,
    stake: float,
    min_rule_selections: int,
) -> dict[str, Any]:
    try:
        from catboost import CatBoostClassifier
        from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "CatBoost評価には catboost と scikit-learn が必要です。"
            " PowerShellで `.\\.venv\\Scripts\\python.exe -m pip install catboost` を実行してください。"
        ) from e

    train_x, test_x, feature_cols, categorical_cols, cat_feature_indices = _prepare_catboost_features(train_df, test_df)
    train_y = train_df["target_top3"]
    test_y = test_df["target_top3"]

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=500,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=5.0,
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(train_x, train_y, cat_features=cat_feature_indices)
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
    eval_df = eval_df.merge(odds_df[odds_cols], on=["race_id", "horse_number"], how="left")
    eval_df["place_odds_mid"] = (eval_df["place_odds_min"] + eval_df["place_odds_max"]) / 2

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

    return {
        "features": feature_cols,
        "categorical_features": categorical_cols,
        "feature_importance": [
            {"feature": feature, "importance": float(importance)}
            for feature, importance in zip(feature_cols, model.get_feature_importance())
        ],
        "metrics": {
            "auc": float(roc_auc_score(test_y, pred)),
            "logloss": float(log_loss(test_y, pred)),
            "brier": float(brier_score_loss(test_y, pred)),
        },
        "eval_df": eval_df,
        "rule_grid": rule_grid_results,
        "rule_conditions": rule_condition_results,
    }


def evaluate_place_top3_catboost_walk_forward(
    training_dataset_path: Path,
    odds_dataset_path: Path,
    engine: str = "auto",
    n_splits: int = 4,
    min_train_ratio: float = 0.5,
    stake: float = 100.0,
    min_rule_selections: int = 30,
) -> dict[str, Any]:
    training_df = _read_parquet(training_dataset_path, engine)
    odds_df = _read_parquet(odds_dataset_path, engine)

    dates = sorted(training_df["date"].dropna().unique())
    splits = _make_walk_forward_splits(dates, n_splits=n_splits, min_train_ratio=min_train_ratio)

    fold_reports = []
    rule_rows = []
    categorical_features = []
    for fold_idx, (test_start, test_end) in enumerate(splits, start=1):
        train_df, test_df = _split_by_start_date(training_df, test_start, test_end)
        split_report = _fit_and_evaluate_catboost_split(
            train_df=train_df,
            test_df=test_df,
            odds_df=odds_df,
            stake=stake,
            min_rule_selections=min_rule_selections,
        )
        categorical_features = split_report["categorical_features"]

        fold_reports.append(
            {
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
        )

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

    return {
        "training_dataset_path": str(training_dataset_path),
        "odds_dataset_path": str(odds_dataset_path),
        "rows": len(training_df),
        "races": int(training_df["race_id"].nunique()),
        "n_splits": len(fold_reports),
        "min_train_ratio": min_train_ratio,
        "min_rule_selections": min_rule_selections,
        "folds": fold_reports,
        "rule_summary": _summarize_rules(rule_rows, n_folds=len(fold_reports))[:20],
        "condition_summary": _summarize_condition_rules(rule_rows, n_folds=len(fold_reports))[:20],
        "categorical_features": categorical_features,
    }


def evaluate_fixed_place_top3_catboost_rule_walk_forward(
    training_dataset_path: Path,
    odds_dataset_path: Path,
    engine: str = "auto",
    n_splits: int = 4,
    min_train_ratio: float = 0.5,
    stake: float = 100.0,
    pred_min: float = 0.40,
    odds_min: float = 3.0,
    odds_max: float = 5.0,
    distance_min: int | None = 1800,
    distance_max: int | None = 2200,
    track_id: int | None = None,
    include_track_ids: list[int] | None = None,
    exclude_track_ids: list[int] | None = None,
    surface_id: int | None = None,
    drop_feature_patterns: list[str] | None = None,
    train_surface_id: int | None = None,
) -> dict[str, Any]:
    training_df = _read_parquet(training_dataset_path, engine)
    training_df, dropped_features = _drop_feature_patterns(training_df, drop_feature_patterns)
    odds_df = _read_parquet(odds_dataset_path, engine)

    dates = sorted(training_df["date"].dropna().unique())
    splits = _make_walk_forward_splits(dates, n_splits=n_splits, min_train_ratio=min_train_ratio)

    fold_reports = []
    selected_rows = []
    for fold_idx, (test_start, test_end) in enumerate(splits, start=1):
        train_df, test_df = _split_by_start_date(training_df, test_start, test_end)
        train_df = _filter_by_surface(train_df, train_surface_id)
        if train_df.empty:
            raise ValueError(f"No training rows after train surface filter: surface_id={train_surface_id}")
        split_report = _fit_and_evaluate_catboost_split(
            train_df=train_df,
            test_df=test_df,
            odds_df=odds_df,
            stake=stake,
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
                "train_rows": len(train_df),
                "train_races": int(train_df["race_id"].nunique()),
                "auc": split_report["metrics"]["auc"],
                "logloss": split_report["metrics"]["logloss"],
                "brier": split_report["metrics"]["brier"],
            }
        )
        fold_reports.append(summary)

    try:
        import pandas as pd
    except ModuleNotFoundError as e:
        raise RuntimeError("CatBoost固定ルール評価には pandas が必要です。") from e

    all_selected = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()
    overall = _selection_summary(all_selected, stake)

    return {
        "training_dataset_path": str(training_dataset_path),
        "odds_dataset_path": str(odds_dataset_path),
        "rows": len(training_df),
        "races": int(training_df["race_id"].nunique()),
        "n_splits": len(fold_reports),
        "min_train_ratio": min_train_ratio,
        "train_filter": {
            "distance_min": None,
            "distance_max": None,
            "surface_id": train_surface_id,
        },
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
        "dropped_features": dropped_features,
        "overall": overall,
        "folds": fold_reports,
        "by_month": _fixed_rule_breakdown(all_selected, "month", stake),
        "by_track": _fixed_rule_breakdown(all_selected, "track_id", stake),
        "by_surface": _fixed_rule_breakdown(all_selected, "surface_id", stake),
    }


def build_catboost_walk_forward_predictions(
    training_dataset_path: Path,
    odds_dataset_path: Path,
    engine: str = "auto",
    n_splits: int = 4,
    min_train_ratio: float = 0.5,
    stake: float = 100.0,
    drop_feature_patterns: list[str] | None = None,
    train_surface_id: int | None = None,
) -> dict[str, Any]:
    training_df = _read_parquet(training_dataset_path, engine)
    training_df, dropped_features = _drop_feature_patterns(training_df, drop_feature_patterns)
    odds_df = _read_parquet(odds_dataset_path, engine)

    dates = sorted(training_df["date"].dropna().unique())
    splits = _make_walk_forward_splits(dates, n_splits=n_splits, min_train_ratio=min_train_ratio)

    prediction_rows = []
    fold_reports = []
    for fold_idx, (test_start, test_end) in enumerate(splits, start=1):
        train_df, test_df = _split_by_start_date(training_df, test_start, test_end)
        train_df = _filter_by_surface(train_df, train_surface_id)
        if train_df.empty:
            raise ValueError(f"No training rows after train surface filter: surface_id={train_surface_id}")
        split_report = _fit_and_evaluate_catboost_split(
            train_df=train_df,
            test_df=test_df,
            odds_df=odds_df,
            stake=stake,
            min_rule_selections=1,
        )
        eval_df = split_report["eval_df"].copy()
        eval_df["fold"] = fold_idx
        eval_df["test_start"] = test_start
        eval_df["test_end"] = test_end
        prediction_rows.append(eval_df)
        fold_reports.append(
            {
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
        )

    try:
        import pandas as pd
    except ModuleNotFoundError as e:
        raise RuntimeError("CatBoost予測保存には pandas が必要です。") from e

    predictions = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    return {
        "training_dataset_path": str(training_dataset_path),
        "odds_dataset_path": str(odds_dataset_path),
        "rows": len(training_df),
        "races": int(training_df["race_id"].nunique()),
        "n_splits": len(fold_reports),
        "min_train_ratio": min_train_ratio,
        "train_filter": {
            "surface_id": train_surface_id,
        },
        "dropped_features": dropped_features,
        "folds": fold_reports,
        "predictions": predictions,
    }
