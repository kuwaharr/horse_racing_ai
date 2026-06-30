import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.paths import FEAT_DIR, MODEL_DIR
from src.features.place_top3 import default_training_dataset_name, default_win_compat_training_dataset_name
from src.models.place_top3_catboost import (
    DEFAULT_CATBOOST_PARAMS,
    _catboost_params,
    _drop_feature_patterns,
    _filter_by_surface,
    _prepare_catboost_features,
)
from src.models.place_top3_lgbm import _read_parquet


DEFAULT_PLACE_MODEL = MODEL_DIR / "catboost_place_top3_model.cbm"
DEFAULT_PLACE_METADATA = MODEL_DIR / "catboost_place_top3_model_metadata.json"
DEFAULT_WIN_MODEL = MODEL_DIR / "catboost_win_top1_model.cbm"
DEFAULT_WIN_METADATA = MODEL_DIR / "catboost_win_top1_model_metadata.json"


def _optional_str_list(value: str) -> list[str] | None:
    if value.lower() in {"none", "null", ""}:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and save a CatBoost model for live prediction.")
    parser.add_argument("--profile", choices=["place", "win"], default="place")
    parser.add_argument("--training-dataset", type=Path, default=None)
    parser.add_argument("--model-output", type=Path, default=None)
    parser.add_argument("--metadata-output", type=Path, default=None)
    parser.add_argument("--engine", choices=["auto", "pyarrow", "fastparquet"], default="auto")
    parser.add_argument("--drop-feature-patterns", type=_optional_str_list, default=None)
    parser.add_argument("--train-surface-id", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=DEFAULT_CATBOOST_PARAMS["iterations"])
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_CATBOOST_PARAMS["learning_rate"])
    parser.add_argument("--depth", type=int, default=DEFAULT_CATBOOST_PARAMS["depth"])
    parser.add_argument("--l2-leaf-reg", type=float, default=DEFAULT_CATBOOST_PARAMS["l2_leaf_reg"])
    parser.add_argument("--random-seed", type=int, default=DEFAULT_CATBOOST_PARAMS["random_seed"])
    args = parser.parse_args()

    try:
        from catboost import CatBoostClassifier
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "CatBoostモデル保存には catboost が必要です。"
            " PowerShellで `.\\.venv\\Scripts\\python.exe -m pip install catboost` を実行してください。"
        ) from e

    training_dataset = args.training_dataset
    if training_dataset is None:
        training_dataset = (
            FEAT_DIR / default_training_dataset_name()
            if args.profile == "place"
            else FEAT_DIR / default_win_compat_training_dataset_name()
        )
    model_output = args.model_output or (DEFAULT_PLACE_MODEL if args.profile == "place" else DEFAULT_WIN_MODEL)
    metadata_output = args.metadata_output or (
        DEFAULT_PLACE_METADATA if args.profile == "place" else DEFAULT_WIN_METADATA
    )

    training_df = _read_parquet(training_dataset, args.engine)
    training_df, dropped_features = _drop_feature_patterns(training_df, args.drop_feature_patterns)
    training_df = _filter_by_surface(training_df, args.train_surface_id)
    if training_df.empty:
        raise ValueError(f"No training rows after train surface filter: surface_id={args.train_surface_id}")

    train_x, _, feature_cols, categorical_cols, cat_feature_indices = _prepare_catboost_features(
        training_df,
        training_df,
    )
    train_y = training_df["target_top3"]
    params = _catboost_params(
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        l2_leaf_reg=args.l2_leaf_reg,
        random_seed=args.random_seed,
    )
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=params["iterations"],
        learning_rate=params["learning_rate"],
        depth=params["depth"],
        l2_leaf_reg=params["l2_leaf_reg"],
        random_seed=params["random_seed"],
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(train_x, train_y, cat_features=cat_feature_indices)

    model_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(model_output)
    metadata = {
        "profile": args.profile,
        "training_dataset": str(training_dataset),
        "model_output": str(model_output),
        "feature_cols": feature_cols,
        "categorical_cols": categorical_cols,
        "catboost_params": params,
        "dropped_features": dropped_features,
        "train_surface_id": args.train_surface_id,
        "rows": len(training_df),
        "races": int(training_df["race_id"].nunique()),
        "date_min": str(training_df["date"].min()),
        "date_max": str(training_df["date"].max()),
    }
    metadata_output.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Training dataset: {training_dataset}")
    print(f"Rows: {len(training_df):,}")
    print(f"Races: {metadata['races']:,}")
    print(f"Date range: {metadata['date_min']} - {metadata['date_max']}")
    print(f"Features: {len(feature_cols):,}")
    print(f"Categorical features: {len(categorical_cols):,}")
    print(f"Dropped features: {len(dropped_features):,}")
    print(f"Model output: {model_output}")
    print(f"Metadata output: {metadata_output}")


if __name__ == "__main__":
    main()
