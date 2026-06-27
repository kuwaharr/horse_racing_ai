import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.paths import FEAT_DIR
from src.features.place_top3 import default_dataset_name
from src.models.place_top3_lgbm import evaluate_place_top3_lgbm, format_lgbm_report


def _float_list(value: str) -> list[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def main() -> None:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--early-dataset", type=Path, default=FEAT_DIR / default_dataset_name("early"))
    arg_parser.add_argument("--late-dataset", type=Path, default=FEAT_DIR / default_dataset_name("late"))
    arg_parser.add_argument("--engine", choices=["auto", "pyarrow", "fastparquet"], default="auto")
    arg_parser.add_argument("--test-ratio", type=float, default=0.2)
    arg_parser.add_argument("--stake", type=float, default=100.0)
    arg_parser.add_argument("--pred-thresholds", type=_float_list, default=None)
    arg_parser.add_argument("--ev-thresholds", type=_float_list, default=None)
    args = arg_parser.parse_args()

    report = evaluate_place_top3_lgbm(
        early_dataset_path=args.early_dataset,
        late_dataset_path=args.late_dataset,
        engine=args.engine,
        test_ratio=args.test_ratio,
        stake=args.stake,
        pred_thresholds=args.pred_thresholds,
        expected_value_thresholds=args.ev_thresholds,
    )
    print(format_lgbm_report(report))


if __name__ == "__main__":
    main()
