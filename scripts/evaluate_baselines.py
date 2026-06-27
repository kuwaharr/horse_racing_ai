import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.paths import FEAT_DIR
from src.evaluate.baselines import evaluate_baselines, format_baseline_report
from src.features.place_top3 import default_eval_odds_dataset_name


def main() -> None:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--dataset", type=Path, default=FEAT_DIR / default_eval_odds_dataset_name())
    arg_parser.add_argument("--engine", choices=["auto", "pyarrow", "fastparquet"], default="auto")
    arg_parser.add_argument("--test-ratio", type=float, default=0.2)
    args = arg_parser.parse_args()

    report = evaluate_baselines(args.dataset, engine=args.engine, test_ratio=args.test_ratio)
    print(format_baseline_report(report))


if __name__ == "__main__":
    main()
