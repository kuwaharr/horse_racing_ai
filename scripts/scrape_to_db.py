import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def main() -> None:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--url", required=True)
    arg_parser.add_argument("--mode", choices=["auto", "manual"], default="manual")
    arg_parser.add_argument("--limit", type=int, default=None)
    args = arg_parser.parse_args()

    from src.pipelines.scrape_to_db import run

    run(args.url, mode=args.mode, limit=args.limit)


if __name__ == "__main__":
    main()
