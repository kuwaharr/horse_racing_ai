import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def main() -> None:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--url", default=None)
    arg_parser.add_argument("--race-id", default=None)
    arg_parser.add_argument("--mode", choices=["auto", "manual"], default="manual")
    arg_parser.add_argument("--limit", type=int, default=None)
    arg_parser.add_argument("--auto-stop-command", default="stop")
    args = arg_parser.parse_args()

    from src.pipelines.scrape_to_db import run, scrape_race

    if args.race_id is not None:
        if not scrape_race(args.race_id):
            raise SystemExit(1)
        return
    if args.url is None:
        arg_parser.error("--url is required unless --race-id is specified")
    run(args.url, mode=args.mode, limit=args.limit, auto_stop_command=args.auto_stop_command)


if __name__ == "__main__":
    main()
