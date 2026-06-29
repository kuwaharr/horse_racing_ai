import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def main() -> None:
    arg_parser = argparse.ArgumentParser(description="Scrape netkeiba race pages and upsert them into SQLite.")
    arg_parser.add_argument("--url", default=None, help="Race list URL. Required unless --race-id is specified.")
    arg_parser.add_argument("--race-id", default=None, help="Scrape a single race_id instead of a race list.")
    arg_parser.add_argument(
        "--mode",
        choices=["auto", "manual"],
        default="manual",
        help="auto increments list pages; manual asks after each page.",
    )
    arg_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Exit before scraping when DB race_id count is already at least this value.",
    )
    arg_parser.add_argument(
        "--auto-stop-command",
        default="stop",
        help="Command accepted on stdin to stop auto mode after the current page.",
    )
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
