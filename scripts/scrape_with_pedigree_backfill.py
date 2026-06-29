import argparse
import queue
import sqlite3
import subprocess
import sys
import threading
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.common.logger import get_logger
from src.data.database import count_race_ids_in_db
from src.data.paths import DB_PATH
from src.pipelines.scrape_to_db import (
    increment_page,
    scrape_race,
    sleep_backoff,
    sleep_between_requests,
)
from src.scrape.extracters import extract_race_ids
from src.scrape.fetchers import make_soup


logger = get_logger("scripts.scrape_with_pedigree_backfill", log_file="scrape_with_pedigree_backfill.log")


def _current_page_from_url(url: str) -> int:
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    return int(qs.get("page", ["1"])[0])


def _check_race_ids_in_db(db_path: Path) -> int:
    n = count_race_ids_in_db(db_path)
    logger.info("Found %s race_ids in DB", n)
    return n


def _start_command_listener() -> queue.Queue[str]:
    commands: queue.Queue[str] = queue.Queue()

    def listen() -> None:
        while True:
            line = sys.stdin.readline()
            if line == "":
                return
            command = line.strip()
            if command:
                logger.info("Command received: %s", command)
            commands.put(command)

    thread = threading.Thread(target=listen, daemon=True)
    thread.start()
    return commands


def _consume_stop_command(commands: queue.Queue[str], stop_command: str) -> bool:
    should_stop = False
    while True:
        try:
            command = commands.get_nowait()
        except queue.Empty:
            break
        if not command:
            continue
        if command == stop_command:
            should_stop = True
            logger.info("Stop command received; will stop after pedigree backfill")
        else:
            logger.warning("Unrecognized command ignored: %s", command)
    return should_stop


def _pending_pedigree_count(db_path: Path) -> int:
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT COUNT(*)
            FROM sqlite_master
            WHERE type = 'table'
              AND name = 'horse'
            """
        )
        if cur.fetchone()[0] == 0:
            return 0
        cur.execute(
            """
            SELECT COUNT(*)
            FROM horse
            WHERE pedigree_fetch_status = 'pending'
            """
        )
        return int(cur.fetchone()[0])


def _run_pedigree_backfill(args) -> None:
    command = [
        sys.executable,
        str(ROOT_DIR / "scripts" / "fetch_pending_horse_pedigrees.py"),
        "--db",
        str(args.db),
        "--until-empty",
        "--db-retries",
        str(args.pedigree_db_retries),
        "--db-retry-sleep",
        str(args.pedigree_db_retry_sleep),
        "--order-by",
        args.pedigree_order_by,
    ]
    logger.info("Starting pedigree backfill: %s", " ".join(command))
    completed = subprocess.run(command, cwd=ROOT_DIR, check=False, stdin=subprocess.DEVNULL)
    if completed.returncode == 0:
        logger.info("Pedigree backfill finished")
    else:
        logger.error("Pedigree backfill failed returncode=%s", completed.returncode)


def _maybe_run_pedigree_backfill(args) -> None:
    pending = _pending_pedigree_count(args.db)
    logger.info("Pending pedigrees: %s", pending)
    if pending < args.pedigree_threshold:
        logger.info(
            "Skipping pedigree backfill: pending=%s < threshold=%s",
            pending,
            args.pedigree_threshold,
        )
        return
    _run_pedigree_backfill(args)


def _process_page(url: str, page: int) -> bool:
    r_soup = make_soup(url)
    sleep_between_requests()
    if not r_soup.success:
        logger.error("make_soup failed %s", r_soup.error)
        sleep_backoff()
        return False

    race_ids = extract_race_ids(r_soup.value)
    if not race_ids:
        logger.info("Scraped all pages")
        return False

    logger.info("Page %s found %s race_ids in race list page", page, len(race_ids))
    for i, race_id in enumerate(race_ids, start=1):
        logger.info("%s/%s race_id=%s Processing", i, len(race_ids), race_id)
        scrape_race(race_id)
    logger.info("Page %s done", page)
    return True


def main() -> None:
    arg_parser = argparse.ArgumentParser(
        description="Scrape race list pages and run pending pedigree backfill after each page."
    )
    arg_parser.add_argument("--url", required=True, help="Race list URL. The page query is used as the starting page.")
    arg_parser.add_argument("--db", type=Path, default=DB_PATH, help="SQLite DB path.")
    arg_parser.add_argument(
        "--limit-pages",
        type=int,
        default=None,
        help="Maximum number of race list pages to process from the starting URL.",
    )
    arg_parser.add_argument(
        "--stop-command",
        default="stop",
        help="Command accepted on stdin to stop after the current page and pedigree backfill.",
    )
    arg_parser.add_argument(
        "--pedigree-threshold",
        type=int,
        default=100,
        help="Run pedigree backfill when pending pedigree count is at least this value.",
    )
    arg_parser.add_argument("--pedigree-db-retries", type=int, default=5, help="DB write retry count for backfill.")
    arg_parser.add_argument(
        "--pedigree-db-retry-sleep",
        type=float,
        default=2.0,
        help="Base sleep seconds between DB write retries for backfill.",
    )
    arg_parser.add_argument(
        "--pedigree-order-by",
        choices=["updated_at", "runner_count"],
        default="runner_count",
        help="Pending pedigree fetch order.",
    )
    args = arg_parser.parse_args()

    logger.info(
        "Starting scrape url=%s db=%s mode=%s limit=%s stop_command=%s pedigree_threshold=%s",
        args.url,
        args.db,
        "auto",
        args.limit_pages,
        args.stop_command,
        args.pedigree_threshold,
    )
    _check_race_ids_in_db(args.db)
    logger.info(
        "Auto mode command listener started. Type '%s' and Enter to stop after the current page and pedigree backfill.",
        args.stop_command,
    )

    commands = _start_command_listener()
    url = args.url
    current_page = _current_page_from_url(url)
    processed_pages = 0
    while args.limit_pages is None or processed_pages < args.limit_pages:
        page_processed = _process_page(url, current_page)
        if not page_processed:
            break

        stop_requested = _consume_stop_command(commands, args.stop_command)
        _maybe_run_pedigree_backfill(args)
        _check_race_ids_in_db(args.db)
        if stop_requested:
            logger.info("Stopped after page %s and pedigree backfill", current_page)
            break

        url = increment_page(url)
        current_page += 1
        processed_pages += 1

    if args.limit_pages is not None and processed_pages >= args.limit_pages:
        logger.info("Limit pages reached: %s", args.limit_pages)
        logger.info("Pending pedigrees: %s", _pending_pedigree_count(args.db))

    logger.info("Done")


if __name__ == "__main__":
    main()
