import argparse
import random
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.common.logger import get_logger
from src.data.database import (
    connect,
    get_horse_for_pedigree_fetch,
    get_horses_for_pedigree_fetch,
    run_write_with_retry,
    update_horse_pedigree,
)
from src.data.paths import DB_PATH
from src.scrape.extracters import extract_horse_pedigree
from src.scrape.fetchers import make_horse_pedigree_url, make_soup


logger = get_logger("scripts.fetch_pending_horse_pedigrees", log_file="fetch_pending_horse_pedigrees.log")

PEDIGREE_REQUEST_SLEEP_MIN = 1.5
PEDIGREE_REQUEST_SLEEP_MAX = 2.0


def _failed_pedigree(horse_id: str, horse_name: str | None, error: str) -> dict:
    return {
        "horse_id": horse_id,
        "horse_name": horse_name,
        "sire_id": None,
        "sire_name": None,
        "dam_id": None,
        "dam_name": None,
        "broodmare_sire_id": None,
        "broodmare_sire_name": None,
        "pedigree_fetch_status": "failed",
        "pedigree_fetch_error": error,
    }


def _get_next_horse(args) -> dict | None:
    with connect(args.db) as conn:
        cur = conn.cursor()
        if args.horse_id is not None:
            return get_horse_for_pedigree_fetch(cur, args.horse_id)
        horses = get_horses_for_pedigree_fetch(
            cur,
            limit=1,
            include_failed=args.include_failed,
            order_by=args.order_by,
        )
        return horses[0] if horses else None


def _save_pedigree_result(args, pedigree: dict) -> None:
    with connect(args.db) as conn:
        cur = conn.cursor()
        run_write_with_retry(
            conn,
            lambda: update_horse_pedigree(cur, pedigree),
            max_attempts=args.db_retries,
            sleep_sec=args.db_retry_sleep,
        )


def _sleep_between_pedigree_requests() -> None:
    sleep_sec = random.uniform(PEDIGREE_REQUEST_SLEEP_MIN, PEDIGREE_REQUEST_SLEEP_MAX)
    logger.info("Sleeping %.2f seconds before next pedigree request", sleep_sec)
    time.sleep(sleep_sec)


def main() -> None:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--db", type=Path, default=DB_PATH)
    arg_parser.add_argument("--limit", type=int, default=20)
    arg_parser.add_argument("--until-empty", action="store_true")
    arg_parser.add_argument("--include-failed", action="store_true")
    arg_parser.add_argument("--horse-id", default=None)
    arg_parser.add_argument("--db-retries", type=int, default=5)
    arg_parser.add_argument("--db-retry-sleep", type=float, default=2.0)
    arg_parser.add_argument(
        "--order-by",
        choices=["updated_at", "runner_count"],
        default="updated_at",
    )
    args = arg_parser.parse_args()
    if args.until_empty and args.include_failed:
        arg_parser.error("--until-empty cannot be used with --include-failed")

    fetched = 0
    failed = 0
    logger.info(
        "Starting pedigree fetch db=%s limit=%s until_empty=%s sleep_range=%.1f-%.1f include_failed=%s horse_id=%s order_by=%s",
        args.db,
        args.limit,
        args.until_empty,
        PEDIGREE_REQUEST_SLEEP_MIN,
        PEDIGREE_REQUEST_SLEEP_MAX,
        args.include_failed,
        args.horse_id,
        args.order_by,
    )

    i = 0
    while args.until_empty or i < args.limit:
        i += 1
        horse = _get_next_horse(args)
        if horse is None:
            if args.horse_id is not None:
                logger.warning("horse_id=%s is not in horse table", args.horse_id)
            else:
                logger.info("No pending horses found")
            break

        horse_id = horse["horse_id"]
        url = make_horse_pedigree_url(horse_id)
        runner_count = horse.get("runner_count")
        runner_count_text = "" if runner_count is None else f" runner_count={runner_count}"
        total_text = "until-empty" if args.until_empty else str(args.limit)
        logger.info("%s/%s horse_id=%s%s url=%s", i, total_text, horse_id, runner_count_text, url)

        soup_result = make_soup(url)
        if not soup_result.success:
            failed += 1
            logger.warning("horse_id=%s fetch failed: %s", horse_id, soup_result.error)
            _save_pedigree_result(
                args,
                _failed_pedigree(horse_id, horse["horse_name"], soup_result.error or "fetch failed"),
            )
            _sleep_between_pedigree_requests()
            continue

        pedigree_result = extract_horse_pedigree(soup_result.value, horse_id)
        if not pedigree_result.success:
            failed += 1
            logger.warning("horse_id=%s parse failed: %s", horse_id, pedigree_result.error)
            _save_pedigree_result(
                args,
                _failed_pedigree(
                    horse_id,
                    horse["horse_name"],
                    pedigree_result.error or "parse failed",
                ),
            )
            _sleep_between_pedigree_requests()
            continue

        _save_pedigree_result(args, pedigree_result.value)
        fetched += 1
        logger.info("horse_id=%s fetched", horse_id)
        _sleep_between_pedigree_requests()

        if args.horse_id is not None:
            break

    logger.info("Done fetched=%s failed=%s", fetched, failed)


if __name__ == "__main__":
    main()
