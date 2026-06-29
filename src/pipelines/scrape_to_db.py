import argparse
import queue
import random
import sys
import threading
import time
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from src.common.logger import get_logger
from src.data.database import (
    connect,
    get_race_ids_in_db,
    run_write_with_retry,
    ensure_horse_table,
    upsert_horse_pending,
    upsert_place,
    upsert_race,
    upsert_runner,
    upsert_trio,
    upsert_win,
    upsert_wide,
)
from src.data.paths import DB_PATH, RAW_DIR
from src.preprocess.json_loader import load_json
from src.preprocess.normalizers import (
    normalize_place,
    normalize_race,
    normalize_runners,
    normalize_trio,
    normalize_win,
    normalize_wide,
)
from src.scrape.extracters import (
    extract_race_ids,
    extract_race_meta,
    extract_runners,
    parse_jsonp,
    parse_place,
    parse_trio,
    parse_url,
    parse_win,
    parse_wide,
)
from src.scrape.fetchers import fetch_odds_jsonp, make_race_url, make_soup
from src.scrape.json_exporter import combine_race_dict, export_json

logger = get_logger("src.pipelines.scrape_to_db")


def increment_page(url: str) -> str:
    p = urlparse(url)
    qs = parse_qs(p.query)

    page = int(qs.get("page", ["1"])[0])
    qs["page"] = [str(page + 1)]

    return urlunparse(p._replace(query=urlencode(qs, doseq=True)))


def check_race_ids_in_db() -> int:
    race_ids_in_db = get_race_ids_in_db(DB_PATH)
    n = len(race_ids_in_db)
    logger.info("Found %s race_ids in DB", n)
    return n


def sleep_between_requests(min_sec: float = 3.0, max_sec: float = 6.0) -> None:
    time.sleep(random.uniform(min_sec, max_sec))


def sleep_backoff(min_sec: float = 30.0, max_sec: float = 60.0) -> None:
    time.sleep(random.uniform(min_sec, max_sec))


def start_auto_command_listener() -> queue.Queue[str]:
    commands: queue.Queue[str] = queue.Queue()

    def listen() -> None:
        while True:
            line = sys.stdin.readline()
            if line == "":
                return
            command = line.strip()
            if command:
                logger.info("Auto command received: %s", command)
            commands.put(command)

    thread = threading.Thread(target=listen, daemon=True)
    thread.start()
    return commands


def should_stop_after_page(commands: queue.Queue[str], stop_command: str) -> bool:
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
            logger.info("Stop command received; stopping after current page")
        else:
            logger.warning("Unrecognized command ignored: %s", command)
    return should_stop


def write_raw_race_to_db(race_id: str, raw_data: dict) -> bool:
    with connect(DB_PATH) as conn:
        cur = conn.cursor()

        failed = False

        r_normalized_race = normalize_race(raw_data["race"])
        if not r_normalized_race.success:
            logger.error("race_id=%s normalize_race failed %s", race_id, r_normalized_race.error)
            failed = True

        r_normalized_runners = normalize_runners(race_id, raw_data["runners"])
        if not r_normalized_runners.success:
            logger.error("race_id=%s normalize_runners failed %s", race_id, r_normalized_runners.error)
            failed = True

        r_normalized_win = normalize_win(race_id, raw_data["odds"]["win"])
        if not r_normalized_win.success:
            logger.error("race_id=%s normalize_win failed %s", race_id, r_normalized_win.error)
            failed = True

        r_normalized_place = normalize_place(race_id, raw_data["odds"]["place"])
        if not r_normalized_place.success:
            logger.error("race_id=%s normalize_place failed %s", race_id, r_normalized_place.error)
            failed = True

        r_normalized_wide = normalize_wide(race_id, raw_data["odds"]["wide"])
        if not r_normalized_wide.success:
            logger.error("race_id=%s normalize_wide failed %s", race_id, r_normalized_wide.error)
            failed = True

        r_normalized_trio = normalize_trio(race_id, raw_data["odds"]["trio"])
        if not r_normalized_trio.success:
            logger.error("race_id=%s normalize_trio failed %s", race_id, r_normalized_trio.error)
            failed = True

        if failed:
            logger.error("Normalizing failed; skipping race_id=%s", race_id)
            return False

        def write() -> None:
            upsert_race(cur, r_normalized_race.value)
            ensure_horse_table(cur)
            for runner in r_normalized_runners.value:
                upsert_runner(cur, runner)
                upsert_horse_pending(
                    cur,
                    runner.get("horse_id"),
                    runner.get("horse_name"),
                    ensure_table=False,
                )
            for odds in r_normalized_win.value:
                upsert_win(cur, odds)
            for odds in r_normalized_place.value:
                upsert_place(cur, odds)
            for odds in r_normalized_wide.value:
                upsert_wide(cur, odds)
            for odds in r_normalized_trio.value:
                upsert_trio(cur, odds)

        run_write_with_retry(conn, write)
        return True


def default_odds_kinds():
    return [
        ("win", 1, parse_win),
        ("place", 2, parse_place),
        ("wide", 5, parse_wide),
        ("trio", 7, parse_trio),
    ]


def scrape_race(race_id: str, race_page_retries: int = 3) -> bool:
    race_url = make_race_url(race_id)

    race = None
    runners = None
    for attempt in range(1, race_page_retries + 1):
        r_soup = make_soup(race_url)
        sleep_between_requests()
        if not r_soup.success:
            logger.error("race_id=%s make_soup failed attempt=%s/%s %s", race_id, attempt, race_page_retries, r_soup.error)
            sleep_backoff()
            continue
        soup = r_soup.value

        r_race_info = parse_url(race_url)
        if not r_race_info.success:
            logger.error("race_id=%s parse_url failed %s", race_id, r_race_info.error)
            return False
        race_info = r_race_info.value

        r_race_meta = extract_race_meta(soup)
        if not r_race_meta.success:
            logger.error(
                "race_id=%s extract_race_meta failed attempt=%s/%s %s",
                race_id,
                attempt,
                race_page_retries,
                r_race_meta.error,
            )
            continue
        race_meta = r_race_meta.value

        r_runners = extract_runners(soup)
        if not r_runners.success:
            logger.error(
                "race_id=%s extract_runners failed attempt=%s/%s %s",
                race_id,
                attempt,
                race_page_retries,
                r_runners.error,
            )
            continue

        race = combine_race_dict(race_info, race_meta)
        runners = r_runners.value
        break

    if race is None or runners is None:
        logger.error("race_id=%s race page retries exhausted; skipping", race_id)
        return False

    odds = {}
    for name, odds_type, parser in default_odds_kinds():
        odds[name] = None
        r_jsonp = fetch_odds_jsonp(race_id, odds_type, compress=0)
        sleep_between_requests()
        if not r_jsonp.success:
            logger.warning("race_id=%s kind=%s fetch_odds_jsonp failed %s", race_id, name, r_jsonp.error)
            sleep_backoff()
            continue
        jsonp = r_jsonp.value

        r_odds_block = parse_jsonp(jsonp, odds_type)
        if not r_odds_block.success:
            logger.warning("race_id=%s kind=%s parse_jsonp failed %s", race_id, name, r_odds_block.error)
            sleep_backoff()
            continue
        odds_block = r_odds_block.value

        odds[name] = parser(odds_block)

    export_json(race, runners, odds)
    logger.info("race_id=%s Scraping/Exporting done", race_id)

    raw_data_file = RAW_DIR / f"{race_id}.json"
    raw_data = load_json(raw_data_file)

    if not write_raw_race_to_db(race_id, raw_data):
        return False

    logger.info("race_id=%s Normalizing/Upserting done", race_id)
    return True


def run(
    race_list_url: str,
    mode: str = "manual",
    limit: int | None = None,
    auto_stop_command: str = "stop",
) -> None:
    parsed = urlparse(race_list_url)
    qs = parse_qs(parsed.query)
    current_page = int(qs.get("page", ["1"])[0])

    logger.info(
        "Starting scrape url=%s db=%s mode=%s limit=%s stop_command=%s",
        race_list_url,
        DB_PATH,
        mode,
        limit,
        auto_stop_command,
    )
    n_race_ids_in_db = check_race_ids_in_db()
    if limit is not None:
        if n_race_ids_in_db >= limit:
            logger.info("Found more race_ids in DB than limit")
            return

    auto_commands = None
    if mode == "auto":
        auto_commands = start_auto_command_listener()
        logger.info(
            "Auto mode command listener started. Type '%s' and Enter to stop after the current page.",
            auto_stop_command,
        )

    while True:
        r_soup = make_soup(race_list_url)
        sleep_between_requests()
        if not r_soup.success:
            logger.error("make_soup failed %s", r_soup.error)
            sleep_backoff()
            return
        soup = r_soup.value

        race_ids = extract_race_ids(soup)

        n_race_ids = len(race_ids)
        if n_race_ids == 0:
            logger.info("Scraped all pages")
            return

        logger.info("Page %s found %s race_ids in race list page", current_page, n_race_ids)

        for i, race_id in enumerate(race_ids, start=1):
            logger.info("%s/%s race_id=%s Processing", i, n_race_ids, race_id)
            scrape_race(race_id)

        logger.info("Page %s done", current_page)

        check_race_ids_in_db()

        if mode == "auto":
            if auto_commands is not None and should_stop_after_page(auto_commands, auto_stop_command):
                break
            current_page += 1
            race_list_url = increment_page(race_list_url)
        else:
            conti = input("Continue? Enter/'n' >>> ").strip()
            if not conti:
                current_page += 1
                race_list_url = increment_page(race_list_url)
            elif conti == "n":
                break


def main(argv: list[str] | None = None) -> None:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--url", default=None)
    arg_parser.add_argument("--race-id", default=None)
    arg_parser.add_argument("--mode", choices=["auto", "manual"], default="manual")
    arg_parser.add_argument("--limit", type=int, default=None)
    arg_parser.add_argument("--auto-stop-command", default="stop")
    args = arg_parser.parse_args(argv)

    if args.race_id is not None:
        if not scrape_race(args.race_id):
            raise SystemExit(1)
        return
    if args.url is None:
        arg_parser.error("--url is required unless --race-id is specified")

    run(args.url, mode=args.mode, limit=args.limit, auto_stop_command=args.auto_stop_command)


if __name__ == "__main__":
    main()
