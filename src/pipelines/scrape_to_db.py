import argparse
import random
import time
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from src.common.logger import get_logger
from src.data.database import (
    connect,
    get_race_ids_in_db,
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


def run(race_list_url: str, mode: str = "manual", limit: int | None = None) -> None:
    parsed = urlparse(race_list_url)
    qs = parse_qs(parsed.query)
    current_page = int(qs.get("page", ["1"])[0])

    odds_kinds = [
        ("win", 1, parse_win),
        ("place", 2, parse_place),
        ("wide", 5, parse_wide),
        ("trio", 7, parse_trio),
    ]

    n_race_ids_in_db = check_race_ids_in_db()
    if limit is not None:
        if n_race_ids_in_db >= limit:
            logger.info("Found more race_ids in DB than limit")
            return

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

        logger.info("page=%s Found %s race_ids in race list page", current_page, n_race_ids)

        for i, race_id in enumerate(race_ids, start=1):
            logger.info("%s/%s race_id=%s Processing", i, n_race_ids, race_id)

            race_url = make_race_url(race_id)

            r_soup = make_soup(race_url)
            sleep_between_requests()
            if not r_soup.success:
                logger.error("make_soup failed %s", r_soup.error)
                sleep_backoff()
                continue
            soup = r_soup.value

            r_race_info = parse_url(race_url)
            if not r_race_info.success:
                logger.error("parse_url failed %s", r_race_info.error)
                continue
            race_info = r_race_info.value

            r_race_meta = extract_race_meta(soup)
            if not r_race_meta.success:
                logger.error("extract_race_meta failed %s", r_race_meta.error)
                continue
            race_meta = r_race_meta.value

            race = combine_race_dict(race_info, race_meta)

            r_runners = extract_runners(soup)
            if not r_runners.success:
                logger.error("extract_runners failed %s", r_runners.error)
                continue
            runners = r_runners.value

            odds = {}
            for name, odds_type, parser in odds_kinds:
                odds[name] = None
                r_jsonp = fetch_odds_jsonp(race_id, odds_type, compress=0)
                sleep_between_requests()
                if not r_jsonp.success:
                    logger.warning("kind=%s fetch_odds_jsonp failed %s", name, r_jsonp.error)
                    sleep_backoff()
                    continue
                jsonp = r_jsonp.value

                r_odds_block = parse_jsonp(jsonp, odds_type)
                if not r_odds_block.success:
                    logger.warning("kind=%s parse_jsonp failed %s", name, r_odds_block.error)
                    sleep_backoff()
                    continue
                odds_block = r_odds_block.value

                odds[name] = parser(odds_block)

            export_json(race, runners, odds)

            logger.info("Scraping/Exporting done")

            raw_data_file = RAW_DIR / f"{race_id}.json"

            raw_data = load_json(raw_data_file)

            with connect(DB_PATH) as conn:
                cur = conn.cursor()

                failed = False

                r_normalized_race = normalize_race(raw_data["race"])
                if r_normalized_race.success:
                    upsert_race(cur, r_normalized_race.value)
                else:
                    logger.error("race_id=%s normalize_race failed %s", race_id, r_normalized_race.error)
                    failed = True

                r_normalized_runners = normalize_runners(race_id, raw_data["runners"])
                if r_normalized_runners.success:
                    for runner in r_normalized_runners.value:
                        upsert_runner(cur, runner)
                else:
                    logger.error("race_id=%s normalize_runners failed %s", race_id, r_normalized_runners.error)
                    failed = True

                r_normalized_win = normalize_win(race_id, raw_data["odds"]["win"])
                if r_normalized_win.success:
                    for odds in r_normalized_win.value:
                        upsert_win(cur, odds)
                else:
                    logger.error("race_id=%s normalize_win failed %s", race_id, r_normalized_win.error)
                    failed = True

                r_normalized_place = normalize_place(race_id, raw_data["odds"]["place"])
                if r_normalized_place.success:
                    for odds in r_normalized_place.value:
                        upsert_place(cur, odds)
                else:
                    logger.error("race_id=%s normalize_place failed %s", race_id, r_normalized_place.error)
                    failed = True

                r_normalized_wide = normalize_wide(race_id, raw_data["odds"]["wide"])
                if r_normalized_wide.success:
                    for odds in r_normalized_wide.value:
                        upsert_wide(cur, odds)
                else:
                    logger.error("race_id=%s normalize_wide failed %s", race_id, r_normalized_wide.error)
                    failed = True

                r_normalized_trio = normalize_trio(race_id, raw_data["odds"]["trio"])
                if r_normalized_trio.success:
                    for odds in r_normalized_trio.value:
                        upsert_trio(cur, odds)
                else:
                    logger.error("race_id=%s normalize_trio failed %s", race_id, r_normalized_trio.error)
                    failed = True

                if failed:
                    conn.rollback()
                    logger.error("Normalizing/Upserting failed; skipping race_id=%s", race_id)
                    continue

                conn.commit()

            logger.info("Normalizing/Upserting done")

        logger.info("page=%s Done", current_page)

        check_race_ids_in_db()

        if mode == "auto":
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
    arg_parser.add_argument("--url", required=True)
    arg_parser.add_argument("--mode", choices=["auto", "manual"], default="manual")
    arg_parser.add_argument("--limit", type=int, default=None)
    args = arg_parser.parse_args(argv)

    run(args.url, mode=args.mode, limit=args.limit)


if __name__ == "__main__":
    main()
