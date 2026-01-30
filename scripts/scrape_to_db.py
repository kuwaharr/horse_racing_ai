import argparse
import time
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from src.common.config import DB_PATH, RAW_DIR
from src.common.db import (
    connect,
    get_race_ids_in_db,
    upsert_race,
    upsert_runner,
    upsert_place,
    upsert_wide,
    upsert_trio,
)
from src.preprocess.load_json import load_json
from src.preprocess.normalize import (
    normalize_race,
    normalize_runners,
    normalize_place,
    normalize_wide,
    normalize_trio,
)
from src.scrape.export import combine_race_dict, export_json
from src.scrape.extract import (
    extract_runners,
    extract_race_meta,
    parse_jsonp,
    parse_url,
    parse_place,
    parse_wide,
    parse_trio,
    extract_race_ids,
)
from src.scrape.fetch import fetch_odds_jsonp, make_race_url, make_soup


def increment_page(url: str) -> str:
    p = urlparse(url)
    qs = parse_qs(p.query)

    page = int(qs.get("page", ["1"])[0])
    qs["page"] = [str(page + 1)]

    return urlunparse(p._replace(query=urlencode(qs, doseq=True)))


def check_race_ids_in_db() -> int:
    race_ids_in_db = get_race_ids_in_db(DB_PATH)
    n = len(race_ids_in_db)
    print(f"[INFO] Found {n} race_ids in DB")
    return n


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--url",
        required=True,
    )
    arg_parser.add_argument(
        "--mode",
        choices=["auto", "manual"],
        default="manual",
    )
    arg_parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )
    args = arg_parser.parse_args()

    race_list_url = args.url
    mode = args.mode
    limit = args.limit

    parsed = urlparse(race_list_url)
    qs = parse_qs(parsed.query)
    current_page = int(qs.get("page", ["1"])[0])

    odds_kinds = [
        ("place", 2, parse_place),
        ("wide", 5, parse_wide),
        ("trio", 7, parse_trio),
    ]

    n_race_ids_in_db = check_race_ids_in_db()
    if limit is not None:
        if n_race_ids_in_db >= limit:
            print("[INFO] Found more race_ids in DB than limit")
            return

    while True:
        r_soup = make_soup(race_list_url)
        if not r_soup.success:
            print(f"[ERROR] make_soup failed\n{r_soup.error}")
            return
        soup = r_soup.value

        race_ids = extract_race_ids(soup)

        n_race_ids = len(race_ids)
        if n_race_ids == 0:
            print("[INFO] Scraped all pages")
            return

        print(f"[INFO][page={current_page}] Found {n_race_ids} race_ids in race list page")

        for i, race_id in enumerate(race_ids, start=1):
            print(f"[INFO][{i}/{n_race_ids}][race_id={race_id}] Processing...")

            race_url = make_race_url(race_id)

            r_soup = make_soup(race_url)
            if not r_soup.success:
                print(f"[ERROR] make_soup failed\n{r_soup.error}")
                continue
            soup = r_soup.value

            r_race_info = parse_url(race_url)
            if not r_race_info.success:
                print(f"[ERROR] parse_url failed\n{r_race_info.error}")
                continue
            race_info = r_race_info.value

            r_race_meta = extract_race_meta(soup)
            if not r_race_meta.success:
                print(f"[ERROR] exctract_race_meta failed\n{r_race_meta.error}")
                continue
            race_meta = r_race_meta.value

            race = combine_race_dict(race_info, race_meta)

            r_runners = extract_runners(soup)
            if not r_runners.success:
                print(f"[ERROR] extract_runners failed\n{r_runners.error}")
                continue
            runners = r_runners.value

            odds = {}
            for name, odds_type, parser in odds_kinds:
                odds[name] = None
                r_jsonp = fetch_odds_jsonp(race_id, odds_type, compress=0)
                if not r_jsonp.success:
                    print(f"[WARN][kind={name}] fetch_odds_jsonp failed\n{r_jsonp.error}")
                    continue
                jsonp = r_jsonp.value

                r_odds_block = parse_jsonp(jsonp, odds_type)
                if not r_odds_block.success:
                    print(f"[WARN][kind={name}] parse_jsonp failed\n{r_odds_block.error}")
                    continue
                odds_block = r_odds_block.value

                odds[name] = parser(odds_block)

            export_json(race, runners, odds)

            print("[INFO] Scraping/Exporting done")

            raw_data_file = RAW_DIR / f"{race_id}.json"

            raw_data = load_json(raw_data_file)

            conn = connect(DB_PATH)
            cur = conn.cursor()

            r_normalized_race = normalize_race(raw_data["race"])
            if r_normalized_race.success:
                upsert_race(cur, r_normalized_race.value)
            else:
                raise RuntimeError(f"normalize_race failed:\n{r_normalized_race.error}")

            r_normalized_runners = normalize_runners(race_id, raw_data["runners"])
            if r_normalized_runners.success:
                for runner in r_normalized_runners.value:
                    upsert_runner(cur, runner)
            else:
                raise RuntimeError(f"normalize_runners failed:\n{r_normalized_runners.error}")

            r_normalized_place = normalize_place(race_id, raw_data["odds"]["place"])
            if r_normalized_place.success:
                for odds in r_normalized_place.value:
                    upsert_place(cur, odds)
            else:
                raise RuntimeError(f"normalize_place failed:\n{r_normalized_place.error}")

            r_normalized_wide = normalize_wide(race_id, raw_data["odds"]["wide"])
            if r_normalized_wide.success:
                for odds in r_normalized_wide.value:
                    upsert_wide(cur, odds)
            else:
                raise RuntimeError(f"normalize_wide failed:\n{r_normalized_wide.error}")

            r_normalized_trio = normalize_trio(race_id, raw_data["odds"]["trio"])
            if r_normalized_trio.success:
                for odds in r_normalized_trio.value:
                    upsert_trio(cur, odds)
            else:
                raise RuntimeError(f"normalize_trio failed:\n{r_normalized_trio.error}")

            conn.commit()
            conn.close()

            print("[INFO] Normalizing/Upserting done")

            time.sleep(0.5)

        print(f"[INFO][page={current_page}] Done")

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


if __name__ == "__main__":
    main()