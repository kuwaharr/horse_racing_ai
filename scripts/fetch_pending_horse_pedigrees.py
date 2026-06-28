import argparse
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.database import (
    connect,
    get_horse_for_pedigree_fetch,
    get_horses_for_pedigree_fetch,
    update_horse_pedigree,
)
from src.data.paths import DB_PATH
from src.scrape.extracters import extract_horse_pedigree
from src.scrape.fetchers import make_horse_pedigree_url, make_soup


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


def main() -> None:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--db", type=Path, default=DB_PATH)
    arg_parser.add_argument("--limit", type=int, default=20)
    arg_parser.add_argument("--sleep", type=float, default=1.0)
    arg_parser.add_argument("--include-failed", action="store_true")
    arg_parser.add_argument("--horse-id", default=None)
    arg_parser.add_argument(
        "--order-by",
        choices=["updated_at", "runner_count"],
        default="updated_at",
    )
    args = arg_parser.parse_args()

    fetched = 0
    failed = 0
    with connect(args.db) as conn:
        cur = conn.cursor()
        if args.horse_id is not None:
            horse = get_horse_for_pedigree_fetch(cur, args.horse_id)
            if horse is None:
                raise RuntimeError(f"horse_id={args.horse_id} is not in horse table")
            horses = [horse]
        else:
            horses = get_horses_for_pedigree_fetch(
                cur,
                limit=args.limit,
                include_failed=args.include_failed,
                order_by=args.order_by,
            )
        print(f"DB: {args.db}")
        print(f"Targets: {len(horses):,}")

        for i, horse in enumerate(horses, start=1):
            horse_id = horse["horse_id"]
            url = make_horse_pedigree_url(horse_id)
            runner_count = horse.get("runner_count")
            runner_count_text = "" if runner_count is None else f" runner_count={runner_count}"
            print(f"{i}/{len(horses)} horse_id={horse_id}{runner_count_text} url={url}")

            soup_result = make_soup(url)
            if not soup_result.success:
                failed += 1
                update_horse_pedigree(
                    cur,
                    _failed_pedigree(horse_id, horse["horse_name"], soup_result.error or "fetch failed"),
                )
                conn.commit()
                time.sleep(args.sleep)
                continue

            pedigree_result = extract_horse_pedigree(soup_result.value, horse_id)
            if not pedigree_result.success:
                failed += 1
                update_horse_pedigree(
                    cur,
                    _failed_pedigree(
                        horse_id,
                        horse["horse_name"],
                        pedigree_result.error or "parse failed",
                    ),
                )
                conn.commit()
                time.sleep(args.sleep)
                continue

            update_horse_pedigree(cur, pedigree_result.value)
            conn.commit()
            fetched += 1
            time.sleep(args.sleep)

    print("")
    print(f"Fetched: {fetched:,}")
    print(f"Failed: {failed:,}")


if __name__ == "__main__":
    main()
