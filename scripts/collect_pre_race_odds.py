import argparse
import random
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.common.logger import get_logger
from src.data.database import (
    connect,
    ensure_pre_race_odds_tables,
    insert_pre_race_odds_rows,
    insert_pre_race_odds_snapshot,
    run_write_with_retry,
)
from src.data.paths import DB_PATH, PRE_RACE_ODDS_RAW_DIR
from src.preprocess.normalizers import normalize_place, normalize_trio, normalize_win, normalize_wide
from src.scrape.extracters import (
    extract_race_ids,
    extract_race_meta,
    parse_jsonp,
    parse_place,
    parse_trio,
    parse_win,
    parse_wide,
)
from src.scrape.fetchers import fetch_odds_jsonp, make_soup


logger = get_logger("scripts.collect_pre_race_odds")


BET_KINDS = {
    "win": (1, parse_win, normalize_win),
    "place": (2, parse_place, normalize_place),
    "wide": (5, parse_wide, normalize_wide),
    "trio": (7, parse_trio, normalize_trio),
}

TIME_BUCKETS = [
    ("over_120", 120.0, None),
    ("pre_60_120", 60.0, 120.0),
    ("pre_30_60", 30.0, 60.0),
    ("pre_15_30", 15.0, 30.0),
    ("pre_5_15", 5.0, 15.0),
    ("pre_2_5", 2.0, 5.0),
    ("pre_0_2", 0.0, 2.0),
]

WATCH_TARGET_MINUTES = {
    "over_120": 120.0,
    "pre_60_120": 60.0,
    "pre_30_60": 30.0,
    "pre_15_30": 15.0,
    "pre_5_15": 5.0,
    "pre_2_5": 2.0,
    "pre_0_2": 0.5,
}


@dataclass(frozen=True)
class RaceTarget:
    race_id: str
    race_date: str | None
    post_time: str | None
    post_time_at: datetime | None


def sleep_between_requests(min_sec: float, max_sec: float) -> None:
    time.sleep(random.uniform(min_sec, max_sec))


def is_jra_race_id(race_id: str) -> bool:
    if len(race_id) != 12 or not race_id.isdigit():
        return False
    return 1 <= int(race_id[4:6]) <= 10


def race_list_url_for_date(value: date) -> str:
    return f"https://race.netkeiba.com/top/race_list.html?kaisai_date={value:%Y%m%d}"


def race_entry_url(race_id: str) -> str:
    return f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}&rf=race_submenu"


def parse_date_arg(value: str) -> date:
    return date.fromisoformat(value)


def weekend_dates(today: date) -> list[date]:
    if today.weekday() == 5:
        return [today, today + timedelta(days=1), today + timedelta(days=2)]
    if today.weekday() == 6:
        return [today, today + timedelta(days=1)]
    if today.weekday() == 0:
        return [today]
    saturday_offset = (5 - today.weekday()) % 7
    saturday = today + timedelta(days=saturday_offset)
    return [saturday, saturday + timedelta(days=1), saturday + timedelta(days=2)]


def time_bucket(minutes_to_post: float | None) -> str:
    if minutes_to_post is None:
        return "unknown"
    if minutes_to_post < 0:
        return "post_time"
    for bucket, lower, upper in TIME_BUCKETS:
        if minutes_to_post >= lower and (upper is None or minutes_to_post < upper):
            return bucket
    return "unknown"


def normalize_race_date(race_id: str, raw_date: str | None) -> str | None:
    if raw_date is None:
        return None
    import re

    m = re.match(r"(\d{1,2})月(\d{1,2})日.*", raw_date)
    if not m:
        return None
    return f"{race_id[:4]}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"


def post_time_at(race_date: str | None, post_time: str | None) -> datetime | None:
    if race_date is None or post_time is None:
        return None
    try:
        hour, minute = post_time.split(":")
        return datetime.fromisoformat(race_date).replace(hour=int(hour), minute=int(minute))
    except ValueError:
        return None


def minutes_to_post(now: datetime, post_at: datetime | None) -> float | None:
    if post_at is None:
        return None
    return (post_at - now).total_seconds() / 60.0


def fetch_race_ids_for_date(value: date, min_sleep: float, max_sleep: float) -> list[str]:
    url = race_list_url_for_date(value)
    result = make_soup(url)
    sleep_between_requests(min_sleep, max_sleep)
    if not result.success:
        logger.warning("date=%s race list fetch failed: %s", value, result.error)
        return []
    race_ids = [race_id for race_id in extract_race_ids(result.value) if is_jra_race_id(race_id)]
    logger.info("date=%s found %s JRA race_ids", value, len(race_ids))
    return race_ids


def fetch_race_target(race_id: str, min_sleep: float, max_sleep: float) -> RaceTarget:
    result = make_soup(race_entry_url(race_id))
    sleep_between_requests(min_sleep, max_sleep)
    if not result.success:
        logger.warning("race_id=%s race page fetch failed: %s", race_id, result.error)
        return RaceTarget(race_id=race_id, race_date=None, post_time=None, post_time_at=None)
    meta_result = extract_race_meta(result.value)
    if not meta_result.success:
        logger.warning("race_id=%s race meta parse failed: %s", race_id, meta_result.error)
        return RaceTarget(race_id=race_id, race_date=None, post_time=None, post_time_at=None)

    raw_meta = meta_result.value
    race_date = normalize_race_date(race_id, raw_meta.get("date"))
    post_time = raw_meta.get("post_time")
    return RaceTarget(
        race_id=race_id,
        race_date=race_date,
        post_time=post_time,
        post_time_at=post_time_at(race_date, post_time),
    )


def resolve_race_targets(
    race_ids: list[str],
    min_sleep: float,
    max_sleep: float,
) -> list[RaceTarget]:
    targets = []
    for i, race_id in enumerate(sorted(set(race_ids)), start=1):
        logger.info("%s/%s race_id=%s resolving race metadata", i, len(set(race_ids)), race_id)
        targets.append(fetch_race_target(race_id, min_sleep, max_sleep))
    return targets


def collect_race_bet(
    target: RaceTarget,
    bet_type: str,
    db_path: Path,
    raw_dir: Path,
    min_sleep: float,
    max_sleep: float,
    record_empty: bool,
) -> bool:
    now = datetime.now()
    minutes = minutes_to_post(now, target.post_time_at)
    bucket = time_bucket(minutes)
    odds_type, parser, normalizer = BET_KINDS[bet_type]
    raw_path = None
    status = "failed"
    error = None
    rows = []

    result = fetch_odds_jsonp(target.race_id, odds_type, compress=0)
    sleep_between_requests(min_sleep, max_sleep)
    if not result.success:
        error = result.error
    else:
        race_raw_dir = raw_dir / target.race_id
        race_raw_dir.mkdir(parents=True, exist_ok=True)
        raw_path_obj = race_raw_dir / f"{now:%Y%m%dT%H%M%S}_{bet_type}.jsonp"
        raw_path_obj.write_text(result.value, encoding="utf-8")
        raw_path = str(raw_path_obj)

        parsed = parse_jsonp(result.value, odds_type)
        if not parsed.success:
            status = "no_odds"
            error = parsed.error
        else:
            odds = parser(parsed.value)
            normalized = normalizer(target.race_id, odds)
            if not normalized.success:
                status = "failed"
                error = normalized.error
            else:
                rows = normalized.value
                status = "fetched" if rows and has_any_odds(rows) else "no_odds"

    if status != "fetched" and not record_empty:
        logger.info(
            "race_id=%s bet_type=%s bucket=%s skipped status=%s error=%s",
            target.race_id,
            bet_type,
            bucket,
            status,
            error,
        )
        return False

    snapshot = {
        "race_id": target.race_id,
        "bet_type": bet_type,
        "source": "netkeiba",
        "snapshot_at": now.isoformat(timespec="seconds"),
        "race_date": target.race_date,
        "post_time": target.post_time,
        "post_time_at": None if target.post_time_at is None else target.post_time_at.isoformat(timespec="seconds"),
        "minutes_to_post": minutes,
        "time_bucket": bucket,
        "status": status,
        "row_count": len(rows),
        "raw_path": raw_path,
        "error": error,
    }

    with connect(db_path) as conn:
        cur = conn.cursor()
        ensure_pre_race_odds_tables(cur)

        def write() -> None:
            snapshot_id = insert_pre_race_odds_snapshot(cur, snapshot)
            if status == "fetched":
                insert_pre_race_odds_rows(cur, bet_type, snapshot_id, rows)

        run_write_with_retry(conn, write)

    logger.info(
        "race_id=%s bet_type=%s bucket=%s status=%s rows=%s",
        target.race_id,
        bet_type,
        bucket,
        status,
        len(rows),
    )
    return status == "fetched"


def has_any_odds(rows: list[dict]) -> bool:
    odds_columns = {"odds", "odds_min", "odds_max"}
    return any(any(row.get(col) is not None for col in odds_columns) for row in rows)


def collect_targets(
    targets: list[RaceTarget],
    bet_types: list[str],
    db_path: Path,
    raw_dir: Path,
    min_sleep: float,
    max_sleep: float,
    record_empty: bool,
) -> None:
    for i, target in enumerate(targets, start=1):
        logger.info("%s/%s race_id=%s collecting odds", i, len(targets), target.race_id)
        for bet_type in bet_types:
            collect_race_bet(target, bet_type, db_path, raw_dir, min_sleep, max_sleep, record_empty)


def should_collect_in_watch(
    target: RaceTarget,
    now: datetime,
    collected: set[tuple[str, str]],
    poll_seconds: int,
) -> str | None:
    minutes = minutes_to_post(now, target.post_time_at)
    if minutes is None or minutes < 0:
        return None
    window_minutes = max(poll_seconds / 60.0, 0.5)
    for bucket, target_minutes in WATCH_TARGET_MINUTES.items():
        key = (target.race_id, bucket)
        if key in collected:
            continue
        if bucket == "pre_0_2":
            if 0 <= minutes <= target_minutes + window_minutes:
                return bucket
        elif target_minutes <= minutes <= target_minutes + window_minutes:
            return bucket
    return None


def run_watch(
    targets: list[RaceTarget],
    bet_types: list[str],
    db_path: Path,
    raw_dir: Path,
    min_sleep: float,
    max_sleep: float,
    poll_seconds: int,
    record_empty: bool,
) -> None:
    collected: set[tuple[str, str]] = set()
    targets = [target for target in targets if target.post_time_at is not None]
    logger.info("watch started targets=%s bet_types=%s poll_seconds=%s", len(targets), ",".join(bet_types), poll_seconds)
    if not targets:
        logger.info("watch finished; no targets have known post_time_at")
        return
    while True:
        now = datetime.now()
        active_targets = [target for target in targets if target.post_time_at is not None and target.post_time_at >= now]
        if not active_targets:
            logger.info("watch finished; no future targets remain")
            return
        for target in active_targets:
            bucket = should_collect_in_watch(target, now, collected, poll_seconds)
            if bucket is None:
                continue
            logger.info("race_id=%s watch due bucket=%s", target.race_id, bucket)
            for bet_type in bet_types:
                collect_race_bet(target, bet_type, db_path, raw_dir, min_sleep, max_sleep, record_empty)
            collected.add((target.race_id, bucket))
        time.sleep(poll_seconds)


def parse_bet_types(value: str) -> list[str]:
    bet_types = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in bet_types if item not in BET_KINDS]
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown bet types: {unknown}")
    return bet_types


def build_targets(args) -> list[RaceTarget]:
    race_ids = []
    for race_id in args.race_id or []:
        if is_jra_race_id(race_id):
            race_ids.append(race_id)
        else:
            logger.warning("race_id=%s skipped because it is not a JRA race_id", race_id)

    target_dates: list[date] = []
    for value in args.date or []:
        target_dates.append(parse_date_arg(value))
    if args.weekend:
        for value in weekend_dates(date.today()):
            if value not in target_dates:
                target_dates.append(value)

    for target_date in target_dates:
        race_ids.extend(fetch_race_ids_for_date(target_date, args.min_sleep, args.max_sleep))

    return resolve_race_targets(race_ids, args.min_sleep, args.max_sleep)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect pre-race JRA odds snapshots from netkeiba.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--sweep", action="store_true", help="Collect currently available odds once.")
    mode.add_argument("--watch", action="store_true", help="Watch races and collect once near each bucket boundary.")
    parser.add_argument("--date", action="append", help="Target date in YYYY-MM-DD. Can be specified multiple times.")
    parser.add_argument("--weekend", action="store_true", help="Target upcoming Saturday, Sunday, and Monday.")
    parser.add_argument("--race-id", action="append", help="Collect a specific race_id. Can be specified multiple times.")
    parser.add_argument("--bet-types", type=parse_bet_types, default=list(BET_KINDS), help="Comma-separated bet types.")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--raw-dir", type=Path, default=PRE_RACE_ODDS_RAW_DIR)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--min-sleep", type=float, default=1.5)
    parser.add_argument("--max-sleep", type=float, default=3.0)
    parser.add_argument("--record-empty", action="store_true", help="Store failed/no-odds snapshots instead of only logging them.")
    args = parser.parse_args()

    if not args.date and not args.weekend and not args.race_id:
        parser.error("At least one of --date, --weekend, or --race-id is required.")

    with connect(args.db) as conn:
        ensure_pre_race_odds_tables(conn.cursor())
        conn.commit()

    targets = build_targets(args)
    if not targets:
        logger.info("No targets found")
        return

    if args.sweep:
        collect_targets(
            targets,
            args.bet_types,
            args.db,
            args.raw_dir,
            args.min_sleep,
            args.max_sleep,
            args.record_empty,
        )
    else:
        run_watch(
            targets,
            args.bet_types,
            args.db,
            args.raw_dir,
            args.min_sleep,
            args.max_sleep,
            args.poll_seconds,
            args.record_empty,
        )


if __name__ == "__main__":
    main()
