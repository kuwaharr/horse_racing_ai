import sqlite3
import time
from collections.abc import Callable
from pathlib import Path


DEFAULT_SQLITE_TIMEOUT_SEC = 60.0
DEFAULT_BUSY_TIMEOUT_MS = 60_000


HORSE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS horse (
    horse_id TEXT PRIMARY KEY,
    horse_name TEXT,
    sire_id TEXT,
    sire_name TEXT,
    dam_id TEXT,
    dam_name TEXT,
    broodmare_sire_id TEXT,
    broodmare_sire_name TEXT,
    pedigree_fetched_at TEXT,
    pedigree_fetch_status TEXT NOT NULL DEFAULT 'pending',
    pedigree_fetch_error TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (pedigree_fetch_status IN ('pending', 'fetched', 'failed', 'not_found'))
);

CREATE INDEX IF NOT EXISTS idx_horse_pedigree_fetch_status
    ON horse(pedigree_fetch_status);
"""


PRE_RACE_ODDS_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS pre_race_odds_snapshot (
    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id TEXT NOT NULL,
    bet_type TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'netkeiba',
    snapshot_at TEXT NOT NULL,
    race_date TEXT,
    post_time TEXT,
    post_time_at TEXT,
    minutes_to_post REAL,
    time_bucket TEXT NOT NULL,
    status TEXT NOT NULL,
    row_count INTEGER NOT NULL DEFAULT 0,
    raw_path TEXT,
    error TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (bet_type IN ('win', 'place', 'wide', 'trio')),
    CHECK (status IN ('fetched', 'no_odds', 'failed'))
);

CREATE INDEX IF NOT EXISTS idx_pre_race_odds_snapshot_race_bucket
    ON pre_race_odds_snapshot(race_id, time_bucket, bet_type);

CREATE INDEX IF NOT EXISTS idx_pre_race_odds_snapshot_post_time
    ON pre_race_odds_snapshot(post_time_at);

CREATE TABLE IF NOT EXISTS pre_race_win_odds (
    snapshot_id INTEGER NOT NULL,
    horse_number INTEGER NOT NULL,
    odds REAL,
    CHECK (odds IS NULL OR odds > 0),
    PRIMARY KEY (snapshot_id, horse_number),
    FOREIGN KEY (snapshot_id) REFERENCES pre_race_odds_snapshot(snapshot_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS pre_race_place_odds (
    snapshot_id INTEGER NOT NULL,
    horse_number INTEGER NOT NULL,
    odds_min REAL,
    odds_max REAL,
    CHECK (odds_min IS NULL OR odds_min > 0),
    CHECK (odds_max IS NULL OR odds_max > 0),
    CHECK (odds_min IS NULL OR odds_max IS NULL OR odds_min <= odds_max),
    PRIMARY KEY (snapshot_id, horse_number),
    FOREIGN KEY (snapshot_id) REFERENCES pre_race_odds_snapshot(snapshot_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS pre_race_wide_odds (
    snapshot_id INTEGER NOT NULL,
    horse_number_1 INTEGER NOT NULL,
    horse_number_2 INTEGER NOT NULL,
    odds_min REAL,
    odds_max REAL,
    CHECK (horse_number_1 < horse_number_2),
    CHECK (odds_min IS NULL OR odds_min > 0),
    CHECK (odds_max IS NULL OR odds_max > 0),
    CHECK (odds_min IS NULL OR odds_max IS NULL OR odds_min <= odds_max),
    PRIMARY KEY (snapshot_id, horse_number_1, horse_number_2),
    FOREIGN KEY (snapshot_id) REFERENCES pre_race_odds_snapshot(snapshot_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS pre_race_trio_odds (
    snapshot_id INTEGER NOT NULL,
    horse_number_1 INTEGER NOT NULL,
    horse_number_2 INTEGER NOT NULL,
    horse_number_3 INTEGER NOT NULL,
    odds REAL,
    CHECK (horse_number_1 < horse_number_2 AND horse_number_2 < horse_number_3),
    CHECK (odds IS NULL OR odds > 0),
    PRIMARY KEY (snapshot_id, horse_number_1, horse_number_2, horse_number_3),
    FOREIGN KEY (snapshot_id) REFERENCES pre_race_odds_snapshot(snapshot_id) ON DELETE CASCADE
);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=DEFAULT_SQLITE_TIMEOUT_SEC)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute(f"PRAGMA busy_timeout = {DEFAULT_BUSY_TIMEOUT_MS}")
    return conn


def run_write_with_retry(
    conn: sqlite3.Connection,
    write_fn: Callable[[], None],
    max_attempts: int = 5,
    sleep_sec: float = 2.0,
) -> None:
    for attempt in range(1, max_attempts + 1):
        try:
            write_fn()
            conn.commit()
            return
        except sqlite3.OperationalError as e:
            if "database is locked" not in str(e).lower() or attempt == max_attempts:
                raise
            conn.rollback()
            time.sleep(sleep_sec * attempt)


def ensure_horse_table(cur: sqlite3.Cursor) -> None:
    cur.executescript(HORSE_SCHEMA_SQL)


def ensure_pre_race_odds_tables(cur: sqlite3.Cursor) -> None:
    cur.executescript(PRE_RACE_ODDS_SCHEMA_SQL)


def get_race_ids_in_db(db_path: Path) -> set:
    with sqlite3.connect(db_path, timeout=DEFAULT_SQLITE_TIMEOUT_SEC) as conn:
        conn.execute(f"PRAGMA busy_timeout = {DEFAULT_BUSY_TIMEOUT_MS}")
        cur = conn.cursor()
        cur.execute("SELECT race_id FROM race")
        return {row[0] for row in cur.fetchall()}


def count_race_ids_in_db(db_path: Path) -> int:
    with sqlite3.connect(db_path, timeout=DEFAULT_SQLITE_TIMEOUT_SEC) as conn:
        conn.execute(f"PRAGMA busy_timeout = {DEFAULT_BUSY_TIMEOUT_MS}")
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM race")
        return int(cur.fetchone()[0])


def upsert_race(cur: sqlite3.Cursor, race: dict) -> None:
    cols = list(race.keys())
    set_cols = [c for c in cols if c != "race_id"]
    sql = f"""
        INSERT INTO race ({", ".join(cols)})
        VALUES ({", ".join(["?"] * len(cols))})
        ON CONFLICT(race_id) DO UPDATE SET
            {", ".join([f"{c}=excluded.{c}" for c in set_cols])}
    """
    cur.execute(sql, [race[c] for c in cols])


def upsert_runner(cur: sqlite3.Cursor, runner: dict) -> None:
    cols = list(runner.keys())
    set_cols = [c for c in cols if c not in ("race_id", "horse_number")]
    sql = f"""
        INSERT INTO runner ({", ".join(cols)})
        VALUES ({", ".join(["?"] * len(cols))})
        ON CONFLICT(race_id, horse_number) DO UPDATE SET
            {", ".join([f"{c}=excluded.{c}" for c in set_cols])}
    """
    cur.execute(sql, [runner[c] for c in cols])


def upsert_horse_pending(
    cur: sqlite3.Cursor,
    horse_id: str | None,
    horse_name: str | None,
    ensure_table: bool = True,
) -> None:
    if not horse_id:
        return
    if ensure_table:
        ensure_horse_table(cur)
    cur.execute(
        """
        INSERT INTO horse (
            horse_id,
            horse_name,
            pedigree_fetch_status,
            created_at,
            updated_at
        )
        VALUES (?, ?, 'pending', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        ON CONFLICT(horse_id) DO UPDATE SET
            horse_name = COALESCE(excluded.horse_name, horse.horse_name),
            updated_at = CURRENT_TIMESTAMP
        """,
        [horse_id, horse_name],
    )


def backfill_horses_from_runners(cur: sqlite3.Cursor) -> int:
    ensure_horse_table(cur)
    cur.execute(
        """
        INSERT INTO horse (
            horse_id,
            horse_name,
            pedigree_fetch_status,
            created_at,
            updated_at
        )
        SELECT
            runner.horse_id,
            MAX(runner.horse_name),
            'pending',
            CURRENT_TIMESTAMP,
            CURRENT_TIMESTAMP
        FROM runner
        WHERE runner.horse_id IS NOT NULL
          AND runner.horse_id != ''
        GROUP BY runner.horse_id
        ON CONFLICT(horse_id) DO UPDATE SET
            horse_name = COALESCE(horse.horse_name, excluded.horse_name),
            updated_at = CURRENT_TIMESTAMP
        """
    )
    return cur.rowcount


def get_horses_for_pedigree_fetch(
    cur: sqlite3.Cursor,
    limit: int,
    include_failed: bool = False,
    order_by: str = "updated_at",
) -> list[dict]:
    ensure_horse_table(cur)
    if order_by not in {"updated_at", "runner_count"}:
        raise ValueError(f"Unknown pedigree fetch order: {order_by}")

    statuses = ("pending", "failed") if include_failed else ("pending",)
    placeholders = ", ".join(["?"] * len(statuses))
    if order_by == "updated_at":
        cur.execute(
            f"""
            SELECT horse_id, horse_name, pedigree_fetch_status, NULL AS runner_count
            FROM horse
            WHERE pedigree_fetch_status IN ({placeholders})
            ORDER BY updated_at, horse_id
            LIMIT ?
            """,
            [*statuses, limit],
        )
        return [
            {
                "horse_id": row[0],
                "horse_name": row[1],
                "pedigree_fetch_status": row[2],
                "runner_count": row[3],
            }
            for row in cur.fetchall()
        ]

    cur.execute(
        f"""
        SELECT
            h.horse_id,
            h.horse_name,
            h.pedigree_fetch_status,
            COUNT(runner.horse_id) AS runner_count
        FROM horse AS h
        LEFT JOIN runner
            ON runner.horse_id = h.horse_id
        WHERE h.pedigree_fetch_status IN ({placeholders})
        GROUP BY h.horse_id, h.horse_name, h.pedigree_fetch_status, h.updated_at
        ORDER BY runner_count DESC, h.updated_at, h.horse_id
        LIMIT ?
        """,
        [*statuses, limit],
    )
    return [
        {
            "horse_id": row[0],
            "horse_name": row[1],
            "pedigree_fetch_status": row[2],
            "runner_count": row[3],
        }
        for row in cur.fetchall()
    ]


def get_horse_for_pedigree_fetch(cur: sqlite3.Cursor, horse_id: str) -> dict | None:
    ensure_horse_table(cur)
    cur.execute(
        """
        SELECT horse_id, horse_name, pedigree_fetch_status
        FROM horse
        WHERE horse_id = ?
        """,
        [horse_id],
    )
    row = cur.fetchone()
    if row is None:
        return None
    return {"horse_id": row[0], "horse_name": row[1], "pedigree_fetch_status": row[2]}


def update_horse_pedigree(
    cur: sqlite3.Cursor,
    pedigree: dict,
    ensure_table: bool = True,
) -> None:
    if ensure_table:
        ensure_horse_table(cur)
    cur.execute(
        """
        UPDATE horse
        SET
            horse_name = COALESCE(?, horse_name),
            sire_id = ?,
            sire_name = ?,
            dam_id = ?,
            dam_name = ?,
            broodmare_sire_id = ?,
            broodmare_sire_name = ?,
            pedigree_fetched_at = CURRENT_TIMESTAMP,
            pedigree_fetch_status = ?,
            pedigree_fetch_error = ?,
            updated_at = CURRENT_TIMESTAMP
        WHERE horse_id = ?
        """,
        [
            pedigree.get("horse_name"),
            pedigree.get("sire_id"),
            pedigree.get("sire_name"),
            pedigree.get("dam_id"),
            pedigree.get("dam_name"),
            pedigree.get("broodmare_sire_id"),
            pedigree.get("broodmare_sire_name"),
            pedigree["pedigree_fetch_status"],
            pedigree.get("pedigree_fetch_error"),
            pedigree["horse_id"],
        ],
    )


def upsert_place(cur: sqlite3.Cursor, odds: dict) -> None:
    cols = list(odds.keys())
    set_cols = [c for c in cols if c not in ("race_id", "horse_number")]
    sql = f"""
        INSERT INTO place_odds ({", ".join(cols)})
        VALUES ({", ".join(["?"] * len(cols))})
        ON CONFLICT(race_id, horse_number) DO UPDATE SET
            {", ".join([f"{c}=excluded.{c}" for c in set_cols])}
    """
    cur.execute(sql, [odds[c] for c in cols])


def upsert_win(cur: sqlite3.Cursor, odds: dict) -> None:
    cols = list(odds.keys())
    set_cols = [c for c in cols if c not in ("race_id", "horse_number")]
    sql = f"""
        INSERT INTO win_odds ({", ".join(cols)})
        VALUES ({", ".join(["?"] * len(cols))})
        ON CONFLICT(race_id, horse_number) DO UPDATE SET
            {", ".join([f"{c}=excluded.{c}" for c in set_cols])}
    """
    cur.execute(sql, [odds[c] for c in cols])


def upsert_wide(cur: sqlite3.Cursor, odds: dict) -> None:
    cols = list(odds.keys())
    set_cols = [c for c in cols if c not in ("race_id", "horse_number_1", "horse_number_2")]
    sql = f"""
        INSERT INTO wide_odds ({", ".join(cols)})
        VALUES ({", ".join(["?"] * len(cols))})
        ON CONFLICT(race_id, horse_number_1, horse_number_2) DO UPDATE SET
            {", ".join([f"{c}=excluded.{c}" for c in set_cols])}
    """
    cur.execute(sql, [odds[c] for c in cols])


def upsert_trio(cur: sqlite3.Cursor, odds: dict) -> None:
    cols = list(odds.keys())
    set_cols = [c for c in cols if c not in ("race_id", "horse_number_1", "horse_number_2", "horse_number_3")]
    sql = f"""
        INSERT INTO trio_odds ({", ".join(cols)})
        VALUES ({", ".join(["?"] * len(cols))})
        ON CONFLICT(race_id, horse_number_1, horse_number_2, horse_number_3) DO UPDATE SET
            {", ".join([f"{c}=excluded.{c}" for c in set_cols])}
    """
    cur.execute(sql, [odds[c] for c in cols])


def insert_pre_race_odds_snapshot(cur: sqlite3.Cursor, snapshot: dict) -> int:
    cols = list(snapshot.keys())
    sql = f"""
        INSERT INTO pre_race_odds_snapshot ({", ".join(cols)})
        VALUES ({", ".join(["?"] * len(cols))})
    """
    cur.execute(sql, [snapshot[c] for c in cols])
    return int(cur.lastrowid)


def insert_pre_race_odds_rows(
    cur: sqlite3.Cursor,
    bet_type: str,
    snapshot_id: int,
    rows: list[dict],
) -> None:
    table_by_bet_type = {
        "win": "pre_race_win_odds",
        "place": "pre_race_place_odds",
        "wide": "pre_race_wide_odds",
        "trio": "pre_race_trio_odds",
    }
    table = table_by_bet_type[bet_type]
    for row in rows:
        row_with_snapshot = {"snapshot_id": snapshot_id}
        row_with_snapshot.update({k: v for k, v in row.items() if k != "race_id"})
        cols = list(row_with_snapshot.keys())
        sql = f"""
            INSERT INTO {table} ({", ".join(cols)})
            VALUES ({", ".join(["?"] * len(cols))})
        """
        cur.execute(sql, [row_with_snapshot[c] for c in cols])
