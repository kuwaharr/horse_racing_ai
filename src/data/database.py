import sqlite3
from pathlib import Path


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def get_race_ids_in_db(db_path: Path) -> set:
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT race_id FROM race")
        return {row[0] for row in cur.fetchall()}


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
