import sqlite3
from pathlib import Path
from typing import Any


def _count(cur: sqlite3.Cursor, table: str) -> int:
    cur.execute(f"SELECT COUNT(*) FROM {table}")
    return int(cur.fetchone()[0])


def _pct(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator * 100


def _odds_coverage(cur: sqlite3.Cursor, table: str) -> dict[str, Any]:
    cur.execute(
        f"""
        SELECT COUNT(*)
        FROM runner r
        INNER JOIN {table} o
            ON o.race_id = r.race_id
            AND o.horse_number = r.horse_number
        """
    )
    covered = int(cur.fetchone()[0])

    total = _count(cur, "runner")
    return {
        "covered": covered,
        "total": total,
        "coverage_pct": _pct(covered, total),
        "missing": total - covered,
    }


def _pair_odds_races(cur: sqlite3.Cursor, table: str) -> dict[str, Any]:
    cur.execute(f"SELECT COUNT(DISTINCT race_id) FROM {table}")
    covered = int(cur.fetchone()[0])

    total = _count(cur, "race")
    return {
        "covered": covered,
        "total": total,
        "coverage_pct": _pct(covered, total),
        "missing": total - covered,
    }


def collect_db_quality(db_path: Path) -> dict[str, Any]:
    if not db_path.exists():
        return {
            "db_path": str(db_path),
            "db_exists": False,
        }

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        cur = conn.cursor()

        race_count = _count(cur, "race")
        runner_count = _count(cur, "runner")

        cur.execute(
            """
            SELECT COUNT(*)
            FROM race ra
            LEFT JOIN (
                SELECT race_id, COUNT(*) AS runner_count
                FROM runner
                GROUP BY race_id
            ) ru ON ru.race_id = ra.race_id
            WHERE ra.race_size IS NOT NULL
              AND COALESCE(ru.runner_count, 0) != ra.race_size
            """
        )
        runner_count_mismatch = int(cur.fetchone()[0])

        cur.execute(
            """
            SELECT COUNT(*)
            FROM race ra
            WHERE NOT EXISTS (SELECT 1 FROM win_odds w WHERE w.race_id = ra.race_id)
              AND NOT EXISTS (SELECT 1 FROM place_odds p WHERE p.race_id = ra.race_id)
              AND NOT EXISTS (SELECT 1 FROM wide_odds wd WHERE wd.race_id = ra.race_id)
              AND NOT EXISTS (SELECT 1 FROM trio_odds t WHERE t.race_id = ra.race_id)
            """
        )
        races_without_any_odds = int(cur.fetchone()[0])

        cur.execute(
            """
            SELECT
                COUNT(*) AS finished,
                SUM(CASE WHEN finish <= 3 THEN 1 ELSE 0 END) AS placed
            FROM runner
            WHERE status_id = 0
              AND finish IS NOT NULL
            """
        )
        finished, placed = cur.fetchone()
        finished = int(finished or 0)
        placed = int(placed or 0)

        cur.execute("SELECT COUNT(*) FROM runner WHERE status_id != 0")
        non_finished = int(cur.fetchone()[0])

        return {
            "db_path": str(db_path),
            "db_exists": True,
            "counts": {
                "race": race_count,
                "runner": runner_count,
                "win_odds": _count(cur, "win_odds"),
                "place_odds": _count(cur, "place_odds"),
                "wide_odds": _count(cur, "wide_odds"),
                "trio_odds": _count(cur, "trio_odds"),
            },
            "runner_count_mismatch": runner_count_mismatch,
            "races_without_any_odds": races_without_any_odds,
            "non_finished_runners": non_finished,
            "placed": {
                "count": placed,
                "finished_runners": finished,
                "rate_pct": _pct(placed, finished),
            },
            "coverage": {
                "win_odds": _odds_coverage(cur, "win_odds"),
                "place_odds": _odds_coverage(cur, "place_odds"),
                "wide_odds_races": _pair_odds_races(cur, "wide_odds"),
                "trio_odds_races": _pair_odds_races(cur, "trio_odds"),
            },
        }


def format_db_quality(report: dict[str, Any]) -> str:
    lines = [f"DB: {report['db_path']}"]
    if not report["db_exists"]:
        lines.append("status: DB file does not exist")
        return "\n".join(lines)

    counts = report["counts"]
    lines.extend(
        [
            "",
            "Counts",
            f"  races: {counts['race']:,}",
            f"  runners: {counts['runner']:,}",
            f"  win_odds: {counts['win_odds']:,}",
            f"  place_odds: {counts['place_odds']:,}",
            f"  wide_odds: {counts['wide_odds']:,}",
            f"  trio_odds: {counts['trio_odds']:,}",
            "",
            "Quality",
            f"  runner count mismatch races: {report['runner_count_mismatch']:,}",
            f"  races without any odds: {report['races_without_any_odds']:,}",
            f"  non-finished runners: {report['non_finished_runners']:,}",
        ]
    )

    placed = report["placed"]
    placed_rate = placed["rate_pct"]
    placed_rate_text = "n/a" if placed_rate is None else f"{placed_rate:.2f}%"
    lines.extend(
        [
            f"  finish<=3 positive rate: {placed_rate_text}",
            f"  finish<=3 runners: {placed['count']:,} / {placed['finished_runners']:,}",
            "",
            "Coverage",
        ]
    )

    for name, item in report["coverage"].items():
        pct = item["coverage_pct"]
        pct_text = "n/a" if pct is None else f"{pct:.2f}%"
        lines.append(
            f"  {name}: {pct_text} ({item['covered']:,} / {item['total']:,}, missing {item['missing']:,})"
        )

    return "\n".join(lines)
