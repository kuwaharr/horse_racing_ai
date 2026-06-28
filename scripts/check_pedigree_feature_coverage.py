import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.database import connect, ensure_horse_table
from src.data.paths import DB_PATH


def _format_pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{numerator / denominator * 100:.2f}%"


def main() -> None:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--db", type=Path, default=DB_PATH)
    arg_parser.add_argument("--months", type=int, default=12)
    args = arg_parser.parse_args()

    with connect(args.db) as conn:
        cur = conn.cursor()
        ensure_horse_table(cur)
        cur.execute(
            """
            SELECT
                COUNT(*) AS runner_rows,
                SUM(CASE WHEN ho.pedigree_fetch_status = 'fetched' THEN 1 ELSE 0 END)
                    AS fetched_runner_rows,
                COUNT(DISTINCT ru.race_id) AS races,
                COUNT(DISTINCT CASE
                    WHEN ho.pedigree_fetch_status = 'fetched' THEN ru.race_id
                END) AS races_with_any_pedigree
            FROM runner ru
            INNER JOIN race ra
                ON ra.race_id = ru.race_id
            LEFT JOIN horse ho
                ON ho.horse_id = ru.horse_id
            WHERE ru.status_id = 0
              AND ru.finish IS NOT NULL
            """
        )
        runner_rows, fetched_runner_rows, races, races_with_any_pedigree = cur.fetchone()

        cur.execute(
            """
            WITH race_coverage AS (
                SELECT
                    ru.race_id,
                    COUNT(*) AS runner_rows,
                    SUM(CASE WHEN ho.pedigree_fetch_status = 'fetched' THEN 1 ELSE 0 END)
                        AS fetched_runner_rows
                FROM runner ru
                LEFT JOIN horse ho
                    ON ho.horse_id = ru.horse_id
                WHERE ru.status_id = 0
                  AND ru.finish IS NOT NULL
                GROUP BY ru.race_id
            )
            SELECT
                SUM(CASE WHEN fetched_runner_rows = runner_rows THEN 1 ELSE 0 END)
                    AS races_full_pedigree,
                SUM(CASE WHEN fetched_runner_rows * 2 >= runner_rows THEN 1 ELSE 0 END)
                    AS races_half_pedigree
            FROM race_coverage
            """
        )
        races_full_pedigree, races_half_pedigree = cur.fetchone()

        cur.execute(
            """
            SELECT
                substr(ra.date, 1, 7) AS year_month,
                COUNT(*) AS runner_rows,
                SUM(CASE WHEN ho.pedigree_fetch_status = 'fetched' THEN 1 ELSE 0 END)
                    AS fetched_runner_rows,
                COUNT(DISTINCT ru.race_id) AS races,
                COUNT(DISTINCT CASE
                    WHEN ho.pedigree_fetch_status = 'fetched' THEN ru.race_id
                END) AS races_with_any_pedigree
            FROM runner ru
            INNER JOIN race ra
                ON ra.race_id = ru.race_id
            LEFT JOIN horse ho
                ON ho.horse_id = ru.horse_id
            WHERE ru.status_id = 0
              AND ru.finish IS NOT NULL
            GROUP BY year_month
            ORDER BY year_month DESC
            LIMIT ?
            """,
            [args.months],
        )
        monthly_rows = cur.fetchall()

    print(f"DB: {args.db}")
    print("")
    print("Overall pedigree coverage")
    print(f"runner rows: {runner_rows:,}")
    print(f"fetched runner rows: {fetched_runner_rows:,} ({_format_pct(fetched_runner_rows, runner_rows)})")
    print(f"races: {races:,}")
    print(
        "races with any pedigree: "
        f"{races_with_any_pedigree:,} ({_format_pct(races_with_any_pedigree, races)})"
    )
    print(
        "races with >=50% pedigree: "
        f"{races_half_pedigree:,} ({_format_pct(races_half_pedigree, races)})"
    )
    print(
        "races with full pedigree: "
        f"{races_full_pedigree:,} ({_format_pct(races_full_pedigree, races)})"
    )
    print("")
    print("Monthly coverage")
    print("month      runner_rows  fetched_rows  row_pct  races  any_races  any_pct")
    for year_month, month_runner_rows, month_fetched_rows, month_races, month_any_races in monthly_rows:
        print(
            f"{year_month:<10} {month_runner_rows:>11,}  {month_fetched_rows:>12,}  "
            f"{_format_pct(month_fetched_rows, month_runner_rows):>7}  "
            f"{month_races:>5,}  {month_any_races:>9,}  "
            f"{_format_pct(month_any_races, month_races):>7}"
        )


if __name__ == "__main__":
    main()
