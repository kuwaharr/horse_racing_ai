import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.database import connect, ensure_horse_table
from src.data.paths import DB_PATH


def _format_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}%"


def main() -> None:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--db", type=Path, default=DB_PATH)
    arg_parser.add_argument("--top-n", type=int, default=20)
    arg_parser.add_argument(
        "--targets",
        type=float,
        nargs="+",
        default=[10.0, 20.0, 30.0, 50.0],
    )
    args = arg_parser.parse_args()

    with connect(args.db) as conn:
        cur = conn.cursor()
        ensure_horse_table(cur)
        cur.execute(
            """
            SELECT
                COUNT(*) AS runner_rows,
                SUM(CASE WHEN ho.pedigree_fetch_status = 'fetched' THEN 1 ELSE 0 END)
                    AS fetched_runner_rows
            FROM runner ru
            LEFT JOIN horse ho
                ON ho.horse_id = ru.horse_id
            WHERE ru.status_id = 0
              AND ru.finish IS NOT NULL
            """
        )
        total_runner_rows, fetched_runner_rows = cur.fetchone()
        fetched_runner_rows = int(fetched_runner_rows or 0)

        cur.execute(
            """
            SELECT
                ho.horse_id,
                MAX(ho.horse_name) AS horse_name,
                COUNT(*) AS runner_rows
            FROM horse ho
            INNER JOIN runner ru
                ON ru.horse_id = ho.horse_id
            WHERE ho.pedigree_fetch_status = 'pending'
              AND ru.status_id = 0
              AND ru.finish IS NOT NULL
            GROUP BY ho.horse_id
            ORDER BY runner_rows DESC, ho.updated_at, ho.horse_id
            """
        )
        pending_rows = cur.fetchall()

    total_runner_rows = int(total_runner_rows or 0)
    current_pct = None if total_runner_rows == 0 else fetched_runner_rows / total_runner_rows * 100

    print(f"DB: {args.db}")
    print("")
    print("Current coverage")
    print(f"runner rows: {total_runner_rows:,}")
    print(f"fetched runner rows: {fetched_runner_rows:,} ({_format_pct(current_pct)})")
    print(f"pending horses with runner rows: {len(pending_rows):,}")

    print("")
    print("Top pending horses by runner rows")
    print("rank  horse_id     runner_rows  projected_pct  horse_name")
    cumulative = 0
    for rank, (horse_id, horse_name, runner_rows) in enumerate(pending_rows[: args.top_n], start=1):
        cumulative += int(runner_rows)
        projected_pct = (fetched_runner_rows + cumulative) / total_runner_rows * 100
        print(
            f"{rank:>4}  {horse_id:<11} {int(runner_rows):>11,}  "
            f"{_format_pct(projected_pct):>13}  {horse_name or ''}"
        )

    print("")
    print("Targets")
    print("target  needed_horses  added_runner_rows  projected_pct")
    sorted_targets = sorted(args.targets)
    cumulative = 0
    target_index = 0
    found_targets: dict[float, tuple[int, int, float]] = {}
    for rank, (_, _, runner_rows) in enumerate(pending_rows, start=1):
        cumulative += int(runner_rows)
        projected_pct = (fetched_runner_rows + cumulative) / total_runner_rows * 100
        while target_index < len(sorted_targets) and projected_pct >= sorted_targets[target_index]:
            found_targets[sorted_targets[target_index]] = (rank, cumulative, projected_pct)
            target_index += 1
        if target_index >= len(sorted_targets):
            break

    for target in sorted_targets:
        result = found_targets.get(target)
        if result is None:
            print(f"{target:>6.2f}%  n/a            n/a                n/a")
            continue
        needed_horses, added_runner_rows, projected_pct = result
        print(
            f"{target:>6.2f}%  {needed_horses:>13,}  "
            f"{added_runner_rows:>16,}  {_format_pct(projected_pct):>13}"
        )


if __name__ == "__main__":
    main()
