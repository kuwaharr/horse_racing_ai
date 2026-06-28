import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.database import connect, ensure_horse_table
from src.data.paths import DB_PATH


def main() -> None:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--db", type=Path, default=DB_PATH)
    args = arg_parser.parse_args()

    with connect(args.db) as conn:
        cur = conn.cursor()
        ensure_horse_table(cur)
        cur.execute("SELECT COUNT(*) FROM horse")
        total = int(cur.fetchone()[0])
        cur.execute(
            """
            SELECT pedigree_fetch_status, COUNT(*)
            FROM horse
            GROUP BY pedigree_fetch_status
            ORDER BY pedigree_fetch_status
            """
        )
        status_counts = cur.fetchall()

    print(f"DB: {args.db}")
    print(f"Horses: {total:,}")
    print("")
    print("status      rows     pct")
    for status, count in status_counts:
        pct = 0 if total == 0 else count / total * 100
        print(f"{status:<10} {count:>6,}  {pct:>6.2f}%")


if __name__ == "__main__":
    main()
