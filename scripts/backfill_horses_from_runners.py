import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.database import backfill_horses_from_runners, connect
from src.data.paths import DB_PATH


def main() -> None:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--db", type=Path, default=DB_PATH)
    args = arg_parser.parse_args()

    with connect(args.db) as conn:
        cur = conn.cursor()
        changed_rows = backfill_horses_from_runners(cur)
        conn.commit()

        cur.execute("SELECT COUNT(*) FROM horse")
        horse_count = int(cur.fetchone()[0])
        cur.execute("SELECT pedigree_fetch_status, COUNT(*) FROM horse GROUP BY pedigree_fetch_status")
        status_counts = cur.fetchall()

    print(f"DB: {args.db}")
    print(f"Backfill changed rows: {changed_rows}")
    print(f"Horses: {horse_count:,}")
    print("")
    print("status      rows")
    for status, count in status_counts:
        print(f"{status:<10} {count:>6,}")


if __name__ == "__main__":
    main()
