import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.database import connect, run_write_with_retry
from src.data.paths import DB_PATH


INDEXES = [
    (
        "idx_runner_horse_id",
        "CREATE INDEX IF NOT EXISTS idx_runner_horse_id ON runner(horse_id)",
    ),
]


def main() -> None:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--db", type=Path, default=DB_PATH)
    args = arg_parser.parse_args()

    with connect(args.db) as conn:
        cur = conn.cursor()
        for name, sql in INDEXES:
            run_write_with_retry(conn, lambda sql=sql: cur.execute(sql))
            print(f"ensured: {name}")

    print(f"DB: {args.db}")


if __name__ == "__main__":
    main()
