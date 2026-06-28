import argparse
import sys
from datetime import datetime
from pathlib import Path


DEFAULT_LOG_DIR = Path("logs") / "codex_conversations"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append a Codex conversation entry to a Markdown log.")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--date", default=None, help="Log date in YYYY-MM-DD format. Defaults to today.")
    parser.add_argument("--role", choices=["user", "assistant", "system", "summary"], default="assistant")
    parser.add_argument("--title", default=None)
    parser.add_argument("--message", default=None, help="Message text. If omitted, stdin is used.")
    return parser.parse_args()


def _read_message(message_arg: str | None) -> str:
    if message_arg is not None:
        return message_arg
    if sys.stdin.isatty():
        raise SystemExit("No message provided. Use --message or pipe text via stdin.")
    return sys.stdin.read()


def _build_entry(role: str, title: str | None, message: str) -> str:
    timestamp = datetime.now().isoformat(timespec="seconds")
    heading = f"## {timestamp} {role}"
    if title:
        heading += f" - {title}"
    message = message.strip()
    return f"{heading}\n\n{message}\n\n"


def main() -> None:
    args = _parse_args()
    log_date = args.date or datetime.now().strftime("%Y-%m-%d")
    log_path = args.log_dir / f"{log_date}.md"
    message = _read_message(args.message)

    args.log_dir.mkdir(parents=True, exist_ok=True)
    entry = _build_entry(args.role, args.title, message)
    with log_path.open("a", encoding="utf-8", newline="") as f:
        f.write(entry)

    print(f"Appended conversation log: {log_path}")


if __name__ == "__main__":
    main()
