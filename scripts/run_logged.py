import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


DEFAULT_LOG_DIR = Path("logs") / "codex_runs"


def _safe_name(value: str) -> str:
    safe_chars = []
    for char in value:
        if char.isalnum() or char in {"-", "_"}:
            safe_chars.append(char)
        else:
            safe_chars.append("_")
    safe = "".join(safe_chars).strip("_")
    return safe or "command"


def _build_log_path(log_dir: Path, name: str | None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = _safe_name(name) if name else "command"
    return log_dir / f"{timestamp}_{suffix}.log"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a command and save its stdout/stderr to a timestamped log file."
    )
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--name", default=None, help="Short label used in the log file name.")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("command is required after --")
    return args


def main() -> None:
    args = _parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = _build_log_path(args.log_dir, args.name)

    command_text = " ".join(args.command)
    started_at = datetime.now().isoformat(timespec="seconds")

    with log_path.open("w", encoding="utf-8", newline="") as log_file:
        header = [
            f"started_at: {started_at}",
            f"cwd: {Path.cwd()}",
            f"command: {command_text}",
            "",
        ]
        for line in header:
            print(line)
            log_file.write(line + "\n")

        process = subprocess.Popen(
            args.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return_code = process.wait()

        footer = [
            "",
            f"finished_at: {datetime.now().isoformat(timespec='seconds')}",
            f"return_code: {return_code}",
            f"log_path: {log_path}",
        ]
        for line in footer:
            print(line)
            log_file.write(line + "\n")

    raise SystemExit(return_code)


if __name__ == "__main__":
    main()
