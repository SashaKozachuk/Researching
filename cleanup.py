#!/usr/bin/env python3
"""Clean generated workspace folders every 3 hours."""

from __future__ import annotations

import argparse
import os
import shutil
import time
from datetime import datetime
from pathlib import Path


DEFAULT_INTERVAL_SECONDS = 3 * 60 * 60
TARGETS = (
    Path("/workspace/civlab_results"),
    Path("/workspace/ComfyUI/output"),
)


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log(message: str) -> None:
    print(f"{timestamp()} [cleanup] {message}", flush=True)


def remove_path(path: Path) -> None:
    if not os.path.lexists(path):
        log(f"Skipping missing path: {path}")
        return

    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)

    log(f"Removed: {path}")


def run_cleanup() -> None:
    for target in TARGETS:
        remove_path(target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove generated workspace folders every 3 hours."
    )
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=DEFAULT_INTERVAL_SECONDS,
        help="Seconds between cleanup runs. Defaults to 10800 seconds.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run cleanup once and exit.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    interval_seconds = max(1, args.interval_seconds)

    while True:
        run_cleanup()

        if args.once:
            return 0

        log(f"Sleeping {interval_seconds} seconds before next cleanup.")
        time.sleep(interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
