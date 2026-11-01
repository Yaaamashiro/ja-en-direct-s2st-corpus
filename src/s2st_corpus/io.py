from __future__ import annotations

import json
import os
import shutil
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from . import metrics


# One writer per output directory. Baselines last for this CLI process only.
_BUDGETS: dict[Path, int] = {}


def reset_budgets() -> None:
    _BUDGETS.clear()


def tracked_directory_size(path: Path) -> int:
    key = path.resolve()
    if key not in _BUDGETS:
        started = time.perf_counter()
        _BUDGETS[key] = directory_size(key)
        metrics.add_time("directory_scan_seconds", time.perf_counter() - started)
        metrics.add_time("directory_scan_count", 1)
    return _BUDGETS[key]


def file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def account_replace(path: Path, previous_bytes: int) -> None:
    new_size = file_size(path)
    account_worker_write(path, previous_bytes, new_size, new_size)


def account_worker_write(path, previous_bytes, new_size, written_bytes):
    resolved = Path(path).resolve()
    delta = new_size - previous_bytes
    for root in _BUDGETS:
        if resolved.is_relative_to(root):
            _BUDGETS[root] += delta
    metrics.add_time("current_run_written_bytes", written_bytes)


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(entry.stat().st_size for entry in path.rglob("*") if entry.is_file())


def ensure_reusable_budget(path: Path, required_total_gib: float) -> None:
    path.mkdir(parents=True, exist_ok=True)
    used = tracked_directory_size(path) / (1024**3)
    free = shutil.disk_usage(path).free / (1024**3)
    available_for_run = used + free
    if available_for_run < required_total_gib:
        raise RuntimeError(
            f"Insufficient run capacity at {path}: {available_for_run:.1f} GiB "
            f"(existing {used:.1f} + free {free:.1f}) available, "
            f"{required_total_gib:.1f} GiB required"
        )


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    previous_bytes = file_size(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    account_replace(path, previous_bytes)
