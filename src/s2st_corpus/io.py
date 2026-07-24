from __future__ import annotations

import json
import os
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(entry.stat().st_size for entry in path.rglob("*") if entry.is_file())


def ensure_reusable_budget(path: Path, required_total_gib: float) -> None:
    path.mkdir(parents=True, exist_ok=True)
    used = directory_size(path) / (1024**3)
    free = shutil.disk_usage(path).free / (1024**3)
    available_for_run = used + free
    if available_for_run < required_total_gib:
        raise RuntimeError(
            f"Insufficient run capacity at {path}: {available_for_run:.1f} GiB "
            f"(existing {used:.1f} + free {free:.1f}) available, "
            f"{required_total_gib:.1f} GiB required"
        )


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
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
