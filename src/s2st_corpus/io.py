from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(entry.stat().st_size for entry in path.rglob("*") if entry.is_file())


def ensure_disk_budget(paths: list[Path], minimum_free_gb: float) -> None:
    checked: set[str] = set()
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
        resolved = str(path.resolve())
        if resolved in checked:
            continue
        checked.add(resolved)
        free = shutil.disk_usage(path).free / (1024**3)
        if free < minimum_free_gb:
            raise RuntimeError(
                f"Insufficient free space at {path}: {free:.1f} GiB available, "
                f"{minimum_free_gb:.1f} GiB required"
            )


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
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
