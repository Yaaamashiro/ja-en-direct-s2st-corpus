from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .io import atomic_write_jsonl


_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,96}$")
_REQUIRED_FIELDS = ("pair_id", "corpus", "split", "ja_text", "en_text")


def stable_shard(pair_id: str, num_shards: int) -> int:
    digest = hashlib.sha256(pair_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % num_shards


def iter_pairs(path: Path) -> Iterator[dict[str, Any]]:
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            missing = [
                field
                for field in _REQUIRED_FIELDS
                if not isinstance(item.get(field), str) or not item[field].strip()
            ]
            if missing:
                raise ValueError(f"{path}:{line_number}: invalid fields: {missing}")
            pair_id = item["pair_id"]
            if not _SAFE_ID.fullmatch(pair_id):
                raise ValueError(f"{path}:{line_number}: unsafe pair_id={pair_id!r}")
            if pair_id in seen:
                raise ValueError(f"{path}:{line_number}: duplicate pair_id={pair_id!r}")
            seen.add(pair_id)
            yield item
    if not seen:
        raise ValueError(f"No input pairs found in {path}")


def pairs_for_shard(
    path: Path,
    shard_index: int,
    num_shards: int,
    max_pairs: int,
) -> list[dict[str, Any]]:
    if not 0 <= shard_index < num_shards:
        raise ValueError(
            f"shard index {shard_index} is outside [0, {num_shards - 1}]"
        )
    selected = [
        pair
        for pair in iter_pairs(path)
        if stable_shard(pair["pair_id"], num_shards) == shard_index
    ]
    if len(selected) > max_pairs:
        raise RuntimeError(
            f"shard {shard_index} has {len(selected)} pairs, exceeding "
            f"run.max_pairs_per_shard={max_pairs}; increase num_shards"
        )
    return selected


def plan_manifest(path: Path, num_shards: int) -> dict[str, Any]:
    shard_counts = [0] * num_shards
    corpus_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    total = 0
    for pair in iter_pairs(path):
        total += 1
        shard_counts[stable_shard(pair["pair_id"], num_shards)] += 1
        corpus_counts[pair["corpus"]] += 1
        split_counts[pair["split"]] += 1
    return {
        "pairs": total,
        "num_shards": num_shards,
        "empty_shards": sum(count == 0 for count in shard_counts),
        "minimum_pairs_per_shard": min(shard_counts),
        "maximum_pairs_per_shard": max(shard_counts),
        "mean_pairs_per_shard": total / num_shards,
        "corpora": dict(sorted(corpus_counts.items())),
        "splits": dict(sorted(split_counts.items())),
        "shard_counts": shard_counts,
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            rows.append(item)
    return rows


def write_checkpoint(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_write_jsonl(path, rows)


def pcm16_storage_gib(
    hours_per_language: float,
    languages: int = 2,
    sample_rate: int = 16000,
    retry_fraction: float = 0.0,
) -> float:
    raw_bytes = hours_per_language * 3600 * sample_rate * 2 * languages
    return raw_bytes * (1 + retry_fraction) / (1024**3)
