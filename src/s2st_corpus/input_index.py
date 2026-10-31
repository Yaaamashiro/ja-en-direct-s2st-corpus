"""Process-local, disk-backed input snapshots; never trust an index across CLI runs."""
from __future__ import annotations

import atexit
import hashlib
import json
import os
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path

from .manifest import iter_pairs, stable_shard


def file_stamp(path):
    stat = Path(path).stat()
    return (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_dev, stat.st_ino)


class InputIndex:
    def __init__(self, path, num_shards, source=None):
        self.source = Path(path).resolve()
        self.num_shards = num_shards
        self.stamp = file_stamp(self.source)
        root = Path(os.environ.get("S2ST_INPUT_INDEX_DIR", Path(tempfile.gettempdir()) / "s2st-input-index"))
        root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="session-", dir=root)
        self.path = Path(self.temporary.name) / "input.sqlite"
        self.db = None
        try:
            self.db = sqlite3.connect(str(self.path))
            self.db.execute("PRAGMA cache_size=-2048")
            self.db.execute("CREATE TABLE pairs (ordinal INTEGER PRIMARY KEY, pair_id TEXT UNIQUE, shard INTEGER, input_json TEXT)")
            self.db.execute("CREATE INDEX by_shard ON pairs(shard, ordinal)")
            digest = hashlib.sha256()
            rows = source.iter_rows() if source is not None else iter_pairs(self.source, check_unique=False, digest=digest)
            self.counts, self.corpora = Counter(), Counter()
            self.shard_counts = [0] * num_shards
            try:
                for ordinal, pair in enumerate(rows):
                    shard = stable_shard(pair["pair_id"], num_shards)
                    self.db.execute("INSERT INTO pairs VALUES (?, ?, ?, ?)",
                                    (ordinal, pair["pair_id"], shard, json.dumps(pair, ensure_ascii=False)))
                    self.counts[pair["split"]] += 1
                    self.corpora[pair["corpus"]] += 1
                    self.shard_counts[shard] += 1
            except sqlite3.IntegrityError as error:
                raise ValueError("duplicate pair_id in input manifest") from error
            self.db.commit()
            self.digest = source.digest if source is not None else digest.hexdigest()
            self.check()
        except BaseException:
            self.close()
            raise

    def check(self):
        if file_stamp(self.source) != self.stamp:
            raise ValueError(f"input manifest differs from indexed snapshot; file changed: {self.source}")

    def iter_rows(self):
        self.check()
        for (encoded,) in self.db.execute("SELECT input_json FROM pairs ORDER BY ordinal"):
            yield json.loads(encoded)

    def pairs(self, shard, maximum):
        self._check_shard(shard, maximum)
        return [json.loads(r[0]) for r in self.db.execute("SELECT input_json FROM pairs WHERE shard=? ORDER BY ordinal", (shard,))]

    def _check_shard(self, shard, maximum):
        self.check()
        if not 0 <= shard < self.num_shards:
            raise ValueError("shard index outside configured range")
        if self.shard_counts[shard] > maximum:
            raise RuntimeError(f"shard {shard} exceeds run.max_pairs_per_shard={maximum}")

    def pair_ids(self, shard, maximum):
        self._check_shard(shard, maximum)
        return {r[0] for r in self.db.execute("SELECT pair_id FROM pairs WHERE shard=?", (shard,))}

    def lookup(self, pair_id):
        # Bulk callers check the source before and after the lookup loop.
        row = self.db.execute("SELECT input_json FROM pairs WHERE pair_id=?", (pair_id,)).fetchone()
        return json.loads(row[0]) if row is not None else None

    def plan(self):
        self.check()
        total = sum(self.shard_counts)
        return {"pairs": total, "num_shards": self.num_shards,
                "empty_shards": sum(n == 0 for n in self.shard_counts),
                "minimum_pairs_per_shard": min(self.shard_counts),
                "maximum_pairs_per_shard": max(self.shard_counts),
                "mean_pairs_per_shard": total / self.num_shards,
                "corpora": dict(sorted(self.corpora.items())), "splits": dict(sorted(self.counts.items())),
                "shard_counts": list(self.shard_counts)}

    def close(self):
        if self.db is not None:
            self.db.close()
        self.temporary.cleanup()


_INDEXES = {}


def get_index(path, num_shards):
    key = (Path(path).resolve(), num_shards)
    if key not in _INDEXES:
        # Smoke changes shard counts. Repartition the local index, not Drive.
        source = next((index for (p, _), index in _INDEXES.items() if p == key[0]), None)
        if source is not None:
            source.check()
        _INDEXES[key] = InputIndex(path, num_shards, source)
    _INDEXES[key].check()
    return _INDEXES[key]


def cached_index(path):
    resolved = Path(path).resolve()
    index = next((index for (p, _), index in _INDEXES.items() if p == resolved), None)
    if index is not None:
        index.check()
    return index


def reset_indexes():
    for index in _INDEXES.values():
        index.close()
    _INDEXES.clear()


atexit.register(reset_indexes)


def index_for_config(config):
    from .external import validate_external
    validate_external(config)
    return get_index(config.run.input_jsonl, config.run.num_shards)


def pairs_for_config(config, shard):
    from .external import is_external
    if is_external(config):
        return index_for_config(config).pairs(shard, config.run.max_pairs_per_shard)
    from .manifest import pairs_for_shard
    return pairs_for_shard(config.run.input_jsonl, shard, config.run.num_shards, config.run.max_pairs_per_shard)
