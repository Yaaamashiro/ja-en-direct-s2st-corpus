"""Disk-indexed, input-ordered release publication; at most one shard in RAM."""
from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import time
from collections import Counter
from contextlib import ExitStack, closing
from pathlib import Path

from . import metrics
from .external import ensure_identity, is_external, write_json
from .integrity import validate_records
from .io import account_replace, file_size
from .manifest import iter_pairs, read_jsonl, stable_shard


def consolidate(config, allow_incomplete=False):
    started = time.perf_counter()
    measurements = {}
    token = metrics.ACTIVE.set(measurements)
    try:
        return _consolidate(config, allow_incomplete, started, measurements)
    finally:
        metrics.ACTIVE.reset(token)


def _consolidate(config, allow_incomplete, started, measurements):
    provenance = ensure_identity(config) if is_external(config) else None
    root = config.run.output_dir
    release = root / "manifests" / "releases"
    # Scratch index stays on runtime-local disk, never on mounted Drive.
    with tempfile.TemporaryDirectory(prefix="s2st-release-") as scratch:
        with closing(sqlite3.connect(str(Path(scratch) / "index.sqlite"))) as db:
            db.execute("PRAGMA cache_size=-2048")
            db.execute("CREATE TABLE pairs (ordinal INTEGER PRIMARY KEY, pair_id TEXT UNIQUE, shard INTEGER, input_json TEXT, qc_json TEXT)")
            db.execute("CREATE INDEX by_shard ON pairs(shard)")
            try:
                for ordinal, pair in enumerate(iter_pairs(config.run.input_jsonl, check_unique=False)):
                    db.execute("INSERT INTO pairs VALUES (?, ?, ?, ?, NULL)",
                               (ordinal, pair["pair_id"], stable_shard(pair["pair_id"], config.run.num_shards),
                                json.dumps(pair, ensure_ascii=False)))
            except sqlite3.IntegrityError as error:
                raise ValueError("duplicate pair_id in input manifest") from error
            db.commit()
            missing = []
            for index in range(config.run.num_shards):
                path = root / "manifests" / "qc" / f"shard-{index:05d}-of-{config.run.num_shards:05d}.jsonl"
                if not path.is_file():
                    missing.append(index)
                    continue
                records = read_jsonl(path)
                if provenance is not None:
                    expected = [json.loads(r[0]) for r in db.execute("SELECT input_json FROM pairs WHERE shard=? ORDER BY ordinal", (index,))]
                    if len(expected) > config.run.max_pairs_per_shard:
                        raise RuntimeError("shard exceeds run.max_pairs_per_shard")
                    validate_records(config, expected, records, provenance, complete=True)
                    del expected
                for row in records:
                    previous = db.execute("SELECT qc_json FROM pairs WHERE pair_id=?", (row["pair_id"],)).fetchone()
                    if previous is None:
                        raise RuntimeError(f"extra pair_id in QC manifests absent from input: {row['pair_id']}")
                    if previous[0] is not None:
                        raise RuntimeError("duplicate pair_id found across shard manifests")
                    db.execute("UPDATE pairs SET qc_json=? WHERE pair_id=?",
                               (json.dumps(row, ensure_ascii=False, sort_keys=True), row["pair_id"]))
                db.commit()
                del records
            if missing and not allow_incomplete:
                raise RuntimeError(f"{len(missing)} shard manifests are missing; first missing={missing[:10]}")
            missing_pairs = db.execute("SELECT count(*) FROM pairs WHERE qc_json IS NULL").fetchone()[0]
            if missing_pairs and not allow_incomplete:
                raise RuntimeError(f"{missing_pairs} input pair_id values are missing from QC manifests")
            return _publish(db, config, release, missing, missing_pairs, provenance, started, measurements)


def _publish(db, config, release, missing, missing_pairs, provenance, started, measurements):
    names = ("all", "accepted", "train", "dev", "test")
    release.mkdir(parents=True, exist_ok=True)
    splits = {s: {"total": 0, "accepted": 0, "rejected": 0, "acceptance_rate": 0.0,
                  "ja_hours": 0.0, "en_hours": 0.0} for s in ("train", "dev", "test")}
    accumulator = metrics.AttemptMetrics()
    corpora = Counter()
    total = accepted = 0
    try:
        with ExitStack() as stack:
            handles = {name: stack.enter_context((release / f"{name}.jsonl.tmp").open("w", encoding="utf-8", newline="\n")) for name in names}
            for (encoded,) in db.execute("SELECT qc_json FROM pairs WHERE qc_json IS NOT NULL ORDER BY ordinal"):
                row = json.loads(encoded)
                split = row.get("split")
                if split not in splits:
                    raise RuntimeError(f"invalid release split: {split}")
                handles["all"].write(encoded + "\n")
                total += 1
                counts = splits[split]
                counts["total"] += 1
                if row.get("qc_status") == "accepted":
                    handles["accepted"].write(encoded + "\n")
                    handles[split].write(encoded + "\n")
                    accepted += 1
                    counts["accepted"] += 1
                    corpora[row.get("corpus", "unknown")] += 1
                    for lang in ("ja", "en"):
                        counts[f"{lang}_hours"] += float(row[f"{lang}_duration"]) / 3600
                else:
                    counts["rejected"] += 1
                accumulator.add(row)
            for handle in handles.values():
                handle.flush()
                os.fsync(handle.fileno())
        for counts in splits.values():
            counts["acceptance_rate"] = counts["accepted"] / counts["total"] if counts["total"] else 0.0
        summary = {"complete": not missing and missing_pairs == 0, "release_order": "input_manifest",
                   "missing_shards": missing, "missing_pair_count": missing_pairs,
                   "total_pairs": total, "accepted_pairs": accepted, "rejected_pairs": total - accepted,
                   "acceptance_rate": accepted / total if total else 0.0,
                   "accepted_by_corpus": dict(sorted(corpora.items())), **splits, **accumulator.result()}
        for lang in ("ja", "en"):
            summary[f"{lang}_hours"] = sum(s[f"{lang}_hours"] for s in splits.values())
            for split in splits:
                summary[f"{split}_{lang}_hours"] = splits[split][f"{lang}_hours"]
        invocations = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((config.run.output_dir / "reports" / "invocations").glob("*.json"))]
        wall = sum(r["full_pipeline_wall_seconds"] for r in invocations)
        generated = sum(s["generated_seconds"] for s in summary["attempt_metrics"].values())
        if provenance is not None:
            summary["provenance"] = provenance
        summary.update(invocations=invocations, full_pipeline_wall_seconds=wall,
                       asr_phase_wall_seconds=sum(r.get("asr_phase_seconds", 0) for r in invocations),
                       tts_phase_wall_seconds=sum(r.get("tts_phase_seconds", 0) for r in invocations),
                       pairs_per_second=total / wall if wall else None,
                       generated_audio_hours_per_wall_hour=generated / wall if wall else None,
                       consolidation_seconds=time.perf_counter() - started,
                       timing_note="Invocation wall includes load/I/O. Hard-killed invocations may lack wall measurements. RTF excludes load and WAV writes.")
        selection_path = config.run.input_jsonl.with_suffix(".selection.json")
        if selection_path.is_file():
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            selected_ids = [r[0] for r in db.execute("SELECT pair_id FROM pairs ORDER BY ordinal")]
            if selection.get("selected_pair_ids") != selected_ids:
                raise RuntimeError("benchmark selection provenance does not match input IDs")
            if provenance is not None and selection.get("source_manifest_sha256") != provenance["input"]["manifest_sha256"]:
                raise RuntimeError("benchmark selection source SHA256 mismatch")
            summary["benchmark_selection"] = selection
        for key in ("checkpoint_writes", "checkpoint_write_seconds", "directory_scan_count", "directory_scan_seconds", "current_run_written_bytes"):
            summary[key] = sum(r.get(key, 0) for r in invocations) + measurements.get(key, 0)
        # Old files remain valid while staging. Invalidate the old completion
        # marker before the first replacement; publish the new marker LAST.
        marker = release / "summary.json"
        previous_marker_bytes = file_size(marker)
        marker.unlink(missing_ok=True)
        account_replace(marker, previous_marker_bytes)
        for name in names:
            path = release / f"{name}.jsonl"
            previous_bytes = file_size(path)
            path.with_suffix(".jsonl.tmp").replace(path)
            account_replace(path, previous_bytes)
        write_json(marker, summary)
        return summary
    finally:
        for name in names:
            (release / f"{name}.jsonl.tmp").unlink(missing_ok=True)
