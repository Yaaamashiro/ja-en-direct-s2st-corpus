import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from s2st_corpus import runner, io, metrics
from s2st_corpus.audio import atomic_write_wav, sha256_file
from s2st_corpus.config import load_config
from s2st_corpus.manifest import write_checkpoint, read_jsonl
from s2st_corpus.smoke import select_benchmark_pairs, BENCHMARK_QUOTAS


@pytest.fixture
def config(tmp_path):
    c = load_config(Path("configs/production-qwen17b.yaml"))
    return replace(c, run=replace(c.run, output_dir=tmp_path, checkpoint_every=100),
                   asr=replace(c.asr, batch_size=8))


def pending_audio(count=130):
    rows, pending = [], []
    for i in range(count):
        row = {"pair_id": f"p{i}", "attempts": {}}
        for lang in ("ja", "en"):
            a = {"generation_status": "ok", "duration": 1, "wav_16k": f"{i}-{lang}", "attempt": 0}
            row["attempts"][lang] = [a]
            pending.append((row, lang, a))
        rows.append(row)
    return rows, pending


@pytest.mark.parametrize("interrupt", [False, True])
def test_qc_checkpoint_interval_and_exception(config, monkeypatch, tmp_path, interrupt):
    records, pending = pending_audio()
    batches, writes = [], []
    def transcriber(paths, **kwargs):
        batches.append(len(paths))
        if interrupt and len(batches) == 3:
            raise RuntimeError("simulated ASR error")
        return [{"text": p} for p in paths]
    def evaluate(c, r, lang, a, *args):
        a["qc_pass"] = True
    monkeypatch.setattr(runner, "_evaluate_attempt", evaluate)
    monkeypatch.setattr(runner, "_load_whisper", lambda c: (transcriber, None, None, None))
    monkeypatch.setattr(runner, "_release_cuda", lambda t: None)
    monkeypatch.setattr(runner, "write_checkpoint", lambda *a: writes.append(sum("qc_pass" in x for _, _, x in pending)))
    if interrupt:
        with pytest.raises(RuntimeError, match="ASR error"):
            runner._qc_batched(config, records, pending, tmp_path / "cp.jsonl", None)
        assert writes == [0, 16]
    else:
        runner._qc_batched(config, records, pending, tmp_path / "cp.jsonl", None)
        assert writes == [0, 100, 130, 230, 260, 260]
        assert len(batches) > len(writes) * 4


def test_tts_checkpoint_unit_is_utterances(config, monkeypatch, tmp_path):
    records = [runner._new_record({"pair_id": f"p{i}", "ja_text": "文", "en_text": "text"}) for i in range(130)]
    writes = []
    monkeypatch.setattr(runner, "_load_qwen", lambda c: (None, None))
    monkeypatch.setattr(runner, "_release_cuda", lambda t: None)
    monkeypatch.setattr(runner, "_generate_attempt", lambda *a: {"attempt": 0, "generation_status": "ok"})
    monkeypatch.setattr(runner, "write_checkpoint", lambda *a: writes.append(sum(len(v) for r in records for v in r["attempts"].values())))
    runner._generate_missing(config, "shard", records, 0,
                             {(r["pair_id"], lang) for r in records for lang in ("ja", "en")}, tmp_path / "cp.jsonl")
    assert writes == [100, 200, 260]


def test_capacity_scan_once_and_incremental_replacement(config, monkeypatch):
    io.reset_budgets()
    observed = []
    original = io.directory_size
    monkeypatch.setattr(io, "directory_size", lambda p: observed.append(p) or original(p))
    timings = {}
    token = metrics.ACTIVE.set(timings)
    try:
        for _ in range(3):
            runner._enforce_output_limit(config)
        path = config.run.output_dir / "audio.wav"
        atomic_write_wav(path, np.zeros(16000, dtype=np.float32), 16000, "PCM_16")
        write_checkpoint(config.run.output_dir / "cp.jsonl", [{"pair_id": "p0"}])
        # Overwriting is a delta, not the entire new size added twice.
        atomic_write_wav(path, np.zeros(8000, dtype=np.float32), 16000, "PCM_16")
        assert io.tracked_directory_size(config.run.output_dir) == original(config.run.output_dir)
        runner._enforce_output_limit(config)
        assert len(observed) == 1
        assert timings["directory_scan_count"] == 1
        assert timings["checkpoint_writes"] == 1
        assert timings["checkpoint_write_seconds"] >= 0
        with pytest.raises(RuntimeError, match="output safety limit"):
            runner._enforce_output_limit(replace(config, run=replace(config.run, maximum_output_gib=0)))
    finally:
        metrics.ACTIVE.reset(token)
        io.reset_budgets()


def test_benchmark_stratification_reproducibility_exact_text(tmp_path):
    source, destination = tmp_path / "source.jsonl", tmp_path / "benchmark.jsonl"
    source_rows = [{"pair_id": f"b{b}-{i}", "split": "dev" if i % 7 == 0 else "train",
                    "corpus": "jparacrawl_v3_filtered", "length_bin": b,
                    "ja_text": f"Ａ　変更しない？{i}", "en_text": f"Exact  text?! {i}", "source_line": i}
                   for b in range(5) for i in range(500)]
    write_checkpoint(source, source_rows)
    selected = select_benchmark_pairs(source, destination)
    before = destination.read_bytes()
    assert len(selected) == len({r["pair_id"] for r in selected}) == 1000
    assert all(r["split"] == "train" for r in selected)
    assert {b: sum(r["length_bin"] == b for r in selected) for b in range(5)} == BENCHMARK_QUOTAS
    chosen = {r["pair_id"] for r in selected}
    assert selected == [r for r in source_rows if r["pair_id"] in chosen]
    assert select_benchmark_pairs(source, destination) == selected
    assert destination.read_bytes() == before
    meta = json.loads(destination.with_suffix(".selection.json").read_text())
    assert meta["seed"] == 42 and meta["source_manifest_sha256"] == sha256_file(source)
    assert meta["selected_pair_ids"] == [r["pair_id"] for r in selected]
    assert select_benchmark_pairs(source, destination, seed=99) != selected


def test_benchmark_fails_when_training_quota_unavailable(tmp_path):
    path = tmp_path / "tiny.jsonl"
    write_checkpoint(path, [{"pair_id": "tiny", "split": "train", "corpus": "jparacrawl_v3_filtered", "length_bin": 0, "ja_text": "文", "en_text": "text"}])
    with pytest.raises(ValueError, match="quotas"):
        select_benchmark_pairs(path, tmp_path / "benchmark.jsonl")


def test_duration_failures_by_language_and_bin():
    rows = [{"length_bin": b, "attempts": {lang: [{"generation_status": "ok", "duration": 31,
             "qc_reasons": ["duration_out_of_range"]}]}} for b, lang in ((2, "ja"), (4, "en"), (4, "ja"))]
    result = metrics.summarize(iter(rows))
    assert result["duration_failures_by_length_bin"] == {"0": 0, "1": 0, "2": 1, "3": 0, "4": 2}
    assert result["duration_failures_by_language_and_length_bin"]["en"]["4"] == 1


def test_tts_checkpoint_on_model_load_failure(config, monkeypatch, tmp_path):
    record = runner._new_record({"pair_id": "p0", "ja_text": "文", "en_text": "text"})
    def fail(c): raise RuntimeError("model load failed")
    monkeypatch.setattr(runner, "_load_qwen", fail)
    checkpoint = tmp_path / "failure.jsonl"
    with pytest.raises(RuntimeError, match="model load failed"):
        runner._generate_missing(config, "shard", [record], 0, {("p0", "ja")}, checkpoint)
    assert read_jsonl(checkpoint) == [record]
