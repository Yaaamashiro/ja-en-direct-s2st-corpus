import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from s2st_corpus import external, kokoro_backend, kokoro_parallel as kp, runner
from s2st_corpus.config import load_config
from s2st_corpus.audio import atomic_write_wav, sha256_file
from s2st_corpus.manifest import read_jsonl


class FakeBackend:
    """Top-level pickleable stand-in; workers still perform real WAV I/O."""
    def __init__(self, config, **kwargs):
        self.config = config
        assert kwargs["local_only"] and kwargs["preload_voices"]

    def generate(self, text, language):
        if text == "CUDA":
            raise RuntimeError("CUDA out of memory")
        if text == "CRASH":
            os._exit(17)
        if text == "EMPTY":
            raise ValueError("empty_audio: mock empty output")
        time.sleep(.005 * (len(text) % 3))
        return np.ones(24000, dtype="float32") * .02, 24000


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.delenv("S2ST_TTS_BATCH_SIZE", raising=False)
    c = load_config(Path("configs/production-kokoro-jparacrawl.yaml"))
    return replace(c, run=replace(c.run, output_dir=tmp_path, checkpoint_every=100),
                   tts=replace(c.tts, device="cpu"))


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_spawn_twenty_utterances_metadata_only_and_atomic_wav(config, workers, tmp_path):
    c = replace(config, tts=replace(config.tts, worker_count=workers))
    tasks = [(f"p{i}", lang, 0, f"Exact  text{i}?!") for i in range(10) for lang in ("ja", "en")]
    events = list(kp.iter_worker_results(c, "test", tasks, tmp_path, backend_factory=FakeBackend))
    assert len(events) == len({tuple(e["key"]) for e in events}) == 20
    for event in events:
        pid, lang, attempt = event["key"]
        result = event["attempt_result"]
        assert result["generation_status"] == "ok"
        assert result["voice"] == getattr(c.tts, f"{lang}_voice")
        assert result["worker_count"] == workers
        assert result["duration"] == 1.0
        assert result["wav_16k"] == str(runner._paths(c, "test", pid, lang, attempt)[1])
        info = sf.info(result["wav_16k"])
        assert (info.samplerate, info.channels, info.subtype) == (16000, 1, "PCM_16")
        assert sha256_file(Path(result["wav_16k"])) == result["sha256"]
        assert "waveform" not in event and "audio" not in event
    assert not list(tmp_path.rglob("*.tmp.wav"))


@pytest.mark.parametrize("text", ["CUDA", "CRASH"])
def test_worker_fatal_and_crash_stop_pool(config, text, tmp_path):
    c = replace(config, tts=replace(config.tts, worker_count=2))
    with pytest.raises(RuntimeError, match="fatal|crashed"):
        list(kp.iter_worker_results(c, "test", [("p0", "ja", 0, text)], tmp_path, backend_factory=FakeBackend))


def test_nonfatal_generation_failure_is_metadata(config, tmp_path):
    tasks = [("p0", "ja", 0, "EMPTY"), ("p1", "en", 0, "text")]
    events = list(kp.iter_worker_results(config, "test", tasks, tmp_path, backend_factory=FakeBackend))
    results = {e["key"][0]: e["attempt_result"] for e in events}
    assert results["p0"]["generation_status"] == "failed"
    assert "empty_audio" in results["p0"]["generation_error"]
    assert results["p1"]["generation_status"] == "ok"


def test_parallel_worker_writes_update_parent_storage_budget(config, tmp_path, monkeypatch):
    from s2st_corpus import io, metrics
    c = replace(config, tts=replace(config.tts, worker_count=2))
    io.reset_budgets()
    io.tracked_directory_size(tmp_path)
    monkeypatch.setattr(kokoro_backend, "materialize_assets", lambda c: tmp_path)
    original = kp.iter_worker_results
    monkeypatch.setattr(kp, "iter_worker_results", lambda *a, **kw: original(*a, **kw, backend_factory=FakeBackend))
    records = [runner._new_record({"pair_id":"p0", "ja_text":"文", "en_text":"text"}, True)]
    measurement = {}
    token = metrics.ACTIVE.set(measurement)
    try:
        runner._generate_missing(c, "test", records, 0, {("p0","ja"), ("p0","en")}, tmp_path / "cp.jsonl")
        assert io.tracked_directory_size(tmp_path) == io.directory_size(tmp_path)
        assert measurement["tts_generated_utterances"] == 2
        assert measurement["tts_generated_audio_seconds"] == 2.
        assert len(measurement["tts_worker_per_process_peak"]) == 2
    finally:
        metrics.ACTIVE.reset(token)
        io.reset_budgets()


def test_duplicate_task_rejected_before_spawn(config, tmp_path):
    task = ("p0", "ja", 0, "text")
    with pytest.raises(ValueError, match="duplicate"):
        list(kp.iter_worker_results(config, "test", [task, task], tmp_path, backend_factory=FakeBackend))


def test_kokoro_batch_override_never_dispatches_to_qwen(config, monkeypatch, tmp_path):
    c = replace(config, tts=replace(config.tts, batch_size=4))
    row = runner._new_record({"pair_id":"p0", "ja_text":"文", "en_text":"text"}, True)
    monkeypatch.setattr(runner, "_generate_missing_batched", lambda *a: pytest.fail("Kokoro must never use Qwen batching"))
    with pytest.raises(ValueError, match="worker_count"):
        runner._generate_missing(c, "test", [row], 0, {("p0","ja")}, tmp_path / "cp.jsonl")


def test_fork_context_is_rejected(config, tmp_path):
    from types import SimpleNamespace
    with pytest.raises(ValueError, match="requires spawn"):
        list(kp.iter_worker_results(config, "test", [("p0", "ja", 0, "text")], tmp_path,
                                    context=SimpleNamespace(get_start_method=lambda: "fork")))


def event_for(config, pid, lang):
    return {"key": (pid, lang, 0), "worker_id": 1, "model_load_seconds": .1,
            "files": [], "memory": {"allocated_gib": 1., "reserved_gib": 2.},
            "attempt_result": {"attempt": 0, "generation_status": "ok", "duration": 1.,
                               "worker_count": config.tts.worker_count,
                               "wav_16k": str(runner._paths(config, "test", pid, lang, 0)[1])}}


def test_parallel_completion_order_parent_checkpoint_and_resume(config, monkeypatch, tmp_path):
    c = replace(config, tts=replace(config.tts, worker_count=2))
    records = [runner._new_record({"pair_id": f"p{i}", "ja_text": "文", "en_text": "text"}, True) for i in range(60)]
    missing = {(r["pair_id"], lang) for r in records for lang in ("ja", "en")}
    calls, snapshots = [], []
    monkeypatch.setattr(kokoro_backend, "materialize_assets", lambda c: tmp_path)
    def results(config, shard, tasks, root, **kwargs):
        calls.extend(tasks)
        for pid, lang, attempt, text in reversed(tasks):
            yield event_for(config, pid, lang)
    monkeypatch.setattr(kp, "iter_worker_results", results)
    checkpoint = tmp_path / "checkpoint.jsonl"
    original = runner.write_checkpoint
    def write(path, rows):
        snapshots.append(sum(len(a) for r in rows for a in r["attempts"].values()))
        original(path, rows)
    monkeypatch.setattr(runner, "write_checkpoint", write)
    runner._generate_missing(c, "test", records, 0, missing, checkpoint)
    assert snapshots == [100, 120]
    assert len(calls) == len({t[:3] for t in calls}) == 120
    assert all(r["attempts"][lang][0]["wav_16k"] == str(runner._paths(c, "test", r["pair_id"], lang, 0)[1]) for r in records for lang in ("ja", "en"))
    restored = read_jsonl(checkpoint)
    monkeypatch.setattr(kokoro_backend, "materialize_assets", lambda c: pytest.fail("resume must skip model loading"))
    runner._generate_missing(c, "test", restored, 0, missing, checkpoint)
    assert len(calls) == 120


def test_parallel_queue_is_longest_first_with_deterministic_ties(config, monkeypatch, tmp_path):
    c = replace(config, tts=replace(config.tts, worker_count=2))
    records = [runner._new_record({"pair_id":"p0", "ja_text":"短", "en_text":"longest text"}, True),
               runner._new_record({"pair_id":"p1", "ja_text":"同じ", "en_text":"ab"}, True)]
    missing = {(r["pair_id"], lang) for r in records for lang in ("ja", "en")}
    captured = []
    monkeypatch.setattr(kokoro_backend, "materialize_assets", lambda c: tmp_path)
    def results(config, shard, tasks, root, **kwargs):
        captured.extend(tasks)
        for pid, lang, _, text in reversed(tasks):
            yield event_for(config, pid, lang)
    monkeypatch.setattr(kp, "iter_worker_results", results)
    runner._generate_missing(c, "test", records, 0, missing, tmp_path / "cp.jsonl")
    assert [t[:2] for t in captured] == [("p0","en"), ("p1","en"), ("p1","ja"), ("p0","ja")]
    expected = {(r["pair_id"], lang): r[f"{lang}_tts_text"] for r in records for lang in ("ja","en")}
    assert len(captured) == len(expected) and all(text == expected[pid, lang] for pid, lang, _, text in captured)
    assert all(len(r["attempts"][lang]) == 1 for r in records for lang in ("ja","en"))


def test_parent_flushes_received_results_on_cuda_failure(config, monkeypatch, tmp_path):
    c = replace(config, tts=replace(config.tts, worker_count=2))
    records = [runner._new_record({"pair_id": "p0", "ja_text": "文", "en_text": "text"}, True)]
    monkeypatch.setattr(kokoro_backend, "materialize_assets", lambda c: tmp_path)
    def fail(config, *args, **kwargs):
        yield event_for(config, "p0", "ja")
        raise RuntimeError("worker CUDA out of memory")
    monkeypatch.setattr(kp, "iter_worker_results", fail)
    checkpoint = tmp_path / "cp.jsonl"
    with pytest.raises(RuntimeError, match="CUDA"):
        runner._generate_missing(c, "test", records, 0, {("p0", "ja"), ("p0", "en")}, checkpoint)
    saved = read_jsonl(checkpoint)
    assert len(saved[0]["attempts"]["ja"]) == 1 and saved[0]["attempts"]["en"] == []


def test_orphan_is_regenerated(config, tmp_path):
    path = runner._paths(config, "test", "p0", "ja", 0)[1]
    atomic_write_wav(path, np.ones(3200, dtype="float32"), 16000, "PCM_16")
    events = list(kp.iter_worker_results(config, "test", [("p0", "ja", 0, "text")], tmp_path, backend_factory=FakeBackend))
    assert events[0]["attempt_result"]["duration"] == 1.0
    assert events[0]["files"][0]["previous_bytes"] > 0


def test_worker_count_is_not_semantic_provenance(config, monkeypatch, tmp_path):
    from dataclasses import asdict
    c = config
    monkeypatch.setattr(external, "validate_external", lambda c: {"manifest_sha256": "abc"})
    monkeypatch.setattr("s2st_corpus.input_index.get_index", lambda *a: type("Index", (), {"digest": "abc"})())
    c = replace(c, run=replace(c.run, input_jsonl=c.prepare.manifest))
    monkeypatch.setattr(external, "version", lambda name: "0.9.4")
    monkeypatch.setattr(kokoro_backend, "dictionary_identity", lambda: {})
    first = external.ensure_identity(c)
    assert first["initial_performance"]["kokoro_worker_count"] == 1
    assert external.ensure_identity(replace(c, tts=replace(c.tts, worker_count=4))) == first
    assert "worker_count" not in first["tts"]
    with pytest.raises(RuntimeError, match="provenance mismatch"):
        external.ensure_identity(replace(c, tts=replace(c.tts, ja_voice="jm_kumo")))
