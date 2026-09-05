from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from s2st_corpus import runner
from s2st_corpus.config import load_config
from s2st_corpus.manifest import read_jsonl


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.delenv("S2ST_TTS_BATCH_SIZE", raising=False)
    monkeypatch.delenv("S2ST_ASR_BATCH_SIZE", raising=False)
    c = load_config(Path(__file__).parents[1] / "configs/production-qwen17b.yaml")
    return replace(c, run=replace(c.run, output_dir=tmp_path),
                   tts=replace(c.tts, batch_size=2), asr=replace(c.asr, batch_size=2))


def test_long_audio_is_rejected_before_asr(config):
    attempt = {"generation_status": "ok", "duration": 31, "asr_error": "old error"}
    runner._evaluate_attempt(config, {}, "en", attempt, None, None, None)
    assert attempt["qc_pass"] is False
    assert attempt["qc_reasons"] == ["duration_out_of_range"]
    assert "asr_error" not in attempt


def test_tts_batches_save_mapping_and_resume_without_generation(config, monkeypatch, tmp_path):
    records = [runner._new_record({"pair_id": str(i), "ja_text": "hello" * (i + 1),
                                  "en_text": "hello"}) for i in range(3)]
    calls = []

    class Model:
        def generate_custom_voice(self, **kwargs):
            calls.append(kwargs)
            return [np.zeros(16000, dtype=np.float32) for _ in kwargs["text"]], 16000

    monkeypatch.setattr(runner, "_load_qwen", lambda c: (Model(), None))
    monkeypatch.setattr(runner, "seed_everything", lambda *a: None)
    monkeypatch.setattr(runner, "_release_cuda", lambda *a: None)
    targets = {(r["pair_id"], "ja") for r in records}
    checkpoint = tmp_path / "checkpoint.jsonl"
    runner._generate_missing(config, "shard", records, 0, targets, checkpoint)
    assert [len(c["text"]) for c in calls] == [2, 1]
    assert all(r["attempts"]["ja"][0]["generation_status"] == "ok" for r in records)
    assert records[0]["attempts"]["ja"][0]["batch_pair_ids"] == ["0", "1"]
    saved = list(read_jsonl(checkpoint))
    runner._generate_missing(config, "shard", saved, 0, targets, checkpoint)
    assert len(calls) == 2


def test_asr_batches_keep_language_mapping_and_skip_long_audio(config, monkeypatch, tmp_path):
    pending = []
    records = []
    for lang, duration, text in [("en", 2, "one"), ("ja", 3, "two"),
                                  ("en", 1, "three"), ("en", 31, "long")]:
        attempt = {"attempt": 0, "generation_status": "ok", "duration": duration,
                   "wav_16k": text, "asr_error": "previous error"}
        record = {"pair_id": text, f"{lang}_tts_text": text, "attempts": {lang: [attempt]}}
        records.append(record)
        pending.append((record, lang, attempt))
    calls = []

    def transcriber(paths, **kwargs):
        calls.append((paths, kwargs))
        return [{"text": p} for p in paths]

    monkeypatch.setattr(runner, "_load_whisper", lambda c: (
        transcriber, SimpleNamespace(tokenizer=object()), None, None))
    monkeypatch.setattr(runner, "_release_cuda", lambda *a: None)
    runner._qc_batched(config, records, pending, tmp_path / "qc.jsonl", lambda s, **kw: s)
    assert [paths for paths, _ in calls] == [["two"], ["three", "one"]]
    assert [kw["generate_kwargs"]["language"] for _, kw in calls] == ["japanese", "english"]
    assert all(a["qc_pass"] for _, _, a in pending[:3])
    assert pending[-1][2]["qc_reasons"] == ["duration_out_of_range"]
    assert all("asr_error" not in a for _, _, a in pending)


def test_batch_config_environment_override(monkeypatch):
    path = Path(__file__).parents[1] / "configs/production-qwen17b.yaml"
    monkeypatch.setenv("S2ST_TTS_BATCH_SIZE", "4")
    monkeypatch.setenv("S2ST_ASR_BATCH_SIZE", "8")
    c = load_config(path)
    assert (c.tts.batch_size, c.asr.batch_size) == (4, 8)
    monkeypatch.setenv("S2ST_ASR_BATCH_SIZE", "0")
    with pytest.raises(ValueError, match="batch_size"):
        load_config(path)
