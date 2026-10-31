import json
import sys
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from s2st_corpus.config import load_config
from s2st_corpus.audio import sha256_file
from s2st_corpus import external, runner, kokoro_backend as kb
from s2st_corpus.manifest import write_checkpoint, read_jsonl
from s2st_corpus.smoke import select_smoke_pairs, smoke_app_config
from s2st_corpus.integrity import validate_records


@pytest.fixture
def profile(tmp_path, monkeypatch):
    monkeypatch.delenv("S2ST_TTS_BATCH_SIZE", raising=False)
    monkeypatch.delenv("S2ST_ASR_BATCH_SIZE", raising=False)
    c = load_config(Path("configs/production-kokoro-jparacrawl.yaml"))
    rows = [{"pair_id": f"p{i}", "corpus": "jparacrawl_v3_filtered", "split": "train" if i == 0 else "dev",
             "ja_text": "ＡＢＣ　そのまま？", "en_text": "Ａ  test?!", "length_bin": 2,
             "bicleaner_score": .95, "source_line": i+20} for i in range(2)]
    path = tmp_path / "manifest.jsonl"
    write_checkpoint(path, rows)
    summary = tmp_path / "summary.json"
    external.write_json(summary, {"complete": True, "corpus": "jparacrawl_v3_filtered", "total_pairs": 2,
                                 "train_pairs": 1, "dev_pairs": 1, "test_pairs": 0, "output_sha256": sha256_file(path)})
    c = replace(c, prepare=replace(c.prepare, manifest=path, summary=summary, expected_total_pairs=2,
                                  expected_train_pairs=1, expected_dev_pairs=1),
                run=replace(c.run, input_jsonl=path, output_dir=tmp_path / "out", num_shards=1),
                smoke=replace(c.smoke, input_jsonl=tmp_path / "smoke-input.jsonl", output_dir=tmp_path / "smoke-out", pair_count=2))
    monkeypatch.setattr(external, "version", lambda name: "0.9.4")
    monkeypatch.setattr(kb, "dictionary_identity", lambda: {"version": "test-dictionary", "sha256": {}})
    return c, rows


def test_external_valid_exact_text_and_metadata(profile):
    c, rows = profile
    assert external.validate_external(c)["total_pairs"] == 2
    from s2st_corpus.corpora import prepare_corpora
    assert prepare_corpora(c)["manifest_sha256"] == sha256_file(c.run.input_jsonl)
    for p in rows:
        r = runner._new_record(p, preserve_text=True)
        assert all(r[k] == v for k, v in p.items())
        assert r["ja_tts_text"] == p["ja_text"]
        assert r["en_tts_text"] == p["en_text"]
    assert runner._new_record(rows[0])["ja_tts_text"] != rows[0]["ja_text"]


@pytest.mark.parametrize("mutation", ["sha", "complete", "count", "summary_corpus", "corpus", "duplicate", "ja", "en", "split", "length_bin", "unsafe_id"])
def test_external_invalid(profile, mutation):
    c, rows = profile
    summary = json.loads(c.prepare.summary.read_text())
    if mutation == "sha": summary["output_sha256"] = "0" * 64
    elif mutation == "complete": summary["complete"] = False
    elif mutation == "count": summary["total_pairs"] = 3
    elif mutation == "summary_corpus": summary["corpus"] = "other"
    else:
        if mutation == "corpus": rows[0]["corpus"] = "other"
        elif mutation == "duplicate": rows[1]["pair_id"] = rows[0]["pair_id"]
        elif mutation in {"ja", "en"}: rows[0][f"{mutation}_text"] = "  "
        elif mutation == "split": rows[0]["split"] = "invalid"
        elif mutation == "length_bin": rows[0]["length_bin"] = "2"
        elif mutation == "unsafe_id": rows[0]["pair_id"] = "../escape"
        write_checkpoint(c.run.input_jsonl, rows)
        summary["output_sha256"] = sha256_file(c.run.input_jsonl)
    external.write_json(c.prepare.summary, summary)
    with pytest.raises(ValueError): external.validate_external(c)


def test_manifest_counts_independent_of_summary(profile):
    c, rows = profile
    rows[1]["split"] = "train"
    write_checkpoint(c.run.input_jsonl, rows)
    summary = json.loads(c.prepare.summary.read_text())
    summary["output_sha256"] = sha256_file(c.run.input_jsonl)
    external.write_json(c.prepare.summary, summary)
    with pytest.raises(ValueError, match="counts mismatch"): external.validate_external(c)


def test_smoke_no_filtering_and_subset_identity(profile):
    c, rows = profile
    assert select_smoke_pairs(c.run.input_jsonl, c.smoke.input_jsonl, 2, external=True) == rows
    sc = smoke_app_config(c)
    identity = external.ensure_identity(sc)
    assert identity["input"]["total_pairs"] == 2
    rows[0]["ja_text"] = "changed"
    write_checkpoint(c.smoke.input_jsonl, rows)
    with pytest.raises(ValueError, match="differs"): external.run_identity(sc)


def test_voice_and_revision_cannot_change_mid_run(profile):
    c, _ = profile
    external.ensure_identity(c)
    assert external.ensure_identity(c)
    for change in ({"ja_voice": "jm_kumo"}, {"en_voice": "am_adam"}, {"speed": 1.1}, {"revision": "0"*40}):
        with pytest.raises(RuntimeError, match="provenance mismatch"):
            external.ensure_identity(replace(c, tts=replace(c.tts, **change)))


def test_device_no_silent_fallback():
    t = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    assert kb.resolve_device("auto", t) == kb.resolve_device("cpu", t) == "cpu"
    with pytest.raises(RuntimeError, match="unavailable"): kb.resolve_device("cuda", t)


def test_shared_model_fixed_voices_chunks_and_empty(profile, monkeypatch, tmp_path):
    c, _ = profile
    created = []
    calls = []
    class Model:
        def __init__(self, **kwargs): created.append(self)
        def to(self, device): return self
        def eval(self): return self
    class Pipeline:
        def __init__(self, **kwargs): self.model = kwargs["model"]
        def __call__(self, text, **kwargs):
            calls.append((text, kwargs))
            return [SimpleNamespace(audio=None), SimpleNamespace(audio=np.array([1., 2.])),
                    SimpleNamespace(audio=np.array([])), SimpleNamespace(audio=np.array([3., 4.]))]
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False),
                            load=lambda path, **kw: Path(path).stem, inference_mode=nullcontext)
    monkeypatch.setitem(sys.modules, "kokoro", SimpleNamespace(KModel=Model, KPipeline=Pipeline))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(kb, "version", lambda n: "0.9.4")
    monkeypatch.setattr(kb, "download_assets", lambda *args: tmp_path)
    backend = kb.KokoroBackend(replace(c.tts, device="cpu"))
    for lang, text, voice in (("ja", "そのまま", "jf_alpha"), ("en", "Exact text", "af_heart")):
        audio, rate = backend.generate(text, lang)
        assert rate == 24000
        np.testing.assert_array_equal(audio, [1,2,3,4])
        assert calls[-1] == (text, {"voice": voice, "speed": 1.0})
    assert len(created) == 1
    assert backend.pipelines["ja"].model is backend.pipelines["en"].model
    backend.pipelines["ja"] = lambda *a, **kw: [SimpleNamespace(audio=None)]
    with pytest.raises(ValueError, match="empty_audio"): backend.generate("test", "ja")


def test_real_audio_conversion_checkpoint_and_corruption(profile):
    c, rows = profile
    c = replace(c, run=replace(c.run, save_native=True))
    identity = external.ensure_identity(c)
    record = runner._new_record(rows[0], preserve_text=True)
    record["provenance"] = identity
    backend = SimpleNamespace(generate=lambda text, lang: (np.sin(np.arange(24000)*.02).astype("float32")*.1, 24000))
    for lang in ("ja", "en"):
        a = runner._generate_attempt(backend, None, c, "shard-00000-of-00001", record, lang, 0)
        assert a["generation_status"] == "ok"
        info = sf.info(a["wav_16k"])
        assert (info.samplerate, info.channels, info.subtype, info.frames) == (16000, 1, "PCM_16", 16000)
        assert sf.info(a["wav_native"]).samplerate == 24000
        record["attempts"][lang].append(a)
    validate_records(c, [rows[0]], [record], identity)
    saved = Path(record["attempts"]["ja"][0]["wav_16k"])
    saved.write_bytes(b"broken")
    with pytest.raises(RuntimeError, match="invalid saved WAV"):
        validate_records(c, [rows[0]], [record], identity)


def test_checkpoint_resume_skips_generated_audio(profile, monkeypatch):
    c, rows = profile
    provenance = external.ensure_identity(c)
    record = runner._new_record(rows[0], preserve_text=True)
    record["provenance"] = provenance
    model = SimpleNamespace(generate=lambda text, lang: (np.ones(24000, dtype="float32")*.1, 24000))
    for lang in ("ja", "en"):
        record["attempts"][lang].append(runner._generate_attempt(model, None, c, "shard-00000-of-00001", record, lang, 0))
    checkpoint = c.run.output_dir / "checkpoint.jsonl"
    write_checkpoint(checkpoint, [record])
    restored = list(read_jsonl(checkpoint))
    validate_records(c, [rows[0]], restored, provenance)
    monkeypatch.setattr(kb, "KokoroBackend", lambda *a: pytest.fail("valid audio must not load TTS"))
    runner._generate_missing(c, "shard-00000-of-00001", restored, 0, {(rows[0]["pair_id"], l) for l in ("ja", "en")}, checkpoint)


def test_legacy_profiles_load_unchanged():
    for name in ("production-qwen17b.yaml", "colab-pro.yaml"):
        c = load_config(Path("configs") / name)
        assert c.tts.backend == "qwen"
        assert not external.is_external(c)


def test_interrupted_shard_resumes_qc_without_tts(profile, monkeypatch):
    c, rows = profile
    # Exercise orchestration with actual WAV I/O and deterministic fake inference.
    # No CUDA or model downloads are required by the test suite.
    torch = SimpleNamespace(float16="float16", float32="float32", bfloat16="bfloat16",
                            cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(runner, "gpu_preflight", lambda c: {"gpu": "test"})
    monkeypatch.setattr(runner, "_enforce_output_limit", lambda c: None)
    loads = []
    def make_backend(config):
        loads.append(True)
        return SimpleNamespace(generate=lambda text, lang: (np.ones(24000, dtype="float32")*.05, 24000))
    monkeypatch.setattr(kb, "KokoroBackend", make_backend)
    def interrupted(*a, **kw): raise RuntimeError("simulated disconnect")
    monkeypatch.setattr(runner, "_qc_unchecked", interrupted)
    with pytest.raises(RuntimeError, match="disconnect"):
        runner.run_shard(c, 0)
    assert len(loads) == 1
    assert not (c.run.output_dir / "manifests/qc/shard-00000-of-00001.jsonl").exists()
    def finish_qc(config, records, attempt, checkpoint):
        for row in records:
            for lang in ("ja", "en"):
                for a in row["attempts"][lang]:
                    a.update(qc_pass=True, qc_reasons=[], metric=0.0, asr_text=row[f"{lang}_text"],
                             **{("cer" if lang == "ja" else "wer"): 0.0})
        write_checkpoint(checkpoint, records)
    monkeypatch.setattr(runner, "_qc_unchecked", finish_qc)
    path = runner.run_shard(c, 0)
    assert len(loads) == 1  # no TTS load on resume
    summary = runner.consolidate(c)
    assert summary["complete"] and summary["accepted_pairs"] == 2
    assert summary["attempt_metrics"]["ja"]["generated_seconds"] == 2
    assert len(summary["invocations"]) == 2
    assert summary["invocations"][0]["success"] is False
    # Finalized shard cannot silently reuse a corrupt WAV either.
    final = list(read_jsonl(path))
    Path(final[0]["ja_wav_16k"]).write_bytes(b"")
    with pytest.raises(RuntimeError, match="invalid saved WAV"):
        runner.run_shard(c, 0)
    with pytest.raises(RuntimeError, match="invalid saved WAV"):
        runner.consolidate(c)


def test_production_gate_and_notebook_syntax(profile, monkeypatch):
    from s2st_corpus import cli
    monkeypatch.setattr(cli, "load_config", lambda p: profile[0])
    monkeypatch.setattr(sys, "argv", ["cli", "run-remaining-shards"])
    with pytest.raises(RuntimeError, match="allow-production"):
        cli.main()
    nb = json.loads(Path("notebooks/colab_kokoro_jparacrawl.ipynb").read_text(encoding="utf-8"))
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"cell-{i}", "exec")


def test_resume_rejects_changed_checkpoint_text(profile):
    c, rows = profile
    identity = external.ensure_identity(c)
    record = runner._new_record(rows[0], preserve_text=True)
    record["provenance"] = identity
    record["ja_tts_text"] = "rewritten"
    with pytest.raises(RuntimeError, match="text mismatch"):
        validate_records(c, [rows[0]], [record], identity)
