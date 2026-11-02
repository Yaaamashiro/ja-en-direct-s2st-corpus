import json
import threading
from dataclasses import replace, asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from s2st_corpus import performance as perf, runner, kokoro_backend
from s2st_corpus.config import load_config
from s2st_corpus.audio import sha256_file
from s2st_corpus.external import write_json
from s2st_corpus.manifest import write_checkpoint, read_jsonl


@pytest.fixture
def profile(tmp_path, monkeypatch):
    monkeypatch.delenv("S2ST_TTS_BATCH_SIZE", raising=False)
    monkeypatch.delenv("S2ST_ASR_BATCH_SIZE", raising=False)
    c = load_config(Path("configs/production-kokoro-jparacrawl-l4.yaml"))
    return replace(c, run=replace(c.run, output_dir=tmp_path / "production", input_jsonl=tmp_path / "sample-source.jsonl"))


def test_l4_profile_and_legacy_defaults(profile):
    assert profile.device.target_gpu == "L4" and profile.device.minimum_vram_gib == 20
    assert profile.tts.worker_count == 1 and profile.asr.batch_size == 8
    assert profile.performance.kokoro_worker_candidates == (1, 2, 4, 6)
    assert profile.performance.asr_batch_candidates == (8, 16, 24, 32)
    for name in ("production-qwen17b.yaml", "colab-pro.yaml"):
        assert load_config(Path("configs") / name).tts.backend == "qwen"


def test_l4_preflight_warning_and_t4_rejection(profile, monkeypatch):
    import sys
    props = SimpleNamespace(name="NVIDIA A100", total_memory=24 * 1024**3)
    torch = SimpleNamespace(__version__="test", float32="float32", float16="float16", bfloat16="bfloat16",
                            version=SimpleNamespace(cuda="test"), cuda=SimpleNamespace(is_available=lambda: True,
                            is_bf16_supported=lambda: False, get_device_properties=lambda n: props))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(runner, "ensure_reusable_budget", lambda *a, **kw: None)
    with pytest.warns(RuntimeWarning, match="L4"):
        result = runner.gpu_preflight(profile)
    assert result["target_gpu"] == "L4" and result["actual_gpu"] == "NVIDIA A100"
    props.name, props.total_memory = "NVIDIA T4", 15 * 1024**3
    with pytest.warns(RuntimeWarning), pytest.raises(RuntimeError, match="20.00 GiB"):
        runner.gpu_preflight(profile)


def test_recommendation_uses_throughput_and_headroom():
    candidates = [dict(worker_count=n, status="success", audio_hours_per_wall_hour=rate) for n, rate in ((1,20),(2,35),(4,47),(6,45))]
    assert perf.recommend(candidates, "worker_count", "audio_hours_per_wall_hour") == 4
    hardware = {"gpu": "NVIDIA L4", "vram_gib": 24}
    candidates[2]["gpu_peak_used_gib_sampled"] = 23
    perf._headroom(candidates[2], hardware)
    assert candidates[2]["headroom_warning"]
    assert perf.recommend(candidates, "worker_count", "audio_hours_per_wall_hour") == 6
    assert perf.recommend([{"status":"oom"}], "worker_count", "audio_hours_per_wall_hour") is None


def test_asr_quality_changes_exclude_candidate():
    quality = {"p:ja": {"asr_text":"文", "metric":0., "qc_pass":True}}
    candidates = [dict(status="success", batch_size=8, quality=quality, audio_seconds_per_wall_second=100),
                  dict(status="success", batch_size=16, quality={"p:ja": {"asr_text":"別", "metric":1., "qc_pass":False}}, audio_seconds_per_wall_second=200)]
    perf.compare_asr_quality(candidates)
    assert candidates[1]["qc_disagreements"] == 1 and not candidates[1]["quality_equivalent"]
    assert perf.recommend(candidates, "batch_size", "audio_seconds_per_wall_second") == 8


def test_transcript_difference_with_equal_metric_and_qc_is_not_equivalent():
    candidates = [
        {"status":"success", "batch_size":batch, "audio_seconds_per_wall_second":speed,
         "quality":{"p:en":{"asr_text":text, "metric":.25, "qc_pass":False}}}
        for batch, speed, text in ((8,100,"I like this book."), (24,200,"I like the book."))]
    perf.compare_asr_quality(candidates)
    candidate = candidates[1]
    assert candidate["metric_changes"] == candidate["qc_disagreements"] == 0
    assert candidate["transcript_changes"] == 1 and not candidate["quality_equivalent"]
    assert candidates[0]["quality_equivalent"]
    assert perf.recommend(candidates, "batch_size", "audio_seconds_per_wall_second") == 8


def test_candidate_subprocess_isolation_and_oom(profile, tmp_path, monkeypatch):
    contexts, processes = [], []
    class Process:
        exitcode = 0
        def __init__(self, target, args): self.args = args; processes.append(self)
        def start(self): write_json(Path(self.args[4]), {"status": "oom", "oom": True})
        def is_alive(self): return False
    class Sampler:
        peak = None
        def __init__(self): self.stop=threading.Event(); self.thread=threading.Thread(target=lambda: None)
    monkeypatch.setattr(perf, "_GpuSampler", Sampler)
    monkeypatch.setattr(perf.multiprocessing, "get_context", lambda method: contexts.append(method) or SimpleNamespace(Process=Process))
    for i in range(2):
        result = perf.run_candidate("asr", profile, tmp_path / "input", tmp_path / str(i), tmp_path / "log")
        assert result["status"] == "oom"
    assert contexts == ["spawn", "spawn"] and len(processes) == 2


@pytest.mark.parametrize("scarce_bin", [False, True])
def test_full_candidate_iteration_same_sample_audio_and_report(profile, tmp_path, monkeypatch, scarce_bin):
    c = profile
    rows = [{"pair_id":f"b{b}-{i}", "corpus":"jparacrawl_v3_filtered", "split":"train", "length_bin":b,
             "ja_text":f"Ａ　そのまま{i}", "en_text":f"Exact  text {i}"} for b in range(5) for i in range(1 if scarce_bin and b == 0 else 100)]
    write_checkpoint(c.run.input_jsonl, rows)
    monkeypatch.setattr(perf, "validate_external", lambda c: {})
    monkeypatch.setattr(perf, "verify_inputs", lambda c: None)
    monkeypatch.setattr(perf, "_implementation_commit", lambda: "a"*40)
    monkeypatch.setattr(runner, "gpu_preflight", lambda c: {"gpu":"NVIDIA L4", "vram_gib":24})
    monkeypatch.setattr(kokoro_backend, "materialize_assets", lambda c: tmp_path)
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kw: str(tmp_path))
    sample_hashes, audio_sources, batch_calls = [], [], []
    fixed = tmp_path / "fixed.wav"
    fixed.write_bytes(b"constant audio bytes")
    def candidate(stage, config, source, output, log):
        if stage == "tts":
            sample_hashes.append(sha256_file(source))
            selected = read_jsonl(source)
            assert len(selected) == 200
            checkpoint = output / "cp.jsonl"
            records = [runner._new_record(r, True) for r in selected]
            for r in records:
                for lang in ("ja", "en"):
                    r["attempts"][lang] = [{"generation_status":"ok", "duration":1., "wav_16k":str(fixed),
                        "sha256":sha256_file(fixed), "voice":getattr(config.tts, f"{lang}_voice")}]
            write_checkpoint(checkpoint, records)
            return {"status":"success", "worker_count":config.tts.worker_count, "generation_failures":0,
                    "audio_hours_per_wall_hour":{1:20,2:35,4:47,6:45}[config.tts.worker_count], "checkpoint":str(checkpoint)}
        audio_sources.append((str(source), sha256_file(source)))
        batch_calls.append(config.asr.batch_size)
        if config.asr.batch_size == 32:
            return {"status":"oom", "batch_size":32}
        return {"status":"success", "batch_size":config.asr.batch_size,
                "audio_seconds_per_wall_second":{8:100,16:165,24:190}[config.asr.batch_size],
                "quality":{"p:ja":{"asr_text":"文", "metric":0., "qc_pass":True}}}
    monkeypatch.setattr(perf, "run_candidate", candidate)
    before = asdict(c)
    report = perf.benchmark_performance(c, local_dir=tmp_path / "local", report_dir=tmp_path / "reports", batches=[8,32,16,24])
    assert len(sample_hashes) == 4 and len(set(sample_hashes)) == 1
    assert batch_calls == [8,32,16,24] and len(set(audio_sources)) == 1
    assert report["recommended_tts_workers"] == 4 and report["recommended_asr_batch_size"] == 24
    expected = {0: 1, 1: 31, 2: 81, 3: 67, 4: 20} if scarce_bin else perf.TUNING_QUOTAS
    assert report["selection"]["length_bin_quotas"] == {str(k):v for k,v in expected.items()}
    assert report["selection"]["requested_length_bin_quotas"] == {str(k):v for k,v in perf.TUNING_QUOTAS.items()}
    assert asdict(c) == before and report["production_config_modified"] is False
    saved = json.loads((tmp_path / "reports/l4-performance.json").read_text(encoding="utf-8"))
    assert saved["asr_candidates"][1]["status"] == "oom"
    # A separate ASR invocation on the same commit can use the fixed TTS audio.
    resumed = perf.benchmark_performance(c, stage="asr", local_dir=tmp_path / "local", report_dir=tmp_path / "reports", batches=[8,16,24])
    assert resumed["implementation_commit"] == "a"*40
    assert resumed["recommended_asr_batch_size"] == 24


@pytest.mark.parametrize("previous,current", [("a"*40,"b"*40), (None,"a"*40), ("a"*40,None), (None,None)])
def test_asr_tuning_rejects_changed_or_unknown_commit_without_overwriting_report(profile, tmp_path, monkeypatch, previous, current):
    report_dir = tmp_path / "reports"
    report_path = report_dir / "l4-performance.json"
    write_json(report_path, {"implementation_commit": previous})
    before = report_path.read_bytes()
    monkeypatch.setattr(runner, "gpu_preflight", lambda c: {"gpu":"NVIDIA L4", "vram_gib":24})
    monkeypatch.setattr(perf, "validate_external", lambda c: {})
    monkeypatch.setattr(perf, "_implementation_commit", lambda: current)
    monkeypatch.setattr(perf, "run_candidate", lambda *a: pytest.fail("commit mismatch must stop before candidate launch"))
    with pytest.raises(ValueError, match="implementation commit mismatch/unknown"):
        perf.benchmark_performance(profile, stage="asr", local_dir=tmp_path / "local", report_dir=report_dir)
    assert report_path.read_bytes() == before


def test_tuning_rejects_drive_audio(profile, tmp_path):
    with pytest.raises(ValueError, match="local disk"):
        perf.benchmark_performance(profile, local_dir="/content/drive/MyDrive/performance-audio", report_dir=tmp_path)


def test_asr_batch_candidates_preserve_mock_text_metric_and_qc(profile, tmp_path, monkeypatch):
    snapshots = []
    def transcriber(paths, **kwargs): return [{"text": p} for p in paths]
    monkeypatch.setattr(runner, "_load_whisper", lambda c: (transcriber, SimpleNamespace(tokenizer=object()), None, None))
    monkeypatch.setattr(runner, "_release_cuda", lambda t: None)
    for batch in (8,16,24,32):
        c = replace(profile, asr=replace(profile.asr, batch_size=batch))
        records, pending = [], []
        for i in range(10):
            row = {"pair_id":f"p{i}", "attempts":{}, "ja_tts_text":f"word{i}", "en_tts_text":f"word{i}"}
            for lang in ("ja", "en"):
                a = {"attempt":0, "generation_status":"ok", "duration":1., "wav_16k":f"word{i}"}
                row["attempts"][lang] = [a]
                pending.append((row, lang, a))
            records.append(row)
        runner._qc_batched(c, records, pending, tmp_path / f"qc-{batch}.jsonl", lambda text, **kw: text)
        snapshots.append([(a["asr_text"], a["metric"], a["qc_pass"]) for _, _, a in pending])
    assert all(s == snapshots[0] for s in snapshots) and len(snapshots[0]) == 20
