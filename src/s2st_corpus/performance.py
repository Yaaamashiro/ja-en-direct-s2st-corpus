"""Isolated L4 performance candidates. Never edits production configuration."""
from __future__ import annotations

import json
import multiprocessing
import os
import subprocess
import threading
import time
from collections import Counter
from dataclasses import replace, asdict
from datetime import datetime, timezone
from pathlib import Path

from . import metrics
from .audio import sha256_file
from .external import validate_external, verify_inputs, write_json
from .manifest import read_jsonl, write_checkpoint


TUNING_QUOTAS = {0: 4, 1: 30, 2: 80, 3: 66, 4: 20}


def recommend(candidates, value_key, throughput_key):
    eligible = [c for c in candidates if c.get("status") == "success"
                and not c.get("headroom_warning") and c.get("quality_equivalent", True)
                and c.get("generation_failures", 0) == 0]
    return max(eligible, key=lambda c: c[throughput_key])[value_key] if eligible else None


def memory_report(torch, measurements):
    from .kokoro_parallel import memory_peaks
    parent = memory_peaks(torch)
    workers = measurements.get("tts_worker_per_process_peak", {})
    return {"parent_per_process_peak": parent, "worker_per_process_peak": workers,
            "peak_allocated_vram_gib_reference": max(parent["allocated_gib"], sum(p["allocated_gib"] for p in workers.values())),
            "peak_reserved_vram_gib_reference": max(parent["reserved_gib"], sum(p["reserved_gib"] for p in workers.values())),
            "max_worker_peak_reserved_gib": max((p["reserved_gib"] for p in workers.values()), default=0),
            "memory_note": "Per-process peaks and sums are reference bounds, not a measured simultaneous GPU-wide peak; CUDA context/other-process memory is excluded."}


def _candidate_main(stage, config, source, output, result_path, log_path):
    # Each candidate owns a fresh CUDA context, including after an OOM.
    from .kokoro_parallel import terminate_as_interrupt
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = {"status": "failed", "stage": stage,
              "oom": False, "asr_failures": 0, "generation_failures": 0,
              "worker_count": config.tts.worker_count, "batch_size": config.asr.batch_size}
    os.environ["HF_HUB_OFFLINE"] = "1"
    measurements = {}
    token = metrics.ACTIVE.set(measurements)
    started = time.perf_counter()
    try:
        with terminate_as_interrupt(), Path(log_path).open("w", encoding="utf-8", buffering=1) as log:
            # Child Kokoro processes inherit these OS descriptors on Colab.
            os.dup2(log.fileno(), 1)
            os.dup2(log.fileno(), 2)
            import torch
            from .runner import _new_record, _generate_missing, _qc_unchecked
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            report["torch_num_threads"] = torch.get_num_threads()
            checkpoint = output / "checkpoint.jsonl"
            records = ([_new_record(p, preserve_text=True) for p in read_jsonl(Path(source))]
                       if stage == "tts" else read_jsonl(Path(source)))
            phase_started = time.perf_counter()
            if stage == "tts":
                targets = {(r["pair_id"], lang) for r in records for lang in ("ja", "en")}
                _generate_missing(config, "performance", records, 0, targets, checkpoint)
            else:
                _qc_unchecked(config, records, 0, checkpoint)
            wall = time.perf_counter() - phase_started
            measured = metrics.summarize(records)
            generated = sum(s["generated_seconds"] for s in measured["attempt_metrics"].values())
            audio = generated if stage == "tts" else measurements.get("asr_audio_seconds", sum(
                a["duration"] for r in records for lang in ("ja", "en") for a in r["attempts"][lang]
                if a.get("generation_status") == "ok" and config.qc.min_duration_seconds <= a["duration"] <= config.qc.max_duration_seconds))
            utterances = len(records) * 2 if stage == "tts" else sum(
                "metric" in a for r in records for lang in ("ja", "en") for a in r["attempts"][lang])
            report.update(status="success", pairs=len(records), utterances=utterances,
                          wall_seconds=wall, generated_audio_seconds=generated, generated_audio_hours=generated / 3600,
                          audio_seconds=audio, audio_seconds_per_wall_second=audio / wall,
                          audio_hours_per_wall_hour=audio / wall, rtf=wall / audio if audio else None,
                          pairs_per_second=len(records) / wall, utterances_per_second=utterances / wall,
                          checkpoint=str(checkpoint), **memory_report(torch, measurements),
                          generation_failures=measured["failures"]["generation_failures"],
                          empty_audio=measured["failures"]["empty_audio"], asr_failures=measured["failures"]["asr_failures"],
                          load_seconds=measurements.get("tts_load_seconds" if stage == "tts" else "asr_load_seconds"),
                          measurements=measurements)
            if stage == "asr":
                report["quality"] = {f"{r['pair_id']}:{lang}": {
                    k: a.get(k) for k in ("asr_text", "metric", "qc_pass", "qc_reasons")}
                    for r in records for lang in ("ja", "en") for a in r["attempts"][lang]}
    except BaseException as error:
        oom = "out of memory" in str(error).lower()
        report.update(status="oom" if oom else "failed", oom=oom,
                      asr_failures=1 if stage == "asr" else 0,
                      error=f"{type(error).__name__}: {error}")
        try:
            import torch
            report.update(memory_report(torch, measurements))
        except Exception:
            pass
    finally:
        metrics.ACTIVE.reset(token)
        report["candidate_total_wall_seconds"] = time.perf_counter() - started
        write_json(Path(result_path), report)


class _GpuSampler:
    """Benchmark only; at most one nvidia-smi request every five seconds."""
    def __init__(self):
        self.peak = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self.stop.is_set():
            try:
                result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
                                        capture_output=True, text=True, timeout=3, check=True)
                value = float(result.stdout.strip().splitlines()[0]) / 1024
                self.peak = max(self.peak or 0, value)
            except (OSError, ValueError, subprocess.SubprocessError):
                return
            self.stop.wait(5)


def run_candidate(stage, config, source, output, log_path, *, timeout=7200):
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "candidate.json"
    sampler = _GpuSampler()
    process = multiprocessing.get_context("spawn").Process(
        target=_candidate_main, args=(stage, config, str(source), str(output), str(result_path), str(log_path)))
    timed_out = False
    try:
        process.start()
        sampler.thread.start()
        started = time.monotonic()
        while process.is_alive():
            process.join(timeout=1)
            if time.monotonic() - started > timeout:
                timed_out = True
                break
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=15)
            if process.is_alive():
                process.kill()
                process.join()
        sampler.stop.set()
        if sampler.thread.ident is not None:
            sampler.thread.join(timeout=4)
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {
        "status": "failed", "error": f"candidate process exited with code {process.exitcode}"}
    if timed_out:
        result.update(status="timeout", error="candidate timeout")
    elif result.get("status") == "success" and process.exitcode != 0:
        result.update(status="failed", error=f"candidate exited abnormally: {process.exitcode}")
    result.update(worker_count=config.tts.worker_count, batch_size=config.asr.batch_size,
                  gpu_peak_used_gib_sampled=sampler.peak, log=str(log_path), exitcode=process.exitcode)
    return result


def compare_asr_quality(candidates):
    successful = [r for r in candidates if r.get("status") == "success"]
    baseline = next((r for r in successful if r["batch_size"] == 8), successful[0] if successful else None)
    if baseline is None:
        return
    for candidate in successful:
        reference, quality = baseline["quality"], candidate["quality"]
        common = reference.keys() & quality.keys()
        mismatches = len(reference.keys() ^ quality.keys()) + sum(
            reference[k]["qc_pass"] != quality[k]["qc_pass"] for k in common)
        metric_changes = sum(reference[k].get("metric") != quality[k].get("metric") for k in common)
        candidate.update(reference_asr_batch_size=baseline["batch_size"], qc_disagreements=mismatches,
                         metric_changes=metric_changes,
                         transcript_changes=sum(reference[k].get("asr_text") != quality[k].get("asr_text") for k in common),
                         quality_equivalent=mismatches == 0 and metric_changes == 0)


def compare_tts_quality(candidates):
    successful = [r for r in candidates if r.get("status") == "success" and r.get("generation_failures", 0) == 0]
    if not successful:
        return
    baseline = next((r for r in successful if r["worker_count"] == 1), successful[0])
    def outputs(candidate):
        return {f"{r['pair_id']}:{lang}": (r[f"{lang}_tts_text"], a.get("voice"), a["duration"])
                for r in read_jsonl(Path(candidate["checkpoint"])) for lang in ("ja", "en")
                for a in r["attempts"][lang] if a.get("generation_status") == "ok"}
    reference = outputs(baseline)
    for candidate in successful:
        current = outputs(candidate)
        common = reference.keys() & current.keys()
        mismatches = len(reference.keys() ^ current.keys()) + sum(reference[k][:2] != current[k][:2] for k in common)
        duration_changes = sum(abs(current[k][2] - reference[k][2]) > max(.25, .05 * reference[k][2]) for k in common)
        candidate.update(reference_tts_workers=baseline["worker_count"], metadata_mismatches=mismatches,
                         large_duration_changes=duration_changes, quality_equivalent=mismatches == 0 and duration_changes == 0,
                         quality_note="TTS-only text/voice/duration comparison. ASR quality across worker counts still requires L4 validation; no byte-identity claim.")


def _headroom(result, hardware):
    result.update(gpu=hardware["gpu"], total_vram_gib=hardware["vram_gib"])
    peak = result.get("gpu_peak_used_gib_sampled")
    peak = max(peak or 0, result.get("peak_reserved_vram_gib_reference", 0))
    result["headroom_warning"] = peak > .9 * hardware["vram_gib"]
    result["headroom_measurement"] = "sampled GPU-wide used memory" if result.get("gpu_peak_used_gib_sampled") is not None else "per-process reserved peak reference; GPU-wide memory unavailable"


def benchmark_performance(config, stage="all", local_dir=None, report_dir=None, workers=None, batches=None):
    from .runner import gpu_preflight
    from .smoke import select_benchmark_pairs
    from .kokoro_backend import materialize_assets
    if config.tts.backend != "kokoro":
        raise ValueError("performance tuning requires Kokoro")
    local_dir = Path(local_dir or config.performance.local_output_dir).resolve()
    report_dir = Path(report_dir or config.performance.report_dir).resolve()
    if local_dir.is_relative_to(Path("/content/drive").resolve()) or any(local_dir.is_relative_to(p.resolve()) for p in (config.run.output_dir, config.smoke.output_dir)):
        raise ValueError("performance audio must use local disk outside Drive/production/smoke")
    if any(report_dir.is_relative_to(p.resolve()) for p in (config.run.output_dir, config.smoke.output_dir)):
        raise ValueError("performance reports must be separate from production/smoke")
    workers = list(workers or config.performance.kokoro_worker_candidates)
    batches = list(batches or config.performance.asr_batch_candidates)
    if len(set(workers)) != len(workers) or len(set(batches)) != len(batches) or any(type(n) is not int or not 1 <= n <= 8 for n in workers) or any(type(n) is not int or n < 1 for n in batches):
        raise ValueError("invalid performance candidates")
    local_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    logs = report_dir / "logs"
    logs.mkdir(exist_ok=True)
    profile = replace(config, run=replace(config.run, output_dir=local_dir, minimum_free_gib=1, maximum_output_gib=20))
    hardware = gpu_preflight(profile)
    validation = validate_external(config)
    name = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    workspace = local_dir / name
    workspace.mkdir()
    report_path = report_dir / "l4-performance.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if stage == "asr" and report_path.is_file() else {
        "target_gpu": config.device.target_gpu or "L4", "gpu": hardware["gpu"], "total_vram_gib": hardware["vram_gib"],
        "tts_candidates": [], "asr_candidates": [], "recommended_tts_workers": None, "recommended_asr_batch_size": None}
    synthesis_conditions = {k: getattr(config.tts, k) for k in ("model_id", "revision", "ja_voice", "en_voice", "speed", "dtype", "package_version")}
    asr_conditions = {k: v for k, v in asdict(config.asr).items() if k != "batch_size"}
    if stage == "asr" and report.get("synthesis_conditions") is not None and (report["synthesis_conditions"] != synthesis_conditions or report["asr_conditions"] != asr_conditions or report["input_validation"] != validation):
        raise ValueError("tuning input/model/voice/ASR conditions changed; rerun TTS tuning")
    report.update(synthesis_conditions=synthesis_conditions, asr_conditions=asr_conditions, input_validation=validation)
    try:
        repository = Path(__file__).resolve().parents[2]
        report["implementation_commit"] = subprocess.check_output(["git", "-c", f"safe.directory={repository.as_posix()}", "rev-parse", "HEAD"], cwd=repository, text=True, timeout=5).strip()
    except (OSError, subprocess.SubprocessError):
        report["implementation_commit"] = None
    if stage in {"tts", "all"}:
        sample = workspace / "sample.jsonl"
        rows = select_benchmark_pairs(config.run.input_jsonl, sample, seed=42, quotas=TUNING_QUOTAS)
        materialize_assets(config.tts)
        report.update(sample_manifest_sha256=sha256_file(sample), selection=json.loads(sample.with_suffix(".selection.json").read_text()),
                      length_bin_distribution=dict(Counter(str(r["length_bin"]) for r in rows)), tts_candidates=[], asr_candidates=[])
        fixed_audio = None
        for count in workers:
            output = workspace / f"tts-workers-{count}"
            candidate_config = replace(profile, run=replace(profile.run, output_dir=output), tts=replace(config.tts, worker_count=count))
            print(f"[performance] TTS workers={count}", flush=True)
            result = run_candidate("tts", candidate_config, sample, output, logs / f"{name}-tts-{count}.log")
            _headroom(result, hardware)
            report["tts_candidates"].append(result)
            if fixed_audio is None and result.get("status") == "success" and result.get("generation_failures", 0) == 0:
                fixed_audio = result["checkpoint"]
            report["fixed_audio_manifest"] = fixed_audio
            compare_tts_quality(report["tts_candidates"])
            report["recommended_tts_workers"] = recommend(report["tts_candidates"], "worker_count", "audio_hours_per_wall_hour")
            write_json(report_path, report)
    if stage in {"asr", "all"}:
        source = report.get("fixed_audio_manifest")
        if not source or not Path(source).is_file():
            raise ValueError("fixed local TTS audio is unavailable; run TTS tuning again in this runtime")
        source = Path(source)
        fixed_rows = read_jsonl(source)
        audio_hashes = {a["wav_16k"]: a["sha256"] for r in fixed_rows for lang in ("ja", "en") for a in r["attempts"][lang]}
        for path, digest in audio_hashes.items():
            if sha256_file(Path(path)) != digest:
                raise ValueError(f"fixed benchmark audio SHA256 mismatch: {path}")
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=config.asr.model_id, revision=config.asr.revision,
                          allow_patterns=["*.json", "merges.txt", "model.safetensors"])
        report.update(asr_candidates=[], fixed_audio_manifest_sha256=sha256_file(Path(source)))
        for batch in batches:
            output = workspace / f"asr-batch-{batch}"
            candidate_config = replace(profile, run=replace(profile.run, output_dir=output), asr=replace(config.asr, batch_size=batch))
            print(f"[performance] ASR batch={batch}", flush=True)
            result = run_candidate("asr", candidate_config, source, output, logs / f"{name}-asr-{batch}.log")
            _headroom(result, hardware)
            result["input_audio_manifest_sha256"] = report["fixed_audio_manifest_sha256"]
            report["asr_candidates"].append(result)
            for path, digest in audio_hashes.items():
                if sha256_file(Path(path)) != digest:
                    raise ValueError(f"benchmark input audio changed: {path}")
            compare_asr_quality(report["asr_candidates"])
            report["recommended_asr_batch_size"] = recommend(report["asr_candidates"], "batch_size", "audio_seconds_per_wall_second")
            write_json(report_path, report)
    verify_inputs(config)
    report["production_config_modified"] = False
    write_json(report_path, report)
    write_json(report_dir / f"l4-performance-{name}.json", report)
    print(f"Recommended: Kokoro workers={report['recommended_tts_workers']}; Whisper batch={report['recommended_asr_batch_size']}", flush=True)
    return report
