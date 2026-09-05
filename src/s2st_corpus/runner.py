from __future__ import annotations

import gc
import json
import os
import platform
import shutil
import time
from datetime import datetime, timezone
from collections import Counter
from pathlib import Path
from typing import Any

import jiwer
import numpy as np

from .audio import (
    atomic_write_wav,
    duration_seconds,
    mono_float32,
    resample_mono,
    sha256_file,
)
from .io import directory_size, ensure_reusable_budget
from .normalization import (
    japanese_reading_normalize,
    minimal_normalize,
    whisper_english_normalize,
)
from .config import AppConfig
from .manifest import (
    pairs_for_shard,
    pcm16_storage_gib,
    plan_manifest,
    read_jsonl,
    write_checkpoint,
)
from .seeding import deterministic_seed, seed_everything


def _shard_name(index: int, total: int) -> str:
    return f"shard-{index:05d}-of-{total:05d}"


def _release_cuda(torch_module: Any) -> None:
    gc.collect()
    if torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()


def _dtype(torch_module: Any, name: str) -> Any:
    if name == "auto":
        return (
            torch_module.bfloat16
            if torch_module.cuda.is_bf16_supported()
            else torch_module.float16
        )
    values = {
        "float16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "float32": torch_module.float32,
    }
    try:
        return values[name]
    except KeyError as error:
        raise ValueError(f"unsupported dtype: {name}") from error


def _materialize_hf_model(model_id: str, revision: str) -> str:
    """Download a pinned model as regular files when a model root is configured.

    Qwen's nested speech-tokenizer loader can resolve a different Hub revision,
    and the normal Hub cache relies on symlinks that are unsuitable for mounted
    Drive storage. A local_dir snapshot avoids both problems.
    """
    root_value = os.environ.get("S2ST_MODEL_ROOT")
    if not root_value:
        return model_id

    from huggingface_hub import snapshot_download

    destination = (
        Path(root_value)
        / model_id.replace("/", "--")
        / revision
    )
    marker = destination / ".s2st-complete"
    required = (
        destination / "config.json",
        destination / "model.safetensors",
        destination / "speech_tokenizer" / "config.json",
        destination / "speech_tokenizer" / "preprocessor_config.json",
        destination / "speech_tokenizer" / "model.safetensors",
    )
    if not marker.is_file() or not all(path.is_file() for path in required):
        print(f"[model] materializing {model_id}@{revision} at {destination}")
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            local_dir=destination,
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise RuntimeError(f"model snapshot is incomplete: {missing}")
        marker.write_text(revision + "\n", encoding="utf-8")
    return str(destination)


def plan(config: AppConfig) -> dict[str, Any]:
    manifest = plan_manifest(config.run.input_jsonl, config.run.num_shards)
    estimated_audio_gib = pcm16_storage_gib(
        config.run.expected_hours_per_language,
        retry_fraction=config.run.retry_storage_fraction,
    )
    return {
        **manifest,
        "input_jsonl": str(config.run.input_jsonl),
        "output_dir": str(config.run.output_dir),
        "expected_hours_per_language": config.run.expected_hours_per_language,
        "canonical_audio_gib_with_retry_allowance": round(estimated_audio_gib, 2),
        "maximum_output_gib": config.run.maximum_output_gib,
        "save_native": config.run.save_native,
        "within_pair_limit": (
            manifest["maximum_pairs_per_shard"]
            <= config.run.max_pairs_per_shard
        ),
        "within_audio_budget": estimated_audio_gib <= config.run.maximum_output_gib,
    }


def gpu_preflight(config: AppConfig) -> dict[str, Any]:
    import torch

    ensure_reusable_budget(
        config.run.output_dir,
        required_total_gib=config.run.minimum_free_gib,
    )
    ensure_reusable_budget(
        Path(os.environ.get("HF_HOME", config.run.output_dir / ".hf-cache")),
        required_total_gib=config.run.minimum_cache_free_gib,
    )
    if config.device.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the production profile")
    properties = torch.cuda.get_device_properties(0)
    vram_gib = properties.total_memory / (1024**3)
    if vram_gib < config.device.minimum_vram_gib:
        raise RuntimeError(
            f"GPU VRAM is {vram_gib:.2f} GiB; "
            f"{config.device.minimum_vram_gib:.2f} GiB is required"
        )
    bf16_supported = bool(torch.cuda.is_bf16_supported())
    if config.device.require_bf16 and not bf16_supported:
        raise RuntimeError("The production profile requires a BF16-capable GPU")
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": properties.name,
        "vram_gib": round(vram_gib, 2),
        "bf16_supported": bf16_supported,
        "tts_dtype": str(_dtype(torch, config.tts.dtype)).removeprefix("torch."),
        "tts_batch_size": config.tts.batch_size,
        "asr_batch_size": config.asr.batch_size,
    }


def _paths(
    config: AppConfig,
    shard_name: str,
    pair_id: str,
    language: str,
    attempt: int,
) -> tuple[Path | None, Path]:
    filename = f"{pair_id}.a{attempt}.wav"
    native = None
    if config.run.save_native:
        native = (
            config.run.output_dir
            / "audio"
            / "native"
            / language
            / shard_name
            / filename
        )
    canonical = (
        config.run.output_dir
        / "audio"
        / "16k"
        / language
        / shard_name
        / filename
    )
    return native, canonical


def _valid_existing(path: Path, config: AppConfig) -> bool:
    try:
        duration = duration_seconds(path)
        return (
            path.is_file()
            and config.qc.min_duration_seconds
            <= duration
            <= config.qc.max_duration_seconds
        )
    except Exception:
        return False


def _fatal_accelerator_error(error: BaseException) -> bool:
    text = str(error).lower()
    markers = (
        "cuda",
        "out of memory",
        "device-side assert",
        "cublas",
        "cudnn",
    )
    return isinstance(error, RuntimeError) and any(marker in text for marker in markers)


def _new_record(pair: dict[str, Any]) -> dict[str, Any]:
    record = dict(pair)
    record.setdefault("ja_text_raw", pair["ja_text"])
    record.setdefault("en_text_raw", pair["en_text"])
    record["ja_tts_text"] = minimal_normalize(pair["ja_text"])
    record["en_tts_text"] = minimal_normalize(pair["en_text"])
    record["attempts"] = {"ja": [], "en": []}
    return record


def _load_qwen(config: AppConfig) -> tuple[Any, Any]:
    import torch
    from qwen_tts import Qwen3TTSModel

    model_source = _materialize_hf_model(
        config.tts.model_id,
        config.tts.revision,
    )
    print(f"[tts] loading {config.tts.model_id}@{config.tts.revision}")
    load_options: dict[str, Any] = {}
    if model_source == config.tts.model_id:
        load_options["revision"] = config.tts.revision
    model = Qwen3TTSModel.from_pretrained(
        model_source,
        device_map=config.tts.device,
        dtype=_dtype(torch, config.tts.dtype),
        attn_implementation=config.device.attention,
        **load_options,
    )
    generation = model.model.talker.code_predictor.generation_config
    generation.remove_invalid_values = config.tts.remove_invalid_values
    generation.renormalize_logits = config.tts.renormalize_logits
    return model, torch


def _generate_attempt(
    model: Any,
    torch_module: Any,
    config: AppConfig,
    shard_name: str,
    record: dict[str, Any],
    language: str,
    attempt: int,
    generated: tuple[Any, int] | None = None,
) -> dict[str, Any]:
    native_path, canonical_path = _paths(
        config, shard_name, record["pair_id"], language, attempt
    )
    seed = deterministic_seed(record["pair_id"], language, attempt)
    started = time.perf_counter()
    try:
        if not (config.run.resume and _valid_existing(canonical_path, config)):
            if generated is None:
                seed_everything(seed, torch_module)
                waveforms, sample_rate = model.generate_custom_voice(
                    text=record[f"{language}_tts_text"],
                    language="Japanese" if language == "ja" else "English",
                    speaker=config.tts.speaker,
                    instruct=config.tts.instruct,
                    max_new_tokens=config.tts.max_new_tokens,
                    do_sample=config.tts.do_sample,
                    subtalker_dosample=config.tts.subtalker_do_sample,
                    remove_invalid_values=config.tts.remove_invalid_values,
                    renormalize_logits=config.tts.renormalize_logits,
                )
            else:
                waveform, sample_rate = generated
                waveforms = [waveform]
            waveform = mono_float32(np.asarray(waveforms[0]))
            canonical = resample_mono(
                waveform, int(sample_rate), config.tts.output_sample_rate
            )
            if native_path is not None:
                atomic_write_wav(native_path, waveform, int(sample_rate), "FLOAT")
            atomic_write_wav(
                canonical_path,
                canonical,
                config.tts.output_sample_rate,
                "PCM_16",
            )
        return {
            "attempt": attempt,
            "seed": seed,
            "generation_status": "ok",
            "wav_native": str(native_path) if native_path is not None else None,
            "wav_16k": str(canonical_path),
            "duration": duration_seconds(canonical_path),
            "sha256": sha256_file(canonical_path),
            "tts_seconds": round(time.perf_counter() - started, 3),
        }
    except Exception as error:
        if _fatal_accelerator_error(error):
            raise
        return {
            "attempt": attempt,
            "seed": seed,
            "generation_status": "failed",
            "generation_error": f"{type(error).__name__}: {error}",
            "wav_native": str(native_path) if native_path is not None else None,
            "wav_16k": str(canonical_path),
            "tts_seconds": round(time.perf_counter() - started, 3),
        }


def _generate_missing(
    config: AppConfig,
    shard_name: str,
    records: list[dict[str, Any]],
    attempt: int,
    targets: set[tuple[str, str]],
    checkpoint_path: Path,
) -> None:
    missing = {
        (record["pair_id"], language)
        for record in records
        for language in ("ja", "en")
        if (record["pair_id"], language) in targets
        and not any(
            item["attempt"] == attempt
            for item in record["attempts"][language]
        )
    }
    if not missing:
        return
    if config.tts.batch_size > 1:
        _generate_missing_batched(config, shard_name, records, attempt, missing, checkpoint_path)
        return
    model, torch_module = _load_qwen(config)
    changed = 0
    try:
        for record in records:
            for language in ("ja", "en"):
                key = (record["pair_id"], language)
                if key not in missing:
                    continue
                attempts = record["attempts"][language]
                if any(item["attempt"] == attempt for item in attempts):
                    continue
                attempts.append(
                    _generate_attempt(
                        model,
                        torch_module,
                        config,
                        shard_name,
                        record,
                        language,
                        attempt,
                    )
                )
                changed += 1
                print(
                    f"[tts] {changed}/{len(missing)} {record['pair_id']} {language} "
                    f"attempt={attempt} status={attempts[-1]['generation_status']}",
                    flush=True,
                )
                if changed % config.run.checkpoint_every == 0:
                    write_checkpoint(checkpoint_path, records)
        write_checkpoint(checkpoint_path, records)
    finally:
        del model
        _release_cuda(torch_module)


def _generate_missing_batched(config, shard_name, records, attempt, missing, checkpoint_path):
    model, torch_module = _load_qwen(config)
    completed = 0
    try:
        for language in ("ja", "en"):
            pending = [r for r in records if (r["pair_id"], language) in missing]
            pending.sort(key=lambda r: (len(r[f"{language}_tts_text"]), r["pair_id"]))
            for start in range(0, len(pending), config.tts.batch_size):
                group = pending[start:start + config.tts.batch_size]
                # Recover already-written audio after an interrupted checkpoint write.
                fresh = []
                for record in group:
                    _, path = _paths(config, shard_name, record["pair_id"], language, attempt)
                    if config.run.resume and _valid_existing(path, config):
                        result = _generate_attempt(model, torch_module, config, shard_name,
                                                   record, language, attempt)
                        # The seed provenance of an orphaned batch WAV is unknown.
                        result.update(seed=None, seed_scope="recovered_audio")
                        record["attempts"][language].append(result)
                        completed += 1
                    else:
                        fresh.append(record)
                if fresh:
                    ids = [r["pair_id"] for r in fresh]
                    seed = deterministic_seed(json.dumps(ids), language, attempt)
                    seed_everything(seed, torch_module)
                    started = time.perf_counter()
                    waveforms, rate = model.generate_custom_voice(
                        text=[r[f"{language}_tts_text"] for r in fresh],
                        language="Japanese" if language == "ja" else "English",
                        speaker=config.tts.speaker, instruct=config.tts.instruct,
                        max_new_tokens=config.tts.max_new_tokens,
                        do_sample=config.tts.do_sample,
                        subtalker_dosample=config.tts.subtalker_do_sample,
                        remove_invalid_values=config.tts.remove_invalid_values,
                        renormalize_logits=config.tts.renormalize_logits,
                    )
                    if len(waveforms) != len(fresh):
                        raise RuntimeError("Qwen batch output count does not match inputs")
                    elapsed = round(time.perf_counter() - started, 3)
                    for record, waveform in zip(fresh, waveforms):
                        result = _generate_attempt(model, torch_module, config, shard_name,
                                                   record, language, attempt, (waveform, rate))
                        result.update(seed=seed, seed_scope="batch", batch_pair_ids=ids,
                                      tts_batch_size=len(fresh), tts_batch_seconds=elapsed,
                                      tts_seconds=round(elapsed / len(fresh), 3))
                        record["attempts"][language].append(result)
                        completed += 1
                write_checkpoint(checkpoint_path, records)
                print(f"[tts-batch] {completed}/{len(missing)} {language} attempt={attempt} "
                      f"batch_size={len(fresh)}", flush=True)
    except Exception:
        write_checkpoint(checkpoint_path, records)
        raise
    finally:
        del model
        _release_cuda(torch_module)


def _load_whisper(config: AppConfig) -> tuple[Any, Any, Any, Any]:
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    print(f"[asr] loading {config.asr.model_id}@{config.asr.revision}")
    dtype = _dtype(torch, config.asr.dtype)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        config.asr.model_id,
        revision=config.asr.revision,
        dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        attn_implementation=config.asr.attention,
    ).to("cuda:0")
    processor = AutoProcessor.from_pretrained(
        config.asr.model_id,
        revision=config.asr.revision,
    )
    transcriber = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        dtype=dtype,
        device=0,
    )
    return transcriber, processor, model, torch


def _evaluate_attempt(
    config: AppConfig,
    record: dict[str, Any],
    language: str,
    attempt: dict[str, Any],
    transcriber: Any,
    processor: Any,
    g2p: Any,
) -> None:
    reasons: list[str] = []
    if attempt.get("generation_status") != "ok":
        attempt["qc_pass"] = False
        attempt["qc_reasons"] = ["generation_failed"]
        return
    duration = float(attempt["duration"])
    if not config.qc.min_duration_seconds <= duration <= config.qc.max_duration_seconds:
        attempt.pop("asr_error", None)
        attempt["qc_pass"] = False
        attempt["qc_reasons"] = ["duration_out_of_range"]
        return
    try:
        started = time.perf_counter()
        result = transcriber(
            attempt["wav_16k"],
            generate_kwargs={
                "language": "japanese" if language == "ja" else "english",
                "task": "transcribe",
                "do_sample": config.asr.do_sample,
                "num_beams": config.asr.num_beams,
                "condition_on_prev_tokens": config.asr.condition_on_prev_tokens,
            },
            return_timestamps=duration > 30.0,
        )
        asr_text = result["text"].strip()
        attempt["asr_text"] = asr_text
        attempt.pop("asr_error", None)
        attempt["asr_seconds"] = round(time.perf_counter() - started, 3)
        if language == "ja":
            reference = japanese_reading_normalize(record["ja_tts_text"], g2p)
            hypothesis = japanese_reading_normalize(asr_text, g2p)
            metric_name = "cer"
            metric = jiwer.cer(reference, hypothesis)
            threshold = config.qc.max_ja_cer
        else:
            reference = whisper_english_normalize(
                record["en_tts_text"], processor.tokenizer
            )
            hypothesis = whisper_english_normalize(asr_text, processor.tokenizer)
            metric_name = "wer"
            metric = jiwer.wer(reference, hypothesis)
            threshold = config.qc.max_en_wer
        attempt["reference_eval_norm"] = reference
        attempt["asr_eval_norm"] = hypothesis
        attempt[metric_name] = metric
        attempt["metric"] = metric
        if metric > threshold:
            reasons.append(f"{metric_name}_above_threshold")
    except Exception as error:
        if _fatal_accelerator_error(error):
            raise
        reasons.append("asr_failed")
        attempt["asr_error"] = f"{type(error).__name__}: {error}"
        raise RuntimeError(
            f"ASR/QC failed for {record['pair_id']} ({language}): {error}"
        ) from error
    attempt["qc_reasons"] = reasons
    attempt["qc_pass"] = not reasons


def _qc_unchecked(
    config: AppConfig,
    records: list[dict[str, Any]],
    only_attempt: int,
    checkpoint_path: Path,
) -> None:
    import pyopenjtalk

    pending = [(r, lang, a) for r in records for lang in ("ja", "en")
               for a in r["attempts"][lang]
               if a["attempt"] == only_attempt and "qc_pass" not in a]
    if not pending:
        return
    if config.asr.batch_size > 1:
        _qc_batched(config, records, pending, checkpoint_path, pyopenjtalk.g2p)
        return
    transcriber, processor, model, torch_module = _load_whisper(config)
    changed = 0
    try:
        for record in records:
            for language in ("ja", "en"):
                for attempt in record["attempts"][language]:
                    if (
                        attempt["attempt"] != only_attempt
                        or "qc_pass" in attempt
                    ):
                        continue
                    try:
                        _evaluate_attempt(
                            config, record, language, attempt,
                            transcriber, processor, pyopenjtalk.g2p,
                        )
                    except Exception:
                        write_checkpoint(checkpoint_path, records)
                        raise
                    changed += 1
                    print(
                        f"[qc] checked={changed} {record['pair_id']} {language} "
                        f"attempt={only_attempt} pass={attempt['qc_pass']} "
                        f"reasons={attempt['qc_reasons']}", flush=True,
                    )
                    if changed % config.run.checkpoint_every == 0:
                        write_checkpoint(checkpoint_path, records)
        write_checkpoint(checkpoint_path, records)
    finally:
        del transcriber
        del processor
        del model
        _release_cuda(torch_module)


def _qc_batched(config, records, pending, checkpoint_path, g2p):
    eligible = []
    for record, language, attempt in pending:
        if (attempt.get("generation_status") != "ok" or not
                config.qc.min_duration_seconds <= float(attempt["duration"]) <= config.qc.max_duration_seconds):
            _evaluate_attempt(config, record, language, attempt, None, None, g2p)
        else:
            eligible.append((record, language, attempt))
    write_checkpoint(checkpoint_path, records)
    if not eligible:
        return
    transcriber, processor, model, torch_module = _load_whisper(config)
    completed = len(pending) - len(eligible)
    try:
        for language in ("ja", "en"):
            # Separate long-form inputs if a custom QC profile permits >30 seconds.
            for longform in (False, True):
                items = [item for item in eligible if item[1] == language
                         and (float(item[2]["duration"]) > 30.0) == longform]
                items.sort(key=lambda item: float(item[2]["duration"]))
                for start in range(0, len(items), config.asr.batch_size):
                    group = items[start:start + config.asr.batch_size]
                    started = time.perf_counter()
                    outputs = list(transcriber(
                        [a["wav_16k"] for _, _, a in group],
                        batch_size=len(group), num_workers=0,
                        generate_kwargs={"language": "japanese" if language == "ja" else "english",
                                         "task": "transcribe", "do_sample": config.asr.do_sample,
                                         "num_beams": config.asr.num_beams,
                                         "condition_on_prev_tokens": config.asr.condition_on_prev_tokens},
                        return_timestamps=longform,
                    ))
                    elapsed = round(time.perf_counter() - started, 3)
                    if len(outputs) != len(group):
                        raise RuntimeError("Whisper batch output count does not match inputs")
                    for (record, lang, attempt), output in zip(group, outputs):
                        _evaluate_attempt(config, record, lang, attempt,
                                          lambda *args, result=output, **kwargs: result, processor, g2p)
                        attempt.update(asr_batch_size=len(group), asr_batch_seconds=elapsed,
                                       asr_seconds=round(elapsed / len(group), 3))
                        completed += 1
                    write_checkpoint(checkpoint_path, records)
                    print(f"[qc-batch] {completed}/{len(pending)} {language} batch_size={len(group)} "
                          f"seconds={elapsed}", flush=True)
    except Exception:
        write_checkpoint(checkpoint_path, records)
        raise
    finally:
        del transcriber, processor, model
        _release_cuda(torch_module)


def select_attempt(attempts: list[dict[str, Any]]) -> dict[str, Any] | None:
    passing = [attempt for attempt in attempts if attempt.get("qc_pass") is True]
    if not passing:
        return None
    return min(passing, key=lambda item: (float(item["metric"]), item["attempt"]))


def _finalize_record(config: AppConfig, record: dict[str, Any]) -> dict[str, Any]:
    selected = {
        language: select_attempt(record["attempts"][language])
        for language in ("ja", "en")
    }
    record["qc_status"] = (
        "accepted" if all(selected.values()) else "rejected"
    )
    reasons: list[str] = []
    for language in ("ja", "en"):
        choice = selected[language]
        if choice is None:
            reasons.append(f"{language}_no_passing_attempt")
            continue
        record[f"{language}_attempt"] = choice["attempt"]
        record[f"{language}_wav_16k"] = choice["wav_16k"]
        record[f"{language}_wav_native"] = choice.get("wav_native")
        record[f"{language}_duration"] = choice["duration"]
        record[f"{language}_sha256"] = choice["sha256"]
        record[f"{language}_{'cer' if language == 'ja' else 'wer'}"] = choice[
            "metric"
        ]
        record[f"{language}_asr_text"] = choice["asr_text"]
    record["qc_reasons"] = reasons
    record.update(
        {
            "tts_model_id": config.tts.model_id,
            "tts_model_revision": config.tts.revision,
            "tts_speaker_id": config.tts.speaker,
            "tts_dtype": (
                str(_dtype(__import__("torch"), config.tts.dtype))
                .removeprefix("torch.")
            ),
            "qc_asr_model_id": config.asr.model_id,
            "qc_asr_model_revision": config.asr.revision,
        }
    )
    return record


def _enforce_output_limit(config: AppConfig) -> None:
    used_gib = directory_size(config.run.output_dir) / (1024**3)
    if used_gib > config.run.maximum_output_gib:
        raise RuntimeError(
            f"output safety limit exceeded: {used_gib:.2f} GiB > "
            f"{config.run.maximum_output_gib:.2f} GiB"
        )


def run_shard(config: AppConfig, shard_index: int) -> Path:
    hardware = gpu_preflight(config)
    _enforce_output_limit(config)
    shard_name = _shard_name(shard_index, config.run.num_shards)
    final_path = (
        config.run.output_dir / "manifests" / "qc" / f"{shard_name}.jsonl"
    )
    if config.run.resume and final_path.is_file():
        print(f"[resume] completed shard found: {final_path}")
        return final_path

    pairs = pairs_for_shard(
        config.run.input_jsonl,
        shard_index,
        config.run.num_shards,
        config.run.max_pairs_per_shard,
    )
    checkpoint = (
        config.run.output_dir
        / "manifests"
        / "generated"
        / f"{shard_name}.jsonl"
    )
    prior = {row["pair_id"]: row for row in read_jsonl(checkpoint)}
    records = [prior.get(pair["pair_id"], _new_record(pair)) for pair in pairs]
    print(
        "[preflight] "
        + json.dumps(
            {**hardware, "shard": shard_index, "pairs": len(records)},
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    if not records:
        write_checkpoint(final_path, [])
        failure_path = (
            config.run.output_dir / "failures" / f"{shard_name}.jsonl"
        )
        write_checkpoint(failure_path, [])
        print(f"[done] shard={shard_index} pairs=0 accepted=0 rejected=0")
        return final_path

    initial_targets = {
        (record["pair_id"], language)
        for record in records
        for language in ("ja", "en")
    }
    _generate_missing(
        config, shard_name, records, 0, initial_targets, checkpoint
    )
    _qc_unchecked(config, records, 0, checkpoint)

    if config.tts.max_content_retries:
        retry_targets = {
            (record["pair_id"], language)
            for record in records
            for language in ("ja", "en")
            if select_attempt(record["attempts"][language]) is None
        }
        if retry_targets:
            print(f"[retry] content failures={len(retry_targets)}")
            _generate_missing(
                config, shard_name, records, 1, retry_targets, checkpoint
            )
            _qc_unchecked(config, records, 1, checkpoint)

    finalized = [_finalize_record(config, record) for record in records]
    write_checkpoint(final_path, finalized)
    failures = [
        record for record in finalized if record["qc_status"] != "accepted"
    ]
    failure_path = (
        config.run.output_dir / "failures" / f"{shard_name}.jsonl"
    )
    write_checkpoint(failure_path, failures)
    _enforce_output_limit(config)
    print(
        f"[done] shard={shard_index} pairs={len(finalized)} "
        f"accepted={len(finalized) - len(failures)} rejected={len(failures)}"
    )
    return final_path


def recheck_shard(config: AppConfig, shard_index: int) -> Path:
    """Re-evaluate completed audio without synthesis, preserving previous reports."""
    import pyopenjtalk

    name = _shard_name(shard_index, config.run.num_shards)
    root = config.run.output_dir
    final_path = root / "manifests" / "qc" / f"{name}.jsonl"
    if not final_path.is_file():
        raise FileNotFoundError(f"Completed QC manifest required: {final_path}")
    records = list(read_jsonl(final_path))
    if not records:
        raise ValueError("No saved records to recheck")
    for record in records:
        for attempts in record["attempts"].values():
            for attempt in attempts:
                if attempt.get("generation_status") == "ok" and not Path(attempt["wav_16k"]).is_file():
                    raise FileNotFoundError(attempt["wav_16k"])
    backup = root / "qc-backups" / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    for directory in ("manifests", "failures"):
        if (root / directory).is_dir():
            shutil.copytree(root / directory, backup / directory)
    print(f"[recheck] previous reports backed up: {backup}", flush=True)
    transcriber, processor, model, torch_module = _load_whisper(config)
    try:
        for record in records:
            for language in ("ja", "en"):
                for attempt in record["attempts"][language]:
                    for key in ("qc_pass", "qc_reasons", "asr_error", "asr_text", "asr_seconds",
                                "reference_eval_norm", "asr_eval_norm", "metric", "cer", "wer"):
                        attempt.pop(key, None)
                    _evaluate_attempt(config, record, language, attempt,
                                      transcriber, processor, pyopenjtalk.g2p)
                    print(f"[recheck] {record['pair_id']} {language} attempt={attempt['attempt']} "
                          f"pass={attempt['qc_pass']} reasons={attempt['qc_reasons']}", flush=True)
    finally:
        del transcriber, processor, model
        _release_cuda(torch_module)
    finalized = [_finalize_record(config, record) for record in records]
    write_checkpoint(root / "manifests" / "generated" / f"{name}.jsonl", records)
    write_checkpoint(final_path, finalized)
    write_checkpoint(root / "failures" / f"{name}.jsonl",
                     [r for r in finalized if r["qc_status"] != "accepted"])
    return final_path


def consolidate(config: AppConfig, allow_incomplete: bool = False) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    missing: list[int] = []
    for shard_index in range(config.run.num_shards):
        name = _shard_name(shard_index, config.run.num_shards)
        path = config.run.output_dir / "manifests" / "qc" / f"{name}.jsonl"
        if not path.is_file():
            missing.append(shard_index)
            continue
        rows.extend(read_jsonl(path))
    if missing and not allow_incomplete:
        raise RuntimeError(
            f"{len(missing)} shard manifests are missing; first missing={missing[:10]}"
        )
    ids = [row["pair_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate pair_id found across shard manifests")

    release_dir = config.run.output_dir / "manifests" / "releases"
    write_checkpoint(release_dir / "all.jsonl", rows)
    accepted = [row for row in rows if row.get("qc_status") == "accepted"]
    write_checkpoint(release_dir / "accepted.jsonl", accepted)
    by_corpus = Counter(row.get("corpus", "unknown") for row in accepted)
    summary = {
        "complete": not missing,
        "missing_shards": missing,
        "total_pairs": len(rows),
        "accepted_pairs": len(accepted),
        "rejected_pairs": len(rows) - len(accepted),
        "acceptance_rate": len(accepted) / len(rows) if rows else 0.0,
        "accepted_by_corpus": dict(sorted(by_corpus.items())),
        "ja_hours": sum(float(row["ja_duration"]) for row in accepted) / 3600,
        "en_hours": sum(float(row["en_duration"]) for row in accepted) / 3600,
    }
    summary_path = release_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(summary_path)
    return summary
