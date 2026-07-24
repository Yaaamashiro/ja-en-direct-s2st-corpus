from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RunConfig:
    input_jsonl: Path
    output_dir: Path
    num_shards: int
    max_pairs_per_shard: int
    checkpoint_every: int
    minimum_free_gib: float
    minimum_cache_free_gib: float
    maximum_output_gib: float
    expected_hours_per_language: float
    retry_storage_fraction: float
    save_native: bool
    resume: bool


@dataclass(frozen=True)
class DeviceConfig:
    require_cuda: bool
    minimum_vram_gib: float
    require_bf16: bool
    attention: str


@dataclass(frozen=True)
class TTSConfig:
    model_id: str
    revision: str
    device: str
    dtype: str
    speaker: str
    instruct: str
    max_new_tokens: int
    do_sample: bool
    subtalker_do_sample: bool
    remove_invalid_values: bool
    renormalize_logits: bool
    output_sample_rate: int
    max_content_retries: int


@dataclass(frozen=True)
class ASRConfig:
    model_id: str
    revision: str
    dtype: str
    attention: str
    do_sample: bool
    num_beams: int
    condition_on_prev_tokens: bool


@dataclass(frozen=True)
class QCConfig:
    max_ja_cer: float
    max_en_wer: float
    min_duration_seconds: float
    max_duration_seconds: float


@dataclass(frozen=True)
class AppConfig:
    run: RunConfig
    device: DeviceConfig
    tts: TTSConfig
    asr: ASRConfig
    qc: QCConfig


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def load_config(path: Path) -> AppConfig:
    with path.open("r", encoding="utf-8") as handle:
        raw = _mapping(yaml.safe_load(handle), "config")

    run_values = _mapping(raw.get("run"), "run").copy()
    run_values["input_jsonl"] = Path(run_values["input_jsonl"])
    run_values["output_dir"] = Path(run_values["output_dir"])
    run = RunConfig(**run_values)
    device = DeviceConfig(**_mapping(raw.get("device"), "device"))
    tts = TTSConfig(**_mapping(raw.get("tts"), "tts"))
    asr = ASRConfig(**_mapping(raw.get("asr"), "asr"))
    qc = QCConfig(**_mapping(raw.get("qc"), "qc"))

    if run.num_shards < 1:
        raise ValueError("run.num_shards must be positive")
    if run.max_pairs_per_shard < 1:
        raise ValueError("run.max_pairs_per_shard must be positive")
    if run.checkpoint_every < 1:
        raise ValueError("run.checkpoint_every must be positive")
    if (
        run.minimum_free_gib < 1
        or run.minimum_cache_free_gib < 1
        or run.maximum_output_gib < 1
    ):
        raise ValueError("disk limits must be positive")
    if not 0 <= run.retry_storage_fraction <= 1:
        raise ValueError("run.retry_storage_fraction must be between 0 and 1")
    if tts.max_content_retries not in (0, 1):
        raise ValueError("tts.max_content_retries must be 0 or 1")
    if tts.device != "cuda:0" or tts.dtype != "bfloat16":
        raise ValueError("production Qwen must use cuda:0 and bfloat16")
    if asr.dtype != "float16":
        raise ValueError("production Whisper must use float16")
    if not device.require_cuda or not device.require_bf16:
        raise ValueError("production profile must require CUDA and BF16")
    if tts.output_sample_rate != 16000:
        raise ValueError("canonical corpus audio must be 16 kHz")
    if not (0 <= qc.max_ja_cer <= 1 and 0 <= qc.max_en_wer <= 1):
        raise ValueError("QC error-rate thresholds must be between 0 and 1")
    if qc.min_duration_seconds >= qc.max_duration_seconds:
        raise ValueError("invalid QC duration range")

    return AppConfig(run=run, device=device, tts=tts, asr=asr, qc=qc)
