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
class PrepareConfig:
    sources_dir: Path
    work_dir: Path
    reports_dir: Path
    target_train_hours: float
    jesc_fraction: float
    selection_seed: int
    ja_chars_per_second: float
    utterance_padding_seconds: float
    short_max_ja_chars: int
    medium_max_ja_chars: int
    short_time_fraction: float
    medium_time_fraction: float
    long_time_fraction: float
    include_dev: bool
    include_test: bool


@dataclass(frozen=True)
class CorpusSourceConfig:
    url: str
    archive_name: str
    sha256: str | None
    version: str
    license_id: str


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
    prepare: PrepareConfig
    sources: dict[str, CorpusSourceConfig]
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
    prepare_values = _mapping(raw.get("prepare"), "prepare").copy()
    for key in ("sources_dir", "work_dir", "reports_dir"):
        prepare_values[key] = Path(prepare_values[key])
    prepare = PrepareConfig(**prepare_values)
    source_values = _mapping(raw.get("sources"), "sources")
    sources = {
        name: CorpusSourceConfig(**_mapping(values, f"sources.{name}"))
        for name, values in source_values.items()
    }
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
    if set(sources) != {"jesc", "kftt"}:
        raise ValueError("sources must contain exactly jesc and kftt")
    if prepare.target_train_hours <= 0:
        raise ValueError("prepare.target_train_hours must be positive")
    if not 0 < prepare.jesc_fraction < 1:
        raise ValueError("prepare.jesc_fraction must be between 0 and 1")
    if prepare.ja_chars_per_second <= 0:
        raise ValueError("prepare.ja_chars_per_second must be positive")
    if prepare.utterance_padding_seconds < 0:
        raise ValueError("prepare.utterance_padding_seconds cannot be negative")
    if not (
        0 < prepare.short_max_ja_chars < prepare.medium_max_ja_chars
    ):
        raise ValueError("invalid prepare length-bin boundaries")
    length_fractions = (
        prepare.short_time_fraction
        + prepare.medium_time_fraction
        + prepare.long_time_fraction
    )
    if abs(length_fractions - 1.0) > 1e-9:
        raise ValueError("prepare length time fractions must sum to 1")
    for name, source in sources.items():
        if not isinstance(source.version, str) or not source.version:
            raise ValueError(f"sources.{name}.version must be a quoted string")
        if not isinstance(source.license_id, str) or not source.license_id:
            raise ValueError(f"sources.{name}.license_id must be a string")
        if not source.url.startswith("https://"):
            raise ValueError(f"sources.{name}.url must use HTTPS")
        if not source.archive_name.endswith(".tar.gz"):
            raise ValueError(f"sources.{name}.archive_name must be a .tar.gz")
        if source.sha256 is not None and (
            len(source.sha256) != 64
            or any(character not in "0123456789abcdef" for character in source.sha256)
        ):
            raise ValueError(f"sources.{name}.sha256 must be lowercase SHA-256")
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

    return AppConfig(
        run=run,
        prepare=prepare,
        sources=sources,
        device=device,
        tts=tts,
        asr=asr,
        qc=qc,
    )
