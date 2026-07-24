from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr


def mono_float32(waveform: np.ndarray) -> np.ndarray:
    audio = np.asarray(waveform, dtype=np.float32)
    audio = np.squeeze(audio)
    if audio.ndim != 1:
        raise ValueError(f"Expected mono audio, received shape={audio.shape}")
    if not np.isfinite(audio).all():
        raise ValueError("Audio contains NaN or infinity")
    return np.ascontiguousarray(audio)


def resample_mono(
    waveform: np.ndarray, source_rate: int, target_rate: int
) -> np.ndarray:
    if source_rate == target_rate:
        return mono_float32(waveform)
    return mono_float32(
        soxr.resample(waveform, source_rate, target_rate, quality="HQ")
    )


def atomic_write_wav(
    path: Path, waveform: np.ndarray, sample_rate: int, subtype: str
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.",
        suffix=".tmp.wav",
        dir=path.parent,
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        sf.write(temporary, waveform, sample_rate, subtype=subtype)
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def duration_seconds(path: Path) -> float:
    info = sf.info(path)
    return info.frames / info.samplerate
