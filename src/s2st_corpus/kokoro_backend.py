"""Pinned official Kokoro runtime; one model shared by two G2P pipelines."""
from __future__ import annotations

import os
import time
from pathlib import Path
from importlib.metadata import version
from functools import lru_cache

import numpy as np

from .audio import mono_float32, resample_mono, atomic_write_wav, sha256_file


@lru_cache(maxsize=16)
def _dictionary_hash(path, size, modified_ns):
    return sha256_file(Path(path))


def dictionary_identity():
    import unidic
    root = Path(unidic.DICDIR)
    marker = root / "version"
    expected = "unidic-3.1.0+2021-08-31"
    if not marker.is_file() or marker.read_text().strip() != expected:
        raise RuntimeError("Install the pinned dictionary: python -m unidic download 3.1.0+2021-08-31")
    hashes = {}
    for name in ("sys.dic", "unk.dic", "char.bin", "matrix.bin", "dicrc"):
        path = root / name
        stat = path.stat()
        hashes[name] = _dictionary_hash(str(path), stat.st_size, stat.st_mtime_ns)
    return {"version": expected, "sha256": hashes}


def resolve_device(requested, torch):
    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested in {"cuda", "cuda:0"}:
        if not torch.cuda.is_available():
            raise RuntimeError("Kokoro CUDA requested but CUDA is unavailable")
        return "cuda:0"
    if requested != "cpu":
        raise ValueError(f"unsupported device: {requested}")
    return "cpu"


def download_assets(config, voices=None):
    from huggingface_hub import snapshot_download
    voices = voices or [config.ja_voice, config.en_voice]
    root = os.environ.get("S2ST_MODEL_ROOT")
    local = str(Path(root) / config.model_id.replace("/", "--") / config.revision) if root else None
    return Path(snapshot_download(repo_id=config.model_id, revision=config.revision,
        local_dir=local, allow_patterns=["config.json", "kokoro-v1_0.pth", *[f"voices/{v}.pt" for v in voices]]))


class KokoroBackend:
    def __init__(self, config):
        import torch
        from kokoro import KModel, KPipeline
        if version("kokoro") != config.package_version:
            raise RuntimeError(f"install kokoro=={config.package_version}")
        self.dictionary = dictionary_identity()
        started = time.perf_counter()
        self.config = config
        self.torch = torch
        self.device = resolve_device(config.device, torch)
        self.root = download_assets(config)
        self.model = KModel(repo_id=config.model_id, config=str(self.root / "config.json"),
                            model=str(self.root / "kokoro-v1_0.pth")).to(self.device).eval()
        self.pipelines = {lang: KPipeline(lang_code=code, repo_id=config.model_id, model=self.model)
                          for lang, code in (("ja", "j"), ("en", "a"))}
        self.voices = {}
        self.load_seconds = time.perf_counter() - started

    def generate(self, text, language, voice=None):
        voice = voice or getattr(self.config, f"{language}_voice")
        if voice not in self.voices:
            path = self.root / "voices" / f"{voice}.pt"
            if not path.is_file():
                path = download_assets(self.config, [voice]) / "voices" / f"{voice}.pt"
            self.voices[voice] = self.torch.load(str(path), map_location="cpu", weights_only=True)
        chunks = []
        with self.torch.inference_mode():
            for result in self.pipelines[language](text, voice=self.voices[voice], speed=self.config.speed):
                audio = result.audio
                if audio is None:
                    continue
                if hasattr(audio, "detach"):
                    audio = audio.detach().cpu().numpy()
                if np.asarray(audio).size:
                    chunks.append(mono_float32(audio))
        if not chunks:
            raise ValueError("empty_audio: Kokoro returned no nonempty audio chunks")
        return np.concatenate(chunks), self.config.native_sample_rate


def voice_samples(config, output_dir, english_voices):
    import re
    ja = ["jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo"]
    if any(not re.fullmatch(r"a[fm]_[a-z]+", v) for v in english_voices):
        raise ValueError("English audition voices must be American English voice names")
    model = KokoroBackend(config.tts)
    outputs = []
    for language, voices, text in (("ja", ja, "今日は良い天気ですね。日本語の音声を確認しています。"),
                                   ("en", english_voices, "The weather is pleasant today. We are checking the English voice.")):
        for voice in voices:
            audio, rate = model.generate(text, language, voice)
            path = output_dir / f"{language}-{voice}.wav"
            atomic_write_wav(path, resample_mono(audio, rate, 16000), 16000, "PCM_16")
            outputs.append(str(path))
    return {"samples": outputs, "revision": config.tts.revision}
