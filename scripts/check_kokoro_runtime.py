"""Small real-model check, not a JParaCrawl/Whisper smoke benchmark."""
import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from importlib.metadata import version

import torch
import soundfile as sf

from s2st_corpus.config import load_config
from s2st_corpus.kokoro_backend import KokoroBackend
from s2st_corpus.audio import atomic_write_wav, resample_mono, sha256_file
from s2st_corpus.external import write_json

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
parser.add_argument("--output", type=Path, default=Path("data/kokoro-runtime-check"))
args = parser.parse_args()
torch.set_num_threads(4)
c = load_config(Path("configs/production-kokoro-jparacrawl.yaml"))
backend = KokoroBackend(replace(c.tts, device=args.device))
report = {"kind": "fixed-sentence TTS only; not corpus smoke", "device": backend.device,
          "model": c.tts.model_id, "revision": c.tts.revision, "kokoro": version("kokoro"),
          "torch": torch.__version__, "load_seconds": backend.load_seconds,
          "dictionary": backend.dictionary, "languages": {}}
for lang, text in (("ja", "今日は良い天気です。音声を確認しています。"),
                   ("en", "The weather is pleasant today. We are checking the English voice.")):
    start = time.perf_counter()
    audio, rate = backend.generate(text, lang)
    elapsed = time.perf_counter() - start
    assert rate == 24000 and audio.ndim == 1 and audio.size > 0
    native = args.output / f"{lang}-24k.wav"
    path = args.output / f"{lang}-16k.wav"
    atomic_write_wav(native, audio, rate, "FLOAT")
    atomic_write_wav(path, resample_mono(audio, rate, 16000), 16000, "PCM_16")
    info = sf.info(path)
    assert info.samplerate == 16000 and info.channels == 1 and info.subtype == "PCM_16"
    report["languages"][lang] = {"text": text, "voice": getattr(c.tts, f"{lang}_voice"),
        "duration_seconds": info.duration, "tts_seconds": elapsed, "rtf": elapsed/info.duration,
        "sha256": sha256_file(path), "wav": str(path)}
write_json(args.output / "report.json", report)
print(json.dumps(report, ensure_ascii=False, indent=2))
