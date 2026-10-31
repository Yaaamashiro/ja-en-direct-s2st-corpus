"""Strict external-profile resume checks, including rejected attempts."""
from pathlib import Path
import math
import soundfile as sf

from .audio import sha256_file


def validate_records(config, pairs, records, provenance, *, complete=False):
    expected = {p["pair_id"]: p for p in pairs}
    seen = set()
    for row in records:
        key = row["pair_id"]
        if key not in expected or key in seen:
            raise RuntimeError(f"checkpoint has unexpected/duplicate pair_id: {key}")
        seen.add(key)
        pair = expected[key]
        if complete:
            selected = {}
            for lang in ("ja", "en"):
                passing = [a for a in row.get("attempts", {}).get(lang, []) if a.get("qc_pass") is True]
                selected[lang] = min(passing, key=lambda a: (float(a["metric"]), a["attempt"])) if passing else None
            status = "accepted" if all(selected.values()) else "rejected"
            if row.get("qc_status") != status:
                raise RuntimeError(f"final QC status mismatch: {key}")
            for lang, choice in selected.items():
                if choice and any(row.get(f"{lang}_{field}") != choice.get(field) for field in ("wav_16k", "duration", "sha256", "attempt")):
                    raise RuntimeError(f"selected audio mismatch: {key} {lang}")
        if any(row.get(k) != v for k, v in pair.items()) or row.get("provenance") != provenance:
            raise RuntimeError(f"checkpoint metadata/provenance mismatch: {key}")
        for lang in ("ja", "en"):
            if row.get(f"{lang}_tts_text") != pair[f"{lang}_text"]:
                raise RuntimeError(f"checkpoint text mismatch: {key} {lang}")
            attempts = row.get("attempts", {}).get(lang, [])
            ids = [a.get("attempt") for a in attempts]
            if len(set(ids)) != len(ids) or any(a not in range(config.tts.max_content_retries + 1) for a in ids):
                raise RuntimeError(f"invalid attempt IDs: {key} {lang}")
            if complete and (not attempts or any("qc_pass" not in a for a in attempts)):
                raise RuntimeError(f"incomplete QC record: {key} {lang}")
            for attempt in attempts:
                if attempt.get("generation_status") != "ok":
                    continue
                path = Path(attempt["wav_16k"])
                if not path.resolve().is_relative_to((config.run.output_dir / "audio" / "16k").resolve()):
                    raise RuntimeError(f"WAV outside output directory: {path}")
                try:
                    info = sf.info(path)
                    duration = info.frames / info.samplerate
                    valid = (info.format == "WAV" and info.subtype == "PCM_16" and info.channels == 1
                             and info.samplerate == 16000 and info.frames > 0
                             and math.isclose(duration, float(attempt["duration"]), abs_tol=1/16000)
                             and sha256_file(path) == attempt["sha256"])
                    if not valid:
                        raise ValueError("format, duration, or SHA256 mismatch")
                    if attempt.get("qc_pass") and not config.qc.min_duration_seconds <= duration <= config.qc.max_duration_seconds:
                        raise ValueError("passing audio has invalid duration")
                except Exception as error:
                    raise RuntimeError(f"invalid saved WAV; refusing resume: {path}: {error}") from error
    if complete and seen != set(expected):
        raise RuntimeError("completed manifest is missing expected pairs")
