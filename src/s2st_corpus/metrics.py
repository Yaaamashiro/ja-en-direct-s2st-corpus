"""Separate historical attempt metrics from wall time of individual invocations."""
from collections import Counter
from contextvars import ContextVar
from statistics import mean

ACTIVE = ContextVar("s2st_measurements", default=None)


def add_time(name, seconds):
    active = ACTIVE.get()
    if active is not None:
        active[name] = active.get(name, 0.0) + seconds


def summarize(records):
    result = {}
    reasons = Counter()
    total_duration = total_tts = total_asr = 0.0
    for lang, metric in (("ja", "cer"), ("en", "wer")):
        attempts = [a for r in records for a in r["attempts"][lang]]
        duration = sum(a.get("duration", 0.0) for a in attempts if a.get("generation_status") == "ok")
        tts = sum(a.get("synthesis_seconds", 0.0) for a in attempts)
        values = [a[metric] for a in attempts if metric in a]
        result[lang] = {"generated_seconds": duration, "generated_hours": duration / 3600,
                        "tts_wall_seconds": tts, "tts_rtf": tts / duration if duration else None,
                        f"mean_{metric}": mean(values) if values else None,
                        "evaluated_attempts": len(values)}
        total_duration += duration
        total_tts += tts
        total_asr += sum(a.get("asr_seconds", 0.0) for a in attempts)
        for a in attempts:
            if a.get("generation_status") != "ok":
                reasons["generation_failures"] += 1
                if "empty_audio" in a.get("generation_error", ""):
                    reasons["empty_audio"] += 1
            if "duration_out_of_range" in a.get("qc_reasons", []):
                reasons["duration_failures"] += 1
            if "asr_error" in a:
                reasons["asr_failures"] += 1
    return {"attempt_metrics": result, "tts_wall_seconds": total_tts,
            "tts_rtf_total": total_tts / total_duration if total_duration else None,
            "asr_inference_seconds": total_asr,
            "failures": {k: reasons[k] for k in ("generation_failures", "empty_audio", "duration_failures", "asr_failures")}}
