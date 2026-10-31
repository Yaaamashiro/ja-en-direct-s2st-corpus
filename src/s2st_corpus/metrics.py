"""Streaming attempt counters and per-invocation wall measurements."""
from collections import Counter
from contextvars import ContextVar

ACTIVE = ContextVar("s2st_measurements", default=None)


def add_time(name, seconds):
    active = ACTIVE.get()
    if active is not None:
        active[name] = active.get(name, 0.0) + seconds


class AttemptMetrics:
    def __init__(self):
        self.languages = {lang: Counter() for lang in ("ja", "en")}
        self.failures = Counter()
        self.duration_bins = {lang: Counter({str(i): 0 for i in range(5)}) for lang in ("ja", "en")}
        self.length_bins = Counter()
        self.generated_pairs = 0

    def add(self, row):
        length_bin = str(row.get("length_bin", "unknown"))
        self.length_bins[length_bin] += 1
        attempts = row.get("attempts", {})
        if all(any(a.get("generation_status") == "ok" for a in attempts.get(lang, [])) for lang in ("ja", "en")):
            self.generated_pairs += 1
        for lang, metric in (("ja", "cer"), ("en", "wer")):
            sums = self.languages[lang]
            for a in attempts.get(lang, []):
                if a.get("generation_status") == "ok":
                    sums["duration"] += a.get("duration", 0.0)
                else:
                    self.failures["generation_failures"] += 1
                    if "empty_audio" in a.get("generation_error", ""):
                        self.failures["empty_audio"] += 1
                sums["tts"] += a.get("synthesis_seconds", 0.0)
                sums["asr"] += a.get("asr_seconds", 0.0)
                if metric in a:
                    sums["metric_sum"] += a[metric]
                    sums["evaluated"] += 1
                if "duration_out_of_range" in a.get("qc_reasons", []):
                    self.failures["duration_failures"] += 1
                    self.duration_bins[lang][length_bin] += 1
                if "asr_error" in a:
                    self.failures["asr_failures"] += 1

    def result(self):
        result = {}
        for lang, metric in (("ja", "cer"), ("en", "wer")):
            s = self.languages[lang]
            result[lang] = {"generated_seconds": s["duration"], "generated_hours": s["duration"] / 3600,
                           "tts_wall_seconds": s["tts"], "tts_rtf": s["tts"] / s["duration"] if s["duration"] else None,
                           f"mean_{metric}": s["metric_sum"] / s["evaluated"] if s["evaluated"] else None,
                           "evaluated_attempts": s["evaluated"]}
        duration = sum(s["duration"] for s in self.languages.values())
        tts = sum(s["tts"] for s in self.languages.values())
        bins = self.duration_bins["ja"] + self.duration_bins["en"]
        return {"attempt_metrics": result, "tts_wall_seconds": tts,
                "tts_rtf_total": tts / duration if duration else None,
                "asr_inference_seconds": sum(s["asr"] for s in self.languages.values()),
                "generated_pairs": self.generated_pairs, "length_bin_distribution": dict(self.length_bins),
                "duration_failures_by_length_bin": {k: bins[k] for k in sorted(set(self.duration_bins["ja"]) | set(self.duration_bins["en"]))},
                "duration_failures_by_language_and_length_bin": {k: dict(v) for k, v in self.duration_bins.items()},
                "failures": {k: self.failures[k] for k in ("generation_failures", "empty_audio", "duration_failures", "asr_failures")}}


def summarize(records):
    accumulator = AttemptMetrics()
    for row in records:
        accumulator.add(row)
    return accumulator.result()
