"""Validate exported text without reselecting or changing it."""
from __future__ import annotations

import json
from itertools import islice
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError

from .audio import sha256_file
from .config import AppConfig, ExternalPrepareConfig
from .manifest import iter_pairs


def is_external(config: AppConfig) -> bool:
    return isinstance(getattr(config, "prepare", None), ExternalPrepareConfig)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_external(config: AppConfig) -> dict:
    p = config.prepare
    if not isinstance(p, ExternalPrepareConfig):
        raise ValueError("external manifest configuration required")
    summary = json.loads(p.summary.read_text(encoding="utf-8"))
    if summary.get("complete") is not True:
        raise ValueError("export summary.complete must be true")
    if summary.get("corpus") != p.expected_corpus:
        raise ValueError("export summary corpus mismatch")
    expected = {s: getattr(p, f"expected_{s}_pairs") for s in ("train", "dev", "test")}
    for key, count in {"total": p.expected_total_pairs, **expected}.items():
        if type(summary.get(f"{key}_pairs")) is not int or summary[f"{key}_pairs"] != count:
            raise ValueError(f"export summary {key}_pairs mismatch")
    digest = sha256_file(p.manifest)
    if summary.get("output_sha256") != digest:
        raise ValueError("export manifest SHA256 mismatch")
    counts = Counter()
    # iter_pairs enforces unique, filename-safe IDs and nonempty text, streaming.
    for pair in iter_pairs(p.manifest):
        if pair["split"] not in expected:
            raise ValueError(f"invalid split: {pair['split']}")
        if pair["corpus"] != p.expected_corpus:
            raise ValueError(f"wrong corpus: {pair['pair_id']}")
        if type(pair.get("length_bin")) is not int:
            raise ValueError(f"length_bin must be an integer: {pair['pair_id']}")
        counts[pair["split"]] += 1
    if any(counts[s] != expected[s] for s in expected) or sum(counts.values()) != p.expected_total_pairs:
        raise ValueError(f"manifest counts mismatch: {dict(counts)}")
    return {"complete": True, "corpus": p.expected_corpus,
            "manifest": str(p.manifest), "manifest_sha256": digest,
            "summary": str(p.summary), "summary_sha256": sha256_file(p.summary),
            "total_pairs": sum(counts.values()), **{f"{s}_pairs": counts[s] for s in expected}}


def run_identity(config: AppConfig) -> dict:
    if config.run.input_jsonl != config.prepare.manifest:
        subset = {p["pair_id"]: p for p in islice(iter_pairs(config.run.input_jsonl), 1001)}
        if len(subset) > 1000 or config.run.num_shards != 1:
            raise ValueError("alternate input is allowed only for <=1000-pair smoke runs")
        found = set()
        for pair in iter_pairs(config.prepare.manifest):
            if pair["pair_id"] in subset:
                if subset[pair["pair_id"]] != pair:
                    raise ValueError("smoke input differs from exported record")
                found.add(pair["pair_id"])
        if found != set(subset):
            raise ValueError("smoke input contains unknown records")
    identity = {"input": validate_external(config), "tts": asdict(config.tts),
                "qc": asdict(config.qc), "asr": asdict(config.asr),
                "num_shards": config.run.num_shards, "save_native": config.run.save_native,
                "run_input_sha256": sha256_file(config.run.input_jsonl)}
    if config.tts.backend == "kokoro":
        installed = version("kokoro")
        if installed != config.tts.package_version:
            raise RuntimeError(f"kokoro {installed} installed; expected {config.tts.package_version}")
        identity["kokoro_package_version"] = installed
        from .kokoro_backend import dictionary_identity
        identity["japanese_dictionary"] = dictionary_identity()
        dependencies = {}
        for name in ("misaki", "pyopenjtalk", "pyopenjtalk-plus", "spacy", "en-core-web-sm",
                     "transformers", "jiwer", "numpy", "soxr", "phonemizer-fork", "espeakng-loader", "unidic"):
            try:
                dependencies[name] = version(name)
            except PackageNotFoundError:
                dependencies[name] = None
        identity["runtime_dependencies"] = dependencies
    return identity


def ensure_identity(config: AppConfig) -> dict:
    """One writer per output directory. Refuse mixing inputs/voices/settings."""
    identity = run_identity(config)
    path = config.run.output_dir / "provenance.json"
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != identity:
            raise RuntimeError("output provenance mismatch; restore the original config or choose a new output_dir")
    else:
        if any((config.run.output_dir / d).exists() for d in ("audio", "manifests")):
            raise RuntimeError("existing output has no provenance; use a separate output_dir")
        write_json(path, identity)
    return identity
