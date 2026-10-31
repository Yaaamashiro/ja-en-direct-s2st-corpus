"""Validate exported text without reselecting or changing it."""
from __future__ import annotations

import json
import hashlib
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError

from .audio import sha256_file
from .config import AppConfig, ExternalPrepareConfig


def is_external(config: AppConfig) -> bool:
    return isinstance(getattr(config, "prepare", None), ExternalPrepareConfig)


def write_json(path: Path, value: dict) -> None:
    from .io import file_size, account_replace
    previous_bytes = file_size(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    account_replace(path, previous_bytes)


_VALIDATIONS = {}
_IDENTITIES = {}


def reset_external_cache():
    _VALIDATIONS.clear()
    _IDENTITIES.clear()


def _validation_key(config):
    return json.dumps(asdict(config.prepare), default=str, sort_keys=True)


def validate_external(config: AppConfig) -> dict:
    from .input_index import file_stamp, get_index
    p = config.prepare
    if not isinstance(p, ExternalPrepareConfig):
        raise ValueError("external manifest configuration required")
    key = _validation_key(config)
    index = get_index(p.manifest, config.run.num_shards)
    stamp = file_stamp(p.summary)
    if key in _VALIDATIONS:
        cached_stamp, cached = _VALIDATIONS[key]
        if stamp != cached_stamp:
            raise ValueError("export summary changed during this process")
        if index.digest != cached["manifest_sha256"]:
            raise ValueError("input manifest differs from validated snapshot")
        return deepcopy(cached)
    raw_summary = p.summary.read_bytes()
    summary = json.loads(raw_summary.decode("utf-8"))
    if summary.get("complete") is not True:
        raise ValueError("export summary.complete must be true")
    if summary.get("corpus") != p.expected_corpus:
        raise ValueError("export summary corpus mismatch")
    expected = {s: getattr(p, f"expected_{s}_pairs") for s in ("train", "dev", "test")}
    for name, count in {"total": p.expected_total_pairs, **expected}.items():
        if type(summary.get(f"{name}_pairs")) is not int or summary[f"{name}_pairs"] != count:
            raise ValueError(f"export summary {name}_pairs mismatch")
    if summary.get("output_sha256") != index.digest:
        raise ValueError("export manifest SHA256 mismatch")
    for pair in index.iter_rows():
        if pair["split"] not in expected:
            raise ValueError(f"invalid split: {pair['split']}")
        if pair["corpus"] != p.expected_corpus:
            raise ValueError(f"wrong corpus: {pair['pair_id']}")
        if type(pair.get("length_bin")) is not int:
            raise ValueError(f"length_bin must be an integer: {pair['pair_id']}")
    counts = index.counts
    if any(counts[s] != expected[s] for s in expected) or sum(counts.values()) != p.expected_total_pairs:
        raise ValueError(f"manifest counts mismatch: {dict(counts)}")
    index.check()
    if file_stamp(p.summary) != stamp:
        raise ValueError("export summary changed during validation")
    result = {"complete": True, "corpus": p.expected_corpus,
              "manifest": str(p.manifest), "manifest_sha256": index.digest,
              "summary": str(p.summary), "summary_sha256": hashlib.sha256(raw_summary).hexdigest(),
              "total_pairs": sum(counts.values()), **{f"{s}_pairs": counts[s] for s in expected}}
    _VALIDATIONS[key] = (stamp, result)
    return deepcopy(result)


def run_identity(config: AppConfig) -> dict:
    from .input_index import get_index
    validation = validate_external(config)
    index = get_index(config.run.input_jsonl, config.run.num_shards)
    key = (_validation_key(config), str(config.run.input_jsonl.resolve()),
           config.run.num_shards, config.run.save_native,
           json.dumps([asdict(config.tts), asdict(config.qc), asdict(config.asr)], sort_keys=True))
    if key in _IDENTITIES:
        if index.digest != _IDENTITIES[key]["run_input_sha256"]:
            raise ValueError("run input differs from validated snapshot")
        return deepcopy(_IDENTITIES[key])
    if config.run.input_jsonl != config.prepare.manifest:
        if sum(index.shard_counts) > 1000 or config.run.num_shards != 1:
            raise ValueError("alternate input is allowed only for <=1000-pair smoke runs")
        source = get_index(config.prepare.manifest, config.run.num_shards)
        for pair in index.iter_rows():
            original = source.lookup(pair["pair_id"])
            if original is None:
                raise ValueError("smoke input contains unknown records")
            if original != pair:
                raise ValueError("smoke input differs from exported record")
        source.check()
        index.check()
    identity = {"input": validation, "tts": asdict(config.tts),
                "qc": asdict(config.qc), "asr": asdict(config.asr),
                "num_shards": config.run.num_shards, "save_native": config.run.save_native,
                "run_input_sha256": index.digest}
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
    _IDENTITIES[key] = deepcopy(identity)
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


def verify_inputs(config: AppConfig) -> None:
    """Final full hashes, including changes hidden by restored size/mtime."""
    from .input_index import get_index
    validation = validate_external(config)
    expected = {config.prepare.manifest: validation["manifest_sha256"],
                config.prepare.summary: validation["summary_sha256"]}
    expected[config.run.input_jsonl] = get_index(config.run.input_jsonl, config.run.num_shards).digest
    for path, digest in expected.items():
        if sha256_file(path) != digest:
            raise ValueError(f"final input SHA256 mismatch: {path}")
    validate_external(config)  # also detect a change while final hashes were read
    get_index(config.run.input_jsonl, config.run.num_shards).check()
