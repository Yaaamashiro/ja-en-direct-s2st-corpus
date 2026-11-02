from __future__ import annotations

import re
import hashlib
import heapq
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

from .config import AppConfig
from .io import atomic_write_jsonl
from .manifest import iter_pairs
from .audio import sha256_file
from .external import write_json, is_external
from .input_index import cached_index


BENCHMARK_QUOTAS = {0: 20, 1: 150, 2: 400, 3: 330, 4: 100}


def _resolve_quotas(quotas, available, shortfall_policy, source_manifest):
    effective = {b: min(n, available[b]) for b, n in quotas.items()}
    deficit = sum(quotas.values()) - sum(effective.values())
    if not deficit:
        return effective
    details = {b: {"requested": n, "available": available[b]}
               for b, n in quotas.items() if available[b] < n}
    if shortfall_policy == "error" or sum(available[b] for b in quotas if quotas[b]) < sum(quotas.values()):
        raise ValueError(
            f"not enough training records for benchmark length-bin quotas: source={source_manifest}; "
            f"shortfalls={details}; train_length_bin_counts={dict(available)}; "
            f"requested_total={sum(quotas.values())}. "
            "Check the input path and export distribution; redistribution requires explicit opt-in."
        )
    # Redistribute only the deficit, retaining the requested weights. Integer
    # largest-remainder allocation with bin-ID tie breaks is deterministic.
    while deficit:
        eligible = [b for b in sorted(quotas) if quotas[b] and effective[b] < available[b]]
        weight = sum(quotas[b] for b in eligible)
        remaining = deficit
        for b in eligible:
            extra = min(available[b] - effective[b], remaining * quotas[b] // weight)
            effective[b] += extra
            deficit -= extra
        for b in sorted(eligible, key=lambda b: (-(remaining * quotas[b] % weight), b)):
            if deficit and effective[b] < available[b]:
                effective[b] += 1
                deficit -= 1
    return effective


def select_benchmark_pairs(source_manifest, destination, seed=42, quotas=None, *, shortfall_policy="error"):
    quotas = BENCHMARK_QUOTAS if quotas is None else quotas
    if shortfall_policy not in {"error", "redistribute"}:
        raise ValueError("invalid benchmark shortfall policy")
    if not quotas or any(type(b) is not int or type(n) is not int or n < 0 for b, n in quotas.items()) or not sum(quotas.values()):
        raise ValueError("benchmark quotas must be integer bins and nonnegative counts with a positive total")
    heaps = {b: [] for b in quotas}
    available = Counter()
    # Keep only a bounded sample, even when one bin must cover the full deficit.
    limits = {b: sum(quotas.values()) if shortfall_policy == "redistribute" and n else n
              for b, n in quotas.items()}
    index = cached_index(source_manifest)
    source_rows = index.iter_rows() if index is not None else iter_pairs(source_manifest)
    for ordinal, pair in enumerate(source_rows):
        b = pair.get("length_bin")
        if pair["split"] != "train":
            continue
        available[b] += 1
        if b not in heaps or not limits[b]:
            continue
        rank = int.from_bytes(hashlib.sha256(f"{seed}:{pair['pair_id']}".encode()).digest(), "big")
        item = (-rank, pair["pair_id"], ordinal, pair)
        heap = heaps[b]
        if len(heap) < limits[b]:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)
    effective = _resolve_quotas(quotas, available, shortfall_policy, source_manifest)
    if effective != quotas:
        print(f"[selection] length-bin quotas redistributed: requested={quotas} actual={effective}", flush=True)
    selected = sorted((item for b, heap in heaps.items()
                       for item in heapq.nlargest(effective[b], heap)), key=lambda item: item[2])
    rows = [item[3] for item in selected]
    atomic_write_jsonl(destination, rows)
    write_json(destination.with_suffix(".selection.json"), {
        "selection_method": "train_length_bin_sha256_ranking", "seed": seed,
        "length_bin_quotas": effective,
        "requested_length_bin_quotas": quotas,
        "shortfall_policy": shortfall_policy,
        "available_train_length_bin_counts": dict(available),
        "source_manifest_sha256": index.digest if index is not None else sha256_file(source_manifest),
        "selected_pair_ids": [r["pair_id"] for r in rows],
    })
    return rows


_PREFERRED_SLOTS = (
    ("jesc", "short"),
    ("jesc", "medium"),
    ("jesc", "long"),
    ("kftt", "medium"),
    ("kftt", "long"),
)
_LATIN_LETTER = re.compile(r"[A-Za-z]")
_NON_JAPANESE_CJK = re.compile(
    r"[你們嗎這發來會與從對裡還說國學體點聲樣讓]"
)
_TEST_NOISE = ("...", "…", "(", ")", "（", "）", "[", "]", "【", "】")


def _is_complete_sentence(pair: dict[str, Any]) -> bool:
    ja = pair["ja_text"].rstrip()
    en = pair["en_text"].rstrip()
    combined = ja + en
    return (
        ja.endswith(("。", "！", "？", "!", "?"))
        and en.endswith((".", "!", "?"))
        and not _LATIN_LETTER.search(ja)
        and not _NON_JAPANESE_CJK.search(ja)
        and not any(marker in combined for marker in _TEST_NOISE)
    )


def select_smoke_pairs(
    source_manifest: Path,
    destination: Path,
    pair_count: int,
    external: bool = False,
    benchmark_seed: int = 42,
    benchmark_shortfall: str = "error",
) -> list[dict[str, Any]]:
    if pair_count < 1:
        raise ValueError("pair_count must be positive")
    if external:
        if pair_count == 1000:
            return select_benchmark_pairs(source_manifest, destination, benchmark_seed,
                                          shortfall_policy=benchmark_shortfall)
        from itertools import islice
        # Export order, all splits, no sentence/length/content re-selection.
        index = cached_index(source_manifest)
        rows = list(islice(index.iter_rows() if index is not None else iter_pairs(source_manifest), pair_count))
        if len(rows) != pair_count:
            raise ValueError(f"only {len(rows)} pairs available; requested {pair_count}")
        atomic_write_jsonl(destination, rows)
        return rows

    preferred = list(_PREFERRED_SLOTS[:pair_count])
    selected: list[dict[str, Any] | None] = [None] * len(preferred)
    fallback: list[dict[str, Any]] = []
    fallback_limit = max(pair_count * 20, pair_count)

    for pair in iter_pairs(source_manifest):
        if pair["split"] != "train" or not _is_complete_sentence(pair):
            continue
        for index, (corpus, length_bin) in enumerate(preferred):
            if (
                selected[index] is None
                and pair["corpus"] == corpus
                and pair.get("length_bin") == length_bin
            ):
                selected[index] = pair
                break
        if len(fallback) < fallback_limit:
            fallback.append(pair)

    chosen = [pair for pair in selected if pair is not None]
    chosen_ids = {pair["pair_id"] for pair in chosen}
    for pair in fallback:
        if len(chosen) >= pair_count:
            break
        if pair["pair_id"] not in chosen_ids:
            chosen.append(pair)
            chosen_ids.add(pair["pair_id"])

    if len(chosen) != pair_count:
        raise RuntimeError(
            f"only {len(chosen)} training pairs are available for a "
            f"{pair_count}-pair smoke test"
        )

    rows: list[dict[str, Any]] = []
    for index, pair in enumerate(chosen):
        row = dict(pair)
        row["smoke_selection_index"] = index
        rows.append(row)
    atomic_write_jsonl(destination, rows)
    return rows


def smoke_app_config(config: AppConfig) -> AppConfig:
    maximum_seconds = config.smoke.pair_count * config.qc.max_duration_seconds
    smoke_run = replace(
        config.run,
        input_jsonl=config.smoke.input_jsonl,
        output_dir=config.smoke.output_dir,
        num_shards=1,
        max_pairs_per_shard=config.smoke.pair_count,
        checkpoint_every=config.run.checkpoint_every if is_external(config) else 1,
        minimum_free_gib=config.smoke.minimum_free_gib,
        maximum_output_gib=config.smoke.maximum_output_gib,
        expected_hours_per_language=maximum_seconds / 3600,
        save_native=False,
    )
    return replace(config, run=smoke_run)
