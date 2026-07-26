from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from typing import Any

from .config import AppConfig
from .io import atomic_write_jsonl
from .manifest import iter_pairs


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
) -> list[dict[str, Any]]:
    if pair_count < 1:
        raise ValueError("pair_count must be positive")

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
        checkpoint_every=1,
        minimum_free_gib=config.smoke.minimum_free_gib,
        maximum_output_gib=config.smoke.maximum_output_gib,
        expected_hours_per_language=maximum_seconds / 3600,
        save_native=False,
    )
    return replace(config, run=smoke_run)
