from __future__ import annotations

import json
from pathlib import Path

from s2st_corpus.config import load_config
from s2st_corpus.smoke import select_smoke_pairs, smoke_app_config


def _row(
    pair_id: str,
    corpus: str,
    length_bin: str,
    split: str = "train",
) -> dict[str, object]:
    return {
        "pair_id": pair_id,
        "corpus": corpus,
        "split": split,
        "ja_text": "これは日本語のテスト文です。",
        "en_text": f"This is the English sentence for {pair_id}.",
        "length_bin": length_bin,
    }


def test_smoke_selection_covers_both_corpora_and_lengths(tmp_path: Path) -> None:
    source = tmp_path / "full.jsonl"
    destination = tmp_path / "smoke" / "pairs.jsonl"
    rows = [
        _row("jesc_short", "jesc", "short"),
        _row("jesc_medium", "jesc", "medium"),
        _row("jesc_long", "jesc", "long"),
        _row("kftt_medium", "kftt", "medium"),
        _row("kftt_long", "kftt", "long"),
        _row("ignored_dev", "jesc", "short", split="dev"),
    ]
    source.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    selected = select_smoke_pairs(source, destination, 5)

    assert [row["pair_id"] for row in selected] == [
        "jesc_short",
        "jesc_medium",
        "jesc_long",
        "kftt_medium",
        "kftt_long",
    ]
    assert destination.is_file()


def test_smoke_selection_skips_incomplete_subtitle_fragments(tmp_path: Path) -> None:
    source = tmp_path / "full.jsonl"
    destination = tmp_path / "smoke.jsonl"
    rows = [
        {
            **_row("fragment", "jesc", "short"),
            "ja_text": "これは途中で切れた字幕で、",
            "en_text": "this subtitle is cut off,",
        },
        _row("complete", "jesc", "short"),
    ]
    source.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    selected = select_smoke_pairs(source, destination, 1)

    assert selected[0]["pair_id"] == "complete"


def test_smoke_selection_skips_pronunciation_markup(tmp_path: Path) -> None:
    source = tmp_path / "full.jsonl"
    destination = tmp_path / "smoke.jsonl"
    rows = [
        {
            **_row("markup", "jesc", "short"),
            "ja_text": "この略称はAPIです。",
        },
        _row("plain", "jesc", "short"),
    ]
    source.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    selected = select_smoke_pairs(source, destination, 1)

    assert selected[0]["pair_id"] == "plain"


def test_smoke_profile_is_separate_and_small() -> None:
    repository = Path(__file__).resolve().parents[1]
    config = load_config(repository / "configs" / "production-qwen17b.yaml")

    smoke = smoke_app_config(config)

    assert smoke.run.input_jsonl == config.smoke.input_jsonl
    assert smoke.run.output_dir == config.smoke.output_dir
    assert smoke.run.output_dir != config.run.output_dir
    assert smoke.run.num_shards == 1
    assert smoke.run.max_pairs_per_shard == 5
    assert smoke.run.maximum_output_gib == 1
    assert smoke.tts == config.tts
    assert smoke.asr == config.asr
    assert smoke.qc == config.qc
