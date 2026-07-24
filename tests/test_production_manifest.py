import json
from pathlib import Path

import pytest

from s2st_corpus.manifest import (
    pairs_for_shard,
    pcm16_storage_gib,
    plan_manifest,
    stable_shard,
)


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _rows(count: int) -> list[dict[str, str]]:
    return [
        {
            "pair_id": f"pair_{index:04d}",
            "corpus": "jesc" if index % 2 else "kftt",
            "split": "train",
            "ja_text": f"日本語の例文{index}です。",
            "en_text": f"This is English example {index}.",
        }
        for index in range(count)
    ]


def test_stable_sharding_assigns_every_pair_exactly_once(tmp_path: Path) -> None:
    path = tmp_path / "pairs.jsonl"
    rows = _rows(80)
    _write_rows(path, rows)
    selected = [
        pair["pair_id"]
        for shard in range(8)
        for pair in pairs_for_shard(path, shard, 8, 100)
    ]
    assert sorted(selected) == sorted(row["pair_id"] for row in rows)
    assert stable_shard("pair_0001", 8) == stable_shard("pair_0001", 8)


def test_plan_reports_counts(tmp_path: Path) -> None:
    path = tmp_path / "pairs.jsonl"
    _write_rows(path, _rows(20))
    result = plan_manifest(path, 4)
    assert result["pairs"] == 20
    assert result["corpora"] == {"jesc": 10, "kftt": 10}
    assert sum(result["shard_counts"]) == 20


def test_duplicate_pair_id_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "pairs.jsonl"
    rows = _rows(2)
    rows[1]["pair_id"] = rows[0]["pair_id"]
    _write_rows(path, rows)
    with pytest.raises(ValueError, match="duplicate pair_id"):
        plan_manifest(path, 2)


def test_pcm16_storage_estimate() -> None:
    assert pcm16_storage_gib(300) == pytest.approx(64.37, abs=0.01)
    assert pcm16_storage_gib(300, retry_fraction=0.10) == pytest.approx(
        70.81, abs=0.01
    )
