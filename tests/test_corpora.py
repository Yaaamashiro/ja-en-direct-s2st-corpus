from __future__ import annotations

import io
import json
import tarfile
from dataclasses import replace
from pathlib import Path

import pytest

from s2st_corpus.config import CorpusSourceConfig, load_config
from s2st_corpus.corpora import (
    _extract,
    filter_pair,
    iter_jesc,
    iter_kftt,
    prepare_corpora,
    sha256_file,
)


def _archive_tree(archive: Path, files: dict[str, str]) -> None:
    with tarfile.open(archive, "w:gz") as handle:
        for name, text in files.items():
            payload = text.encode("utf-8")
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            handle.addfile(member, io.BytesIO(payload))


def _jesc_files() -> dict[str, str]:
    return {
        "split/train": (
            "Please call me when you arrive at the station.\t"
            "駅に着いたら私に電話してください。\n"
            "We will discuss the project again tomorrow morning.\t"
            "明日の朝、この計画についてもう一度話し合いましょう。\n"
        ),
        "split/dev": (
            "I finished work earlier than usual today.\t"
            "今日はいつもより早く仕事が終わりました。\n"
        ),
        "split/test": (
            "Could you show me the way to the library?\t"
            "図書館までの道を教えていただけますか。\n"
        ),
    }


def _kftt_files() -> dict[str, str]:
    base = "kftt-data-1.0/data/orig"
    values = {
        "train": (
            "この庭園は江戸時代に造られたと伝えられています。\n"
            "春になると多くの観光客が桜を見に訪れます。\n",
            "It is said that this garden was built during the Edo period.\n"
            "In spring, many tourists visit to see the cherry blossoms.\n",
        ),
        "dev": (
            "この寺院には多くの文化財が保存されています。\n",
            "Many cultural properties are preserved in this temple.\n",
        ),
        "test": (
            "この地域は古くから交通の要所として栄えました。\n",
            "This area prospered as a transportation hub for centuries.\n",
        ),
    }
    files: dict[str, str] = {}
    for split, (ja, en) in values.items():
        files[f"{base}/kyoto-{split}.ja"] = ja
        files[f"{base}/kyoto-{split}.en"] = en
    return files


def test_official_layout_readers_preserve_language_direction(tmp_path: Path) -> None:
    jesc_root = tmp_path / "jesc"
    kftt_root = tmp_path / "kftt"
    for root, files in ((jesc_root, _jesc_files()), (kftt_root, _kftt_files())):
        for relative, text in files.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")

    jesc = list(iter_jesc(jesc_root, ("train",)))
    kftt = list(iter_kftt(kftt_root, ("train",)))
    assert jesc[0][2].startswith("駅に")
    assert jesc[0][3].startswith("Please")
    assert kftt[0][2].startswith("この庭園")
    assert kftt[0][3].startswith("It is said")


@pytest.mark.parametrize(
    ("ja", "en", "reason"),
    [
        ("音楽", "Music", "subtitle_cue"),
        ("詳細はhttps://example.comです。", "Read the website for details.", "url_or_email"),
        ("短い", "Too short.", "too_short"),
        (
            "駅に着いたら私に電話してください。",
            "Please call me when you arrive at the station.",
            None,
        ),
    ],
)
def test_pair_filter(ja: str, en: str, reason: str | None) -> None:
    actual, *_ = filter_pair("jesc", ja, en)
    assert actual == reason


def test_archive_path_traversal_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.tar.gz"
    _archive_tree(archive, {"../escape.txt": "no"})
    with pytest.raises(RuntimeError, match="unsafe archive member"):
        _extract("unsafe", archive, tmp_path / "work", sha256_file(archive))


def test_prepare_downloads_both_corpora_and_builds_manifest(tmp_path: Path) -> None:
    jesc_archive = tmp_path / "fixtures" / "jesc.tar.gz"
    kftt_archive = tmp_path / "fixtures" / "kftt.tar.gz"
    jesc_archive.parent.mkdir()
    _archive_tree(jesc_archive, _jesc_files())
    _archive_tree(kftt_archive, _kftt_files())

    repository = Path(__file__).resolve().parents[1]
    base = load_config(repository / "configs" / "production-qwen17b.yaml")
    data = tmp_path / "data"
    stale_partial = data / "sources" / "jesc.tar.gz.part"
    stale_partial.parent.mkdir(parents=True)
    stale_partial.write_bytes(b"incomplete")
    config = replace(
        base,
        run=replace(
            base.run,
            input_jsonl=data / "input" / "pairs.jsonl",
            output_dir=data / "production",
        ),
        prepare=replace(
            base.prepare,
            sources_dir=data / "sources",
            work_dir=data / "interim",
            reports_dir=data / "reports",
            target_train_hours=0.001,
        ),
        sources={
            "jesc": CorpusSourceConfig(
                url=jesc_archive.as_uri(),
                archive_name="jesc.tar.gz",
                sha256=sha256_file(jesc_archive),
                version="fixture",
                license_id="CC-BY-SA-4.0",
            ),
            "kftt": CorpusSourceConfig(
                url=kftt_archive.as_uri(),
                archive_name="kftt.tar.gz",
                sha256=sha256_file(kftt_archive),
                version="fixture",
                license_id="CC-BY-SA-3.0",
            ),
        },
    )

    first = prepare_corpora(config)
    rows = [
        json.loads(line)
        for line in config.run.input_jsonl.read_text(encoding="utf-8").splitlines()
    ]
    assert first["reused"] is False
    assert {row["corpus"] for row in rows} == {"jesc", "kftt"}
    assert {row["split"] for row in rows} == {"train", "dev", "test"}
    assert all(row["ja_text"] and row["en_text"] for row in rows)
    assert not stale_partial.exists()

    second = prepare_corpora(config)
    assert second["reused"] is True
