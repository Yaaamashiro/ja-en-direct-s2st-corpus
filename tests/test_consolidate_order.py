from dataclasses import replace
from pathlib import Path

import pytest

from s2st_corpus.audio import sha256_file
from s2st_corpus.config import load_config
from s2st_corpus.manifest import read_jsonl, stable_shard, write_checkpoint
from s2st_corpus.runner import consolidate


@pytest.fixture
def corpus(tmp_path):
    config = load_config(Path("configs/production-qwen17b.yaml"))
    config = replace(config, run=replace(
        config.run, input_jsonl=tmp_path / "input/pairs.jsonl",
        output_dir=tmp_path / "production", num_shards=3,
    ))
    pairs = [
        {"pair_id": f"p{i}", "corpus": "jesc", "split": "dev" if i == 4 else "train",
         "ja_text": f"Ａ　そのままの文{i}？！", "en_text": f"Exact  text {i}?!"}
        for i in range(5)
    ]
    write_checkpoint(config.run.input_jsonl, pairs)
    qc = {
        p["pair_id"]: {
            **p, "qc_status": "rejected" if i in (1, 3) else "accepted",
            "ja_duration": 1.25, "en_duration": 1.5,
            "ja_wav_16k": f"unchanged/ja/{p['pair_id']}.wav",
            "en_wav_16k": f"unchanged/en/{p['pair_id']}.wav",
            "attempts": {"ja": [{"metric": 0.01}], "en": [{"metric": 0.02}]},
        } for i, p in enumerate(pairs)
    }
    # Real hash partitions, deliberately reversed within each partition.
    shard_rows = [[qc[p["pair_id"]] for p in reversed(pairs)
                   if stable_shard(p["pair_id"], config.run.num_shards) == shard]
                  for shard in range(config.run.num_shards)]
    assert [r["pair_id"] for group in shard_rows for r in group] != [p["pair_id"] for p in pairs]
    for shard, rows in enumerate(shard_rows):
        write_checkpoint(shard_path(config, shard), rows)
    return config, qc


def shard_path(config, shard):
    return config.run.output_dir / "manifests/qc" / f"shard-{shard:05d}-of-{config.run.num_shards:05d}.jsonl"


def releases(config):
    return config.run.output_dir / "manifests/releases"


def drop_pair(config, pair_id):
    shard = stable_shard(pair_id, config.run.num_shards)
    path = shard_path(config, shard)
    write_checkpoint(path, [r for r in read_jsonl(path) if r["pair_id"] != pair_id])


def test_all_follows_input_order_and_preserves_qc_rows(corpus):
    config, qc = corpus
    summary = consolidate(config)
    assert read_jsonl(releases(config) / "all.jsonl") == [qc[f"p{i}"] for i in range(5)]
    assert summary["complete"] is True
    assert summary["release_order"] == "input_manifest"
    assert summary["missing_pair_count"] == 0
    assert summary["total_pairs"] == 5


def test_accepted_subset_preserves_input_order(corpus):
    config, qc = corpus
    summary = consolidate(config)
    assert read_jsonl(releases(config) / "accepted.jsonl") == [qc[k] for k in ("p0", "p2", "p4")]
    assert (summary["accepted_pairs"], summary["rejected_pairs"]) == (3, 2)


def test_release_bytes_and_hashes_are_deterministic(corpus):
    config, _ = corpus
    consolidate(config)
    paths = [releases(config) / name for name in ("all.jsonl", "accepted.jsonl")]
    before = [(p.read_bytes(), sha256_file(p)) for p in paths]
    consolidate(config)
    assert [(p.read_bytes(), sha256_file(p)) for p in paths] == before
    # Even changes to ordering within QC shards must not change release bytes.
    for shard in range(config.run.num_shards):
        path = shard_path(config, shard)
        write_checkpoint(path, list(reversed(read_jsonl(path))))
    consolidate(config)
    assert [(p.read_bytes(), sha256_file(p)) for p in paths] == before


def test_complete_rejects_missing_pair_even_when_all_shards_exist(corpus):
    config, _ = corpus
    drop_pair(config, "p2")
    with pytest.raises(RuntimeError, match="input pair_id values are missing"):
        consolidate(config)
    assert not releases(config).exists()


@pytest.mark.parametrize("allow_incomplete", [False, True])
def test_extra_pair_is_always_rejected(corpus, allow_incomplete):
    config, qc = corpus
    path = shard_path(config, 0)
    write_checkpoint(path, [*read_jsonl(path), {**qc["p0"], "pair_id": "extra"}])
    with pytest.raises(RuntimeError, match="extra pair_id"):
        consolidate(config, allow_incomplete=allow_incomplete)
    assert not releases(config).exists()


@pytest.mark.parametrize("same_shard", [False, True])
def test_duplicate_pair_is_rejected(corpus, same_shard):
    config, qc = corpus
    original = stable_shard("p0", config.run.num_shards)
    target = original if same_shard else (original + 1) % config.run.num_shards
    path = shard_path(config, target)
    write_checkpoint(path, [*read_jsonl(path), qc["p0"]])
    with pytest.raises(RuntimeError, match="duplicate pair_id"):
        consolidate(config)
    assert not releases(config).exists()


def test_incomplete_release_with_missing_shard_is_canonical(corpus):
    config, qc = corpus
    removed_shard = stable_shard("p0", config.run.num_shards)
    shard_path(config, removed_shard).unlink()
    summary = consolidate(config, allow_incomplete=True)
    expected = [qc[f"p{i}"] for i in range(5) if stable_shard(f"p{i}", config.run.num_shards) != removed_shard]
    assert read_jsonl(releases(config) / "all.jsonl") == expected
    assert read_jsonl(releases(config) / "accepted.jsonl") == [r for r in expected if r["qc_status"] == "accepted"]
    assert summary["complete"] is False
    assert summary["missing_shards"] == [removed_shard]
    assert summary["missing_pair_count"] == 5 - len(expected)


def test_incomplete_with_missing_pair_cannot_be_marked_complete(corpus):
    config, qc = corpus
    drop_pair(config, "p2")
    summary = consolidate(config, allow_incomplete=True)
    assert summary["complete"] is False
    assert summary["missing_shards"] == []
    assert summary["missing_pair_count"] == 1
    assert read_jsonl(releases(config) / "all.jsonl") == [qc[k] for k in ("p0", "p1", "p3", "p4")]


def test_strict_run_still_rejects_missing_shards(corpus):
    config, _ = corpus
    shard_path(config, 0).unlink()
    with pytest.raises(RuntimeError, match="shard manifests are missing"):
        consolidate(config)
    assert not releases(config).exists()


def test_split_releases_counts_and_hours(corpus):
    config, qc = corpus
    summary = consolidate(config)
    assert read_jsonl(releases(config) / "train.jsonl") == [qc["p0"], qc["p2"]]
    assert read_jsonl(releases(config) / "dev.jsonl") == [qc["p4"]]
    assert (releases(config) / "test.jsonl").read_bytes() == b""
    assert (summary["train"]["total"], summary["train"]["accepted"], summary["train"]["rejected"]) == (4, 2, 2)
    assert summary["train"]["acceptance_rate"] == .5
    assert summary["dev"]["accepted"] == 1
    assert summary["test"]["total"] == 0
    assert summary["train_ja_hours"] == pytest.approx(2 * 1.25 / 3600)
    assert summary["dev_en_hours"] == pytest.approx(1.5 / 3600)


def test_test_split_is_separate(corpus):
    config, qc = corpus
    pairs = read_jsonl(config.run.input_jsonl)
    pairs[-1]["split"] = qc["p4"]["split"] = "test"
    write_checkpoint(config.run.input_jsonl, pairs)
    path = shard_path(config, stable_shard("p4", config.run.num_shards))
    write_checkpoint(path, [qc[r["pair_id"]] for r in read_jsonl(path)])
    summary = consolidate(config)
    assert read_jsonl(releases(config) / "test.jsonl") == [qc["p4"]]
    assert read_jsonl(releases(config) / "dev.jsonl") == []
    assert summary["test"]["accepted"] == 1


def test_publish_failure_invalidates_old_completion_marker(corpus, monkeypatch):
    config, _ = corpus
    consolidate(config)
    original = Path.replace
    def fail_mid_publish(path, target):
        if path.name == "dev.jsonl.tmp":
            raise OSError("simulated publication interruption")
        return original(path, target)
    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", fail_mid_publish)
        with pytest.raises(OSError, match="interruption"):
            consolidate(config)
    assert not (releases(config) / "summary.json").exists()
    assert not list(releases(config).glob("*.tmp"))
    assert consolidate(config)["complete"] is True


def test_consolidate_reads_only_one_shard_at_a_time(corpus, monkeypatch):
    from s2st_corpus import release
    config, _ = corpus
    reads = []
    original = release.read_jsonl
    def read(path):
        rows = original(path)
        reads.append(len(rows))
        assert len(rows) <= 3
        return rows
    monkeypatch.setattr(release, "read_jsonl", read)
    assert consolidate(config)["total_pairs"] == 5
    assert len(reads) == config.run.num_shards
