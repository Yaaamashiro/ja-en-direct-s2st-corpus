import json
from dataclasses import replace
from pathlib import Path

import pytest

from s2st_corpus import cli, external, input_index, kokoro_backend, runner
from s2st_corpus.audio import sha256_file
from s2st_corpus.config import load_config
from s2st_corpus.manifest import stable_shard, write_checkpoint


@pytest.fixture
def indexed_corpus(tmp_path, monkeypatch):
    external.reset_external_cache()
    input_index.reset_indexes()
    monkeypatch.delenv("S2ST_TTS_BATCH_SIZE", raising=False)
    monkeypatch.delenv("S2ST_ASR_BATCH_SIZE", raising=False)
    monkeypatch.setenv("S2ST_INPUT_INDEX_DIR", str(tmp_path / "local-index"))
    c = load_config(Path("configs/production-kokoro-jparacrawl.yaml"))
    rows = [{"pair_id": f"p{i}", "corpus": "jparacrawl_v3_filtered", "split": "dev" if i < 6 else "train",
             "ja_text": f"Ａ　そのまま？{i}", "en_text": f"Exact  text {i}", "length_bin": i % 5,
             "source_line": i} for i in range(600)]
    path = tmp_path / "source.jsonl"
    write_checkpoint(path, rows)
    summary = tmp_path / "summary.json"
    external.write_json(summary, {"complete": True, "corpus": "jparacrawl_v3_filtered", "total_pairs": 600,
                                 "train_pairs": 594, "dev_pairs": 6, "test_pairs": 0, "output_sha256": sha256_file(path)})
    c = replace(c, prepare=replace(c.prepare, manifest=path, summary=summary, expected_total_pairs=600,
                                  expected_train_pairs=594, expected_dev_pairs=6),
                run=replace(c.run, input_jsonl=path, output_dir=tmp_path / "output"))
    monkeypatch.setattr(external, "version", lambda n: "0.9.4")
    monkeypatch.setattr(kokoro_backend, "dictionary_identity", lambda: {"version": "test"})
    yield c, rows
    external.reset_external_cache()
    input_index.reset_indexes()


def count_reads(monkeypatch, path):
    reads = []
    original = Path.open
    def open_file(p, mode="r", *args, **kwargs):
        if p.resolve() == path.resolve() and "r" in mode:
            reads.append(mode)
        return original(p, mode, *args, **kwargs)
    monkeypatch.setattr(Path, "open", open_file)
    return reads


def test_128_shards_share_single_drive_read_and_identity(indexed_corpus, monkeypatch):
    c, rows = indexed_corpus
    reads = count_reads(monkeypatch, c.run.input_jsonl)
    first = external.ensure_identity(c)
    assert first["run_input_sha256"] == first["input"]["manifest_sha256"]
    for shard in range(c.run.num_shards):
        assert external.ensure_identity(c) == first
        assert input_index.pairs_for_config(c, shard) == [r for r in rows if stable_shard(r["pair_id"], c.run.num_shards) == shard]
    assert runner.plan(c)["pairs"] == 600
    assert reads == ["rb"]  # validation + hash + partition from one source pass
    assert not input_index.index_for_config(c).path.is_relative_to(c.run.output_dir)


def test_status_many_completed_shards_reads_source_once(indexed_corpus, monkeypatch, capsys):
    c, _ = indexed_corpus
    identity = external.ensure_identity(c)
    for shard in range(100):
        ids = input_index.index_for_config(c).pair_ids(shard, c.run.max_pairs_per_shard)
        path = c.run.output_dir / "manifests/qc" / f"shard-{shard:05d}-of-00128.jsonl"
        write_checkpoint(path, [{"pair_id": p, "provenance": identity} for p in sorted(ids)])
    # CLI resets caches just like a fresh process after reconnect.
    reads = count_reads(monkeypatch, c.run.input_jsonl)
    monkeypatch.setattr(cli, "load_config", lambda p: c)
    monkeypatch.setattr("sys.argv", ["cli", "status"])
    cli.main()
    status = json.loads(capsys.readouterr().out)
    assert status["completed_shards"] == 100 and status["next_shard"] == 100
    assert reads == ["rb"]


@pytest.mark.parametrize("target", ["manifest", "summary"])
def test_midprocess_change_stops_without_rebuilding(indexed_corpus, target):
    c, _ = indexed_corpus
    external.ensure_identity(c)
    path = getattr(c.prepare, target)
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="changed|differs"):
        external.ensure_identity(c)


def test_final_hash_verifies_even_when_file_stamp_is_hidden(indexed_corpus, monkeypatch):
    c, _ = indexed_corpus
    external.ensure_identity(c)
    old_stamp = input_index.file_stamp(c.run.input_jsonl)
    actual_stamp = input_index.file_stamp
    monkeypatch.setattr(input_index, "file_stamp", lambda p: old_stamp if Path(p) == c.run.input_jsonl else actual_stamp(p))
    data = c.run.input_jsonl.read_bytes()
    c.run.input_jsonl.write_bytes(data.replace(b'Exact', b'Other', 1))
    # Cheap intermediate checks intentionally cannot see this simulated change.
    external.ensure_identity(c)
    with pytest.raises(ValueError, match="final input SHA256 mismatch"):
        external.verify_inputs(c)


def test_final_hash_is_only_additional_full_read(indexed_corpus, monkeypatch):
    c, _ = indexed_corpus
    reads = count_reads(monkeypatch, c.run.input_jsonl)
    external.ensure_identity(c)
    external.verify_inputs(c)
    assert reads == ["rb", "rb"]


def test_smoke_repartitions_and_looks_up_source_locally(indexed_corpus, monkeypatch, tmp_path):
    c, rows = indexed_corpus
    reads = count_reads(monkeypatch, c.run.input_jsonl)
    external.validate_external(c)
    subset = tmp_path / "smoke.jsonl"
    write_checkpoint(subset, rows[:10])
    smoke = replace(c, run=replace(c.run, input_jsonl=subset, output_dir=tmp_path / "smoke-output", num_shards=1))
    external.ensure_identity(smoke)
    assert input_index.pairs_for_config(smoke, 0) == rows[:10]
    assert reads == ["rb"]


def test_restart_revalidates_and_rebuilds(indexed_corpus, monkeypatch):
    c, _ = indexed_corpus
    reads = count_reads(monkeypatch, c.run.input_jsonl)
    original = external.ensure_identity(c)
    old_path = input_index.index_for_config(c).path
    external.reset_external_cache()
    input_index.reset_indexes()
    assert not old_path.exists()
    assert external.ensure_identity(c) == original
    assert reads == ["rb", "rb"]


def test_index_enforces_pair_limit_and_invalid_shard(indexed_corpus):
    c, _ = indexed_corpus
    index = input_index.index_for_config(c)
    with pytest.raises(ValueError, match="shard index"):
        index.pairs(128, 3000)
    with pytest.raises(RuntimeError, match="max_pairs_per_shard"):
        index.pairs(0, 0)


def test_crlf_and_blank_lines_use_original_file_hash(indexed_corpus):
    c, _ = indexed_corpus
    raw = c.run.input_jsonl.read_bytes().replace(b"\n", b"\r\n") + b"\r\n"
    c.run.input_jsonl.write_bytes(raw)
    summary = json.loads(c.prepare.summary.read_text())
    summary["output_sha256"] = sha256_file(c.run.input_jsonl)
    external.write_json(c.prepare.summary, summary)
    validation = external.validate_external(c)
    assert validation["manifest_sha256"] == summary["output_sha256"]
    assert validation["total_pairs"] == 600


def test_change_during_index_build_is_rejected(indexed_corpus, monkeypatch):
    c, _ = indexed_corpus
    real_iter = input_index.iter_pairs
    def changing_iter(path, **kwargs):
        for i, row in enumerate(real_iter(path, **kwargs)):
            yield row
            if i == 0:
                with path.open("ab") as handle:
                    handle.write(b"\n")
    monkeypatch.setattr(input_index, "iter_pairs", changing_iter)
    with pytest.raises(ValueError, match="changed|differs"):
        external.validate_external(c)
    assert not list((c.prepare.manifest.parent / "local-index").glob("session-*"))


def test_resetting_only_index_cannot_bypass_validated_digest(indexed_corpus):
    c, _ = indexed_corpus
    external.ensure_identity(c)
    input_index.reset_indexes()
    raw = c.run.input_jsonl.read_bytes()
    c.run.input_jsonl.write_bytes(raw.replace(b'Exact', b'Other', 1))
    with pytest.raises(ValueError, match="differs"):
        external.ensure_identity(c)
