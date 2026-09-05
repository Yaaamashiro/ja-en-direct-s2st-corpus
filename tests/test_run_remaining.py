import json
import sys
from types import SimpleNamespace

import pytest

from s2st_corpus import cli


def test_remaining_skips_completed_and_resumes_after_error(tmp_path, monkeypatch, capsys):
    config = SimpleNamespace(run=SimpleNamespace(num_shards=4, output_dir=tmp_path))
    directory = tmp_path / "manifests/qc"
    directory.mkdir(parents=True)

    def path(index):
        return directory / f"shard-{index:05d}-of-00004.jsonl"

    path(0).touch()
    calls = []
    fail = True

    def run(config, index):
        calls.append(index)
        if index == 2 and fail:
            raise RuntimeError("interrupted")
        path(index).touch()
        return path(index)

    releases = []
    monkeypatch.setattr(cli, "load_config", lambda p: config)
    monkeypatch.setattr(cli, "run_shard", run)
    monkeypatch.setattr(cli, "consolidate", lambda c: releases.append(True) or {"complete": True})
    monkeypatch.setattr(sys, "argv", ["cli", "run-remaining-shards"])
    with pytest.raises(RuntimeError, match="interrupted"):
        cli.main()
    assert calls == [1, 2] and not releases
    fail = False
    cli.main()
    assert calls == [1, 2, 2, 3] and releases == [True]
    capsys.readouterr()
    monkeypatch.setattr(sys, "argv", ["cli", "status"])
    cli.main()
    result = json.loads(capsys.readouterr().out)
    assert result["completed_shards"] == 4
    assert result["next_shard"] is None


def test_single_shard_reports_post_run_status(tmp_path, monkeypatch, capsys):
    config = SimpleNamespace(run=SimpleNamespace(num_shards=2, output_dir=tmp_path))
    monkeypatch.setattr(cli, "load_config", lambda p: config)
    monkeypatch.setattr(cli, "run_shard", lambda c, i: tmp_path / "result.jsonl")
    monkeypatch.setattr(sys, "argv", ["cli", "run-next-shard"])
    cli.main()
    result = json.loads(capsys.readouterr().out)
    assert result["completed_shards"] == 1
    assert result["next_shard"] == 1
