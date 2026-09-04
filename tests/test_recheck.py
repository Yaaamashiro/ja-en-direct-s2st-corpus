import json
import sys
from types import SimpleNamespace

import pytest

from s2st_corpus import runner
from s2st_corpus.manifest import write_checkpoint


@pytest.mark.parametrize("fail", [False, True])
def test_recheck_preserves_audio_and_backs_up_reports(tmp_path, monkeypatch, fail):
    wav = tmp_path / "existing.wav"
    wav.write_bytes(b"existing audio")
    row = {"pair_id": "one", "attempts": {
        lang: [{"attempt": 0, "generation_status": "ok", "wav_16k": str(wav),
                "qc_pass": False, "qc_reasons": ["asr_failed"], "asr_error": "old"}]
        for lang in ("ja", "en")
    }}
    path = tmp_path / "manifests/qc/shard-00000-of-00001.jsonl"
    write_checkpoint(path, [row])
    old = path.read_bytes()
    config = SimpleNamespace(run=SimpleNamespace(output_dir=tmp_path, num_shards=1))
    monkeypatch.setitem(sys.modules, "pyopenjtalk", SimpleNamespace(g2p=None))
    monkeypatch.setattr(runner, "_load_whisper", lambda c: (None, None, None, None))
    monkeypatch.setattr(runner, "_release_cuda", lambda t: None)
    calls = []

    def evaluate(config, record, language, attempt, *args):
        assert "asr_error" not in attempt and "qc_pass" not in attempt
        calls.append(language)
        if fail:
            raise RuntimeError("decoder broken")
        attempt.update(qc_pass=True, qc_reasons=[])

    monkeypatch.setattr(runner, "_evaluate_attempt", evaluate)
    monkeypatch.setattr(runner, "_finalize_record", lambda c, r: dict(r, qc_status="accepted"))
    if fail:
        with pytest.raises(RuntimeError, match="decoder broken"):
            runner.recheck_shard(config, 0)
        assert path.read_bytes() == old
    else:
        assert runner.recheck_shard(config, 0) == path
        assert calls == ["ja", "en"]
        assert json.loads(path.read_text())["qc_status"] == "accepted"
    assert wav.read_bytes() == b"existing audio"
    backups = list((tmp_path / "qc-backups").glob("*/manifests/qc/*.jsonl"))
    assert len(backups) == 1 and backups[0].read_bytes() == old


def test_asr_environment_error_stops_before_content_retry():
    config = SimpleNamespace(
        qc=SimpleNamespace(min_duration_seconds=0.7, max_duration_seconds=30),
        asr=SimpleNamespace(do_sample=False, num_beams=1, condition_on_prev_tokens=False),
    )

    def transcriber(*args, **kwargs):
        raise RuntimeError("Could not load libtorchcodec")

    with pytest.raises(RuntimeError, match="ASR/QC failed"):
        runner._evaluate_attempt(config, {"pair_id": "one"}, "en",
                                 {"generation_status": "ok", "duration": 2, "wav_16k": "a.wav"},
                                 transcriber, None, None)
