from pathlib import Path

from s2st_corpus.runner import _dtype, _materialize_hf_model, select_attempt


class _FakeCuda:
    def __init__(self, bf16_supported: bool) -> None:
        self._bf16_supported = bf16_supported

    def is_bf16_supported(self) -> bool:
        return self._bf16_supported


class _FakeTorch:
    float16 = "fp16"
    bfloat16 = "bf16"
    float32 = "fp32"

    def __init__(self, bf16_supported: bool) -> None:
        self.cuda = _FakeCuda(bf16_supported)


def test_select_attempt_prefers_lower_metric() -> None:
    attempts = [
        {"attempt": 0, "qc_pass": True, "metric": 0.08},
        {"attempt": 1, "qc_pass": True, "metric": 0.03},
    ]
    assert select_attempt(attempts)["attempt"] == 1


def test_select_attempt_prefers_attempt_zero_on_tie() -> None:
    attempts = [
        {"attempt": 1, "qc_pass": True, "metric": 0.03},
        {"attempt": 0, "qc_pass": True, "metric": 0.03},
    ]
    assert select_attempt(attempts)["attempt"] == 0


def test_select_attempt_returns_none_when_all_fail() -> None:
    assert select_attempt([{"attempt": 0, "qc_pass": False}]) is None


def test_auto_dtype_uses_bf16_when_supported() -> None:
    assert _dtype(_FakeTorch(True), "auto") == "bf16"


def test_auto_dtype_falls_back_to_fp16() -> None:
    assert _dtype(_FakeTorch(False), "auto") == "fp16"


def test_materialize_hf_model_returns_id_without_model_root(monkeypatch) -> None:
    monkeypatch.delenv("S2ST_MODEL_ROOT", raising=False)

    assert _materialize_hf_model("org/model", "revision") == "org/model"


def test_materialize_hf_model_downloads_complete_snapshot(
    monkeypatch, tmp_path: Path
) -> None:
    calls = []

    def fake_snapshot_download(*, repo_id: str, revision: str, local_dir: Path) -> None:
        calls.append((repo_id, revision, local_dir))
        for relative_path in (
            "config.json",
            "model.safetensors",
            "speech_tokenizer/config.json",
            "speech_tokenizer/preprocessor_config.json",
            "speech_tokenizer/model.safetensors",
        ):
            path = local_dir / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

    monkeypatch.setenv("S2ST_MODEL_ROOT", str(tmp_path))
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        fake_snapshot_download,
    )

    result = Path(_materialize_hf_model("org/model", "revision"))

    assert result == tmp_path / "org--model" / "revision"
    assert (result / ".s2st-complete").read_text(encoding="utf-8") == "revision\n"
    assert calls == [("org/model", "revision", result)]

    assert Path(_materialize_hf_model("org/model", "revision")) == result
    assert len(calls) == 1
