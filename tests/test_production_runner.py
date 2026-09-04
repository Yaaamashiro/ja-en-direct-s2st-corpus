from s2st_corpus.runner import _dtype, select_attempt


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
