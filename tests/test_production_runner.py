from s2st_corpus.runner import select_attempt


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
