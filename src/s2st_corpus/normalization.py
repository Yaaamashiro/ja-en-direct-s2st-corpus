from __future__ import annotations

import html
import re
import unicodedata
from collections.abc import Callable


_CONTROL_OR_ZERO_WIDTH = re.compile(
    r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f\u200b-\u200f\ufeff]"
)
_WHITESPACE = re.compile(r"\s+")
_JA_IGNORED = re.compile(
    r"[\s、。！？!?,.・「」『』（）()\[\]【】〈〉《》…―:：;；'\"“”‘’]"
)
_EN_PUNCTUATION = re.compile(r"[^\w\s']")


def minimal_normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", html.unescape(text))
    normalized = _CONTROL_OR_ZERO_WIDTH.sub("", normalized)
    return _WHITESPACE.sub(" ", normalized).strip()


def japanese_reading_normalize(
    text: str, g2p: Callable[..., str]
) -> str:
    normalized = minimal_normalize(text)
    reading = g2p(normalized, kana=True)
    reading = unicodedata.normalize("NFKC", reading)
    return _JA_IGNORED.sub("", reading)


def english_basic_normalize(text: str) -> str:
    normalized = minimal_normalize(text).lower()
    normalized = _EN_PUNCTUATION.sub(" ", normalized)
    return _WHITESPACE.sub(" ", normalized).strip()


def whisper_english_normalize(text: str, tokenizer: object) -> str:
    normalizer = getattr(tokenizer, "normalize", None)
    if callable(normalizer):
        return _WHITESPACE.sub(" ", normalizer(text)).strip()
    normalizer = getattr(tokenizer, "_normalize", None)
    if callable(normalizer):
        return _WHITESPACE.sub(" ", normalizer(text)).strip()
    return english_basic_normalize(text)
