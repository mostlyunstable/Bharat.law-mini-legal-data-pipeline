from __future__ import annotations

from utils import canonicalize_url, estimate_tokens, url_to_hash


def test_canonicalize_url_drops_fragment() -> None:
    assert canonicalize_url("https://Example.com/a#x") == "https://example.com/a"


def test_url_hash_length() -> None:
    assert len(url_to_hash("https://example.com")) == 12
    assert len(url_to_hash("https://example.com", length=20)) == 20


def test_estimate_tokens_nonzero() -> None:
    assert estimate_tokens("hello world " * 100) > 0

