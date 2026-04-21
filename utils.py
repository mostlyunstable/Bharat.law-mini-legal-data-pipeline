from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator
from urllib.parse import urljoin, urlsplit, urlunsplit


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonicalize_url(url: str) -> str:
    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower() or "http"
    netloc = parts.netloc.lower()
    path = re.sub(r"/{2,}", "/", parts.path or "/")
    query = parts.query
    # Drop fragments; keep query as-is (re-ordering can be lossy for some sites).
    return urlunsplit((scheme, netloc, path, query, ""))


def url_to_hash(url: str, length: int = 12) -> str:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return digest[:length]


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    safe_mkdir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return iter(())

    def _iter() -> Iterator[dict[str, Any]]:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

    return _iter()


def write_text(path: Path, text: str) -> None:
    safe_mkdir(path.parent)
    path.write_text(text, encoding="utf-8")


def estimate_tokens(text: str) -> int:
    # Simple, dependency-free approximation: ~1 token ≈ 0.75 words.
    words = len(re.findall(r"\S+", text))
    return int(words / 0.75) if words else 0


def sleep_polite(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


def resolve_url(base_url: str, href: str) -> str | None:
    href = (href or "").strip()
    if not href:
        return None
    if href.startswith("mailto:") or href.startswith("javascript:"):
        return None
    return canonicalize_url(urljoin(base_url, href))


def detect_language(text: str) -> str:
    # The assignment sources are English; keep this lightweight and robust.
    try:
        from langdetect import detect  # type: ignore

        return detect(text) if text.strip() else "en"
    except Exception:
        return "en"


@dataclass(frozen=True)
class Metrics:
    pages_crawled: int
    chunks_created: int
    avg_tokens_per_chunk: float
    ner_f1: float | None

