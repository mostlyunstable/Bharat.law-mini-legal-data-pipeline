from __future__ import annotations

import json
import math
import re
import heapq
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
from bs4 import BeautifulSoup
from urllib.robotparser import RobotFileParser

from utils import (
    append_jsonl,
    canonicalize_url,
    resolve_url,
    sleep_polite,
    url_to_hash,
    utc_now_iso,
    write_text,
)


@dataclass(frozen=True)
class CrawlConfig:
    user_agent: str = "BharatLawPipelineBot/0.1 (take-home; contact: local)"
    max_depth: int = 3
    max_pages: int = 25
    politeness_delay_s: float = 1.0
    timeout_s: float = 25.0
    max_retries: int = 3
    backoff_base_s: float = 1.0
    enable_js: bool = False
    js_on_http_status: tuple[int, ...] = (403, 429)
    js_when_html_small_chars: int = 2500
    # Heuristic URL filtering to avoid wasting crawl budget on auth and nav pages.
    deny_url_regexes: tuple[str, ...] = (
        r"/login(?:/|$)",
        r"/user(?:/|$)",
        r"/register(?:/|$)",
        r"/lostPassword(?:/|$)",
        r"/search(?:/|$)",
        r"[\?&]source=",
    )


class RobotsCache:
    def __init__(self, client: httpx.Client, user_agent: str):
        self._client = client
        self._user_agent = user_agent
        self._cache: dict[str, RobotFileParser] = {}

    def allowed(self, url: str) -> bool:
        parts = urlsplit(url)
        key = f"{parts.scheme}://{parts.netloc}"
        rp = self._cache.get(key)
        if rp is None:
            robots_url = f"{key}/robots.txt"
            rp = RobotFileParser()
            rp.set_url(robots_url)
            try:
                resp = self._client.get(robots_url, headers={"User-Agent": self._user_agent}, follow_redirects=True)
                if resp.status_code == 200:
                    rp.parse(resp.text.splitlines())
                else:
                    rp.parse([])
            except Exception:
                rp.parse([])
            self._cache[key] = rp
        return rp.can_fetch(self._user_agent, url)


def _retry_sleep(base_s: float, attempt: int) -> float:
    # Exponential backoff with gentle cap.
    return min(base_s * (2**attempt), 15.0)


def _content_ext(content_type: str | None, url: str) -> tuple[str, str]:
    ct = (content_type or "").lower()
    if "pdf" in ct or url.lower().endswith(".pdf"):
        return ("application/pdf", "pdf")
    return ("text/html", "html")


def _same_domain(a: str, b: str) -> bool:
    return urlsplit(a).netloc.lower() == urlsplit(b).netloc.lower()


def _should_skip(url: str, config: CrawlConfig) -> bool:
    for pat in config.deny_url_regexes:
        if re.search(pat, url, flags=re.IGNORECASE):
            return True
    return False


def _url_score(url: str) -> int:
    u = url.lower()
    score = 10
    if u.endswith(".pdf") or "pdf" in u:
        score += 100
    if "/article/" in u:
        score += 80
    if "/issue/view/" in u:
        score += 60
    if "/issue/" in u:
        score += 40
    if "?" in u:
        score -= 5
    return score


def _looks_like_block_page(html: str) -> bool:
    h = (html or "").lower()
    if "access denied" in h and ("you don't have permission" in h or "errors.edgesuite.net" in h):
        return True
    if "request blocked" in h:
        return True
    return False


def _looks_sparse_html(html: str) -> bool:
    h = (html or "").lower()
    # OJS issue pages can render a ToC client-side; server response can be sparse.
    if "obj_issue_toc" in h and "obj_article_summary" not in h:
        return True
    return False


def _fetch_with_playwright(url: str, user_agent: str, timeout_s: float) -> str:
    try:
        from playwright.sync_api import sync_playwright  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "Playwright not installed. Install with `pip install playwright` and run `playwright install chromium`."
        ) from e

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(user_agent=user_agent)
        page = context.new_page()
        page.set_default_timeout(int(timeout_s * 1000))
        page.goto(url, wait_until="networkidle")
        content = page.content()
        context.close()
        browser.close()
        return content


def crawl(seed_urls: dict[str, str], project_root: Path, config: CrawlConfig) -> dict[str, Any]:
    raw_dir = project_root / "raw"
    index_path = project_root / "crawl_index.jsonl"
    raw_dir.mkdir(parents=True, exist_ok=True)
    index_path.write_text("", encoding="utf-8")

    visited: set[str] = set()
    # Max-heap via negative scores.
    queue: list[tuple[int, str, int]] = []
    for _, u in seed_urls.items():
        cu = canonicalize_url(u)
        if _should_skip(cu, config):
            continue
        heapq.heappush(queue, (-_url_score(cu), cu, 0))

    with httpx.Client(timeout=config.timeout_s, follow_redirects=True) as client:
        robots = RobotsCache(client, config.user_agent)

        pages_crawled = 0
        while queue and pages_crawled < config.max_pages:
            _, url, depth = heapq.heappop(queue)
            if url in visited:
                continue
            visited.add(url)

            if depth > config.max_depth:
                continue

            if not robots.allowed(url):
                append_jsonl(
                    index_path,
                    {
                        "url": url,
                        "status": None,
                        "content_type": None,
                        "used_js": False,
                        "timestamp": utc_now_iso(),
                        "path_to_raw": None,
                        "blocked_by_robots": True,
                    },
                )
                continue

            sleep_polite(config.politeness_delay_s)

            used_js = False
            status: int | None = None
            content_type: str | None = None
            raw_path: str | None = None
            error: str | None = None
            js_error: str | None = None

            for attempt in range(config.max_retries + 1):
                try:
                    resp = client.get(url, headers={"User-Agent": config.user_agent})
                    status = resp.status_code
                    content_type, ext = _content_ext(resp.headers.get("content-type"), url)

                    if status == 200:
                        url_hash = url_to_hash(url)
                        raw_file = raw_dir / f"{url_hash}.{ext}"
                        if ext == "pdf":
                            raw_file.write_bytes(resp.content)
                        else:
                            write_text(raw_file, resp.text)
                            if _looks_like_block_page(resp.text):
                                error = "Blocked (access denied page)"
                                status = 403
                        raw_path = str(raw_file.relative_to(project_root))
                    else:
                        error = f"HTTP {status}"
                    break
                except Exception as e:  # noqa: BLE001
                    error = f"{type(e).__name__}: {e}"
                    if attempt < config.max_retries:
                        sleep_polite(_retry_sleep(config.backoff_base_s, attempt))
                        continue
                    break

            # Optional JS-render fallback for blocked or JS-heavy pages.
            if config.enable_js and content_type == "text/html":
                needs_js = False
                if status in config.js_on_http_status:
                    needs_js = True
                elif status == 200 and raw_path:
                    try:
                        raw_html = (project_root / raw_path).read_text(encoding="utf-8", errors="ignore")
                        if len(raw_html) < config.js_when_html_small_chars or _looks_sparse_html(raw_html):
                            needs_js = True
                    except Exception:
                        pass

                if needs_js:
                    try:
                        rendered = _fetch_with_playwright(url, config.user_agent, config.timeout_s)
                        if _looks_like_block_page(rendered):
                            raise RuntimeError("Blocked (access denied page)")
                        url_hash = url_to_hash(url)
                        raw_file = raw_dir / f"{url_hash}.html"
                        write_text(raw_file, rendered)
                        raw_path = str(raw_file.relative_to(project_root))
                        status = 200
                        used_js = True
                        error = None
                    except Exception as e:  # noqa: BLE001
                        js_error = f"{type(e).__name__}: {e}"
                        used_js = True

            append_jsonl(
                index_path,
                {
                    "url": url,
                    "status": status,
                    "content_type": content_type,
                    "used_js": used_js,
                    "timestamp": utc_now_iso(),
                    "path_to_raw": raw_path,
                    "error": error,
                    "js_error": js_error,
                },
            )

            if status != 200 or not raw_path:
                continue

            pages_crawled += 1

            # Link extraction for HTML only.
            if content_type == "text/html":
                try:
                    soup = BeautifulSoup((project_root / raw_path).read_text(encoding="utf-8", errors="ignore"), "lxml")
                    for a in soup.select("a[href]"):
                        next_url = resolve_url(url, a.get("href"))
                        if not next_url:
                            continue
                        if not _same_domain(url, next_url):
                            continue
                        if _should_skip(next_url, config):
                            continue
                        if next_url in visited:
                            continue
                        heapq.heappush(queue, (-_url_score(next_url), next_url, depth + 1))
                except Exception:
                    pass

    return {"pages_crawled": pages_crawled, "crawl_index": str(index_path)}
