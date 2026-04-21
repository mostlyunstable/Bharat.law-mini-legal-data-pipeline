from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

from parsers.html_to_md import html_to_markdownish
from utils import append_jsonl, detect_language, read_jsonl, url_to_hash, write_text


@dataclass(frozen=True)
class NormalizeConfig:
    include_source_header: bool = True


def _extract_title_from_html(html: str) -> str | None:
    soup = BeautifulSoup(html, "lxml")
    title = soup.title.get_text(strip=True) if soup.title else None
    return title or None


def _extract_main_html(html: str) -> str:
    try:
        from readability import Document  # type: ignore

        return Document(html).summary(html_partial=True)
    except Exception:
        return html


def _normalize_html(url: str, html: str, config: NormalizeConfig) -> tuple[str, str]:
    title = _extract_title_from_html(html) or url
    main_html = _extract_main_html(html)
    body = html_to_markdownish(main_html)
    if len(body.strip()) < 200:
        body = html_to_markdownish(html)
    if config.include_source_header:
        md = f"# {title}\n\nSource: {url}\n\n{body}\n"
    else:
        md = f"# {title}\n\n{body}\n"
    return title, md


def _normalize_pdf(url: str, pdf_path: Path, config: NormalizeConfig) -> tuple[str, str]:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    title = (reader.metadata.title if reader.metadata else None) or url
    parts: list[str] = [f"# {title}", f"\nSource: {url}\n"] if config.include_source_header else [f"# {title}\n"]
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        parts.append(f"## Page {i}\n\n{text.strip()}\n")
    return title, "\n\n".join(parts).strip() + "\n"


def normalize_all(project_root: Path, config: NormalizeConfig) -> dict[str, Any]:
    crawl_index = project_root / "crawl_index.jsonl"
    normalized_dir = project_root / "normalized"
    normalized_index = project_root / "normalized_index.jsonl"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    normalized_index.write_text("", encoding="utf-8")

    docs_normalized = 0
    for row in read_jsonl(crawl_index):
        if row.get("status") != 200:
            continue
        url = row["url"]
        raw_rel = row.get("path_to_raw")
        if not raw_rel:
            continue
        raw_path = project_root / raw_rel
        url_hash = url_to_hash(url)

        source_type = "pdf" if raw_path.suffix.lower() == ".pdf" else "html"
        try:
            if source_type == "pdf":
                title, md = _normalize_pdf(url, raw_path, config)
            else:
                html = raw_path.read_text(encoding="utf-8", errors="ignore")
                title, md = _normalize_html(url, html, config)
        except Exception:
            continue

        out_path = normalized_dir / f"{url_hash}.md"
        write_text(out_path, md)
        lang = detect_language(md[:4000])

        append_jsonl(
            normalized_index,
            {
                "url": url,
                "url_hash": url_hash,
                "source_type": source_type,
                "title": title,
                "detected_language": lang,
                "char_count": len(md),
                "path_to_text": str(out_path.relative_to(project_root)),
            },
        )
        docs_normalized += 1

    return {"docs_normalized": docs_normalized, "normalized_index": str(normalized_index)}
