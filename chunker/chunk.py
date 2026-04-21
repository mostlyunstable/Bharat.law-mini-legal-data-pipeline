from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from utils import append_jsonl, estimate_tokens, read_jsonl, safe_mkdir


@dataclass(frozen=True)
class ChunkConfig:
    min_tokens: int = 400
    max_tokens: int = 800
    overlap_tokens: int = 80
    drop_small_docs: bool = False
    merge_small_docs: bool = False


_PAGE_RE = re.compile(r"^##\s+Page\s+(\d+)\s*$", re.IGNORECASE)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def _split_paragraphs(text: str) -> list[tuple[int, str]]:
    paragraphs: list[tuple[int, str]] = []
    pos = 0
    for block in re.split(r"\n{2,}", text):
        block = block.strip()
        if not block:
            pos += 2
            continue
        idx = text.find(block, pos)
        if idx == -1:
            idx = pos
        paragraphs.append((idx, block))
        pos = idx + len(block)
    return paragraphs


def _apply_overlap(prev_text: str, overlap_tokens: int) -> str:
    if overlap_tokens <= 0:
        return ""
    words = re.findall(r"\S+", prev_text)
    if not words:
        return ""
    tail = words[-int(overlap_tokens * 0.75) :]  # reverse of estimate_tokens
    return " ".join(tail).strip()


def _explode_large_block(offset: int, block: str, max_tokens: int) -> list[tuple[int, str]]:
    t = estimate_tokens(block)
    if t <= max_tokens:
        return [(offset, block)]

    max_words = max(1, int(max_tokens * 0.75))
    words = list(re.finditer(r"\S+", block))
    if not words:
        return [(offset, block)]

    parts: list[tuple[int, str]] = []
    for i in range(0, len(words), max_words):
        start = words[i].start()
        end = words[min(i + max_words, len(words)) - 1].end()
        piece = block[start:end].strip()
        if piece:
            parts.append((offset + start, piece))
    return parts if parts else [(offset, block)]


def _domain(url: str) -> str:
    try:
        return (urlsplit(url).netloc or "").lower()
    except Exception:
        return ""


def _merge_small_docs(
    docs: list[dict[str, Any]],
    min_tokens: int,
    project_root: Path,
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    buffer: dict[str, Any] | None = None

    def flush() -> None:
        nonlocal buffer
        if not buffer:
            return
        merged.append(buffer)
        buffer = None

    for d in docs:
        url = d["url"]
        md_path = project_root / d["path_to_text"]
        text = md_path.read_text(encoding="utf-8", errors="ignore")
        tok = estimate_tokens(text)

        if buffer is None:
            buffer = {
                "url": url,
                "url_hash": d["url_hash"],
                "title": d.get("title") or url,
                "source_urls": [url],
                "source_url_hashes": [d["url_hash"]],
                "source_domains": [_domain(url)],
                "text": text,
                "token_estimate": tok,
            }
            continue

        if int(buffer["token_estimate"]) < min_tokens:
            buffer["source_urls"].append(url)
            buffer["source_url_hashes"].append(d["url_hash"])
            buffer["source_domains"].append(_domain(url))
            buffer["text"] = (buffer["text"].rstrip() + "\n\n---\n\n" + text.lstrip()).strip() + "\n"
            buffer["token_estimate"] = estimate_tokens(buffer["text"])
        else:
            flush()
            buffer = {
                "url": url,
                "url_hash": d["url_hash"],
                "title": d.get("title") or url,
                "source_urls": [url],
                "source_url_hashes": [d["url_hash"]],
                "source_domains": [_domain(url)],
                "text": text,
                "token_estimate": tok,
            }

    flush()
    return merged


def chunk_all(project_root: Path, config: ChunkConfig) -> dict[str, Any]:
    normalized_index = project_root / "normalized_index.jsonl"
    chunks_dir = project_root / "chunks"
    chunks_path = chunks_dir / "chunks.jsonl"
    safe_mkdir(chunks_dir)

    # Reset chunks output for reproducibility.
    chunks_path.write_text("", encoding="utf-8")

    chunks_created = 0
    token_sum = 0

    docs = list(read_jsonl(normalized_index))
    if config.merge_small_docs:
        docs = _merge_small_docs(docs, config.min_tokens, project_root)

    for doc in docs:
        url = doc["url"]
        url_hash = doc["url_hash"]
        title = doc.get("title") or url
        if config.merge_small_docs:
            text = doc["text"]
            source_urls = doc.get("source_urls") or [url]
            source_url_hashes = doc.get("source_url_hashes") or [url_hash]
            source_domains = doc.get("source_domains") or [_domain(url)]
        else:
            md_path = project_root / doc["path_to_text"]
            text = md_path.read_text(encoding="utf-8", errors="ignore")
            source_urls = [url]
            source_url_hashes = [url_hash]
            source_domains = [_domain(url)]

        current_page: int | None = None
        section_stack: list[tuple[int, str]] = []

        paragraphs = _split_paragraphs(text)
        current_chunk_parts: list[tuple[int, str, list[str], int | None]] = []
        current_chunk_tokens = 0
        prev_chunk_text = ""
        doc_chunks: list[dict[str, Any]] = []

        def flush() -> None:
            nonlocal current_chunk_parts, current_chunk_tokens, prev_chunk_text, doc_chunks
            if not current_chunk_parts:
                return

            start = current_chunk_parts[0][0]
            end = current_chunk_parts[-1][0] + len(current_chunk_parts[-1][1])
            section_path = current_chunk_parts[0][2]
            page_no = current_chunk_parts[0][3]
            chunk_text = "\n\n".join([p[1] for p in current_chunk_parts]).strip()
            token_est = estimate_tokens(chunk_text)

            doc_chunks.append(
                {
                    "url": url,
                    "title": title,
                    "source_urls": source_urls,
                    "source_url_hashes": source_url_hashes,
                    "source_domains": source_domains,
                    "section_path": section_path,
                    "page_no": page_no,
                    "char_start": start,
                    "char_end": end,
                    "token_estimate": token_est,
                    "text": chunk_text,
                }
            )
            prev_chunk_text = chunk_text
            current_chunk_parts = []
            current_chunk_tokens = 0

        for offset, para in paragraphs:
            # Track headings/pages.
            m_page = _PAGE_RE.match(para.splitlines()[0].strip())
            if m_page:
                current_page = int(m_page.group(1))

            m_heading = _HEADING_RE.match(para.splitlines()[0].strip())
            if m_heading:
                level = len(m_heading.group(1))
                heading_text = m_heading.group(2).strip()
                while section_stack and section_stack[-1][0] >= level:
                    section_stack.pop()
                section_stack.append((level, heading_text))

            section_path = [h[1] for h in section_stack]
            for part_offset, part_text in _explode_large_block(offset, para, config.max_tokens):
                t = estimate_tokens(part_text)

                if current_chunk_tokens + t > config.max_tokens and current_chunk_tokens >= config.min_tokens:
                    flush()
                    overlap = _apply_overlap(prev_chunk_text, config.overlap_tokens)
                    if overlap:
                        current_chunk_parts.append((part_offset, overlap, section_path, current_page))
                        current_chunk_tokens += estimate_tokens(overlap)

                current_chunk_parts.append((part_offset, part_text, section_path, current_page))
                current_chunk_tokens += t

                if current_chunk_tokens >= config.max_tokens:
                    flush()

        flush()

        # If the final chunk is tiny, merge it into the previous chunk to better match the target size.
        if len(doc_chunks) >= 2 and doc_chunks[-1]["token_estimate"] < int(config.min_tokens * 0.6):
            last = doc_chunks.pop()
            prev = doc_chunks[-1]
            prev["text"] = (prev["text"].rstrip() + "\n\n" + last["text"].lstrip()).strip()
            prev["char_end"] = max(int(prev["char_end"]), int(last["char_end"]))
            prev["token_estimate"] = estimate_tokens(prev["text"])

        if config.drop_small_docs and len(doc_chunks) == 1 and int(doc_chunks[0]["token_estimate"]) < config.min_tokens:
            doc_chunks = []

        for i, ch in enumerate(doc_chunks, start=1):
            chunk_id = f"{url_hash}:{i:04d}"
            ch_out = {"chunk_id": chunk_id, **ch}
            append_jsonl(chunks_path, ch_out)
            chunks_created += 1
            token_sum += int(ch_out["token_estimate"])

    avg_tokens = (token_sum / chunks_created) if chunks_created else 0.0
    return {"chunks_created": chunks_created, "avg_tokens_per_chunk": avg_tokens}
