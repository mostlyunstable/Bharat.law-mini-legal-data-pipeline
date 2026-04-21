from __future__ import annotations

import re
from typing import Iterable

from bs4 import BeautifulSoup, NavigableString, Tag


def _text(el: Tag) -> str:
    return re.sub(r"\s+", " ", el.get_text(" ", strip=True)).strip()


def _emit_table(table: Tag) -> str:
    rows = []
    for tr in table.select("tr"):
        cells = [re.sub(r"\s+", " ", c.get_text(" ", strip=True)) for c in tr.select("th,td")]
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows).strip()


def html_to_markdownish(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for t in soup(["script", "style", "noscript"]):
        t.decompose()

    body = (
        soup.select_one("main, article, [role=main], .pkp_structure_main, #content, .content")
        or soup.body
        or soup
    )
    if isinstance(body, Tag) and body.name.lower() == "a" and not body.get_text(strip=True):
        body = body.find_parent() or soup.body or soup

    lines: list[str] = []
    block_markers = {"h1", "h2", "h3", "h4", "h5", "h6", "p", "ul", "ol", "table"}

    def is_noise_container(tag: Tag) -> bool:
        name = tag.name.lower()
        if name in {"nav", "footer", "header", "aside"}:
            return True
        role = (tag.get("role") or "").lower()
        if role in {"navigation", "banner", "contentinfo"}:
            return True
        klass = " ".join(tag.get("class") or []).lower()
        return any(k in klass for k in ["nav", "menu", "breadcrumb", "footer", "header", "cookie"])

    def is_flat_text_container(tag: Tag) -> bool:
        if tag.name.lower() not in {"div", "span", "section"}:
            return False
        if is_noise_container(tag):
            return False
        # If it contains any structural blocks, let recursion handle it.
        return tag.find(list(block_markers), recursive=True) is None

    def walk(node: Tag) -> None:
        for child in node.children:
            if isinstance(child, NavigableString):
                continue
            if not isinstance(child, Tag):
                continue

            name = child.name.lower()
            if is_noise_container(child):
                continue
            if name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
                level = int(name[1])
                txt = _text(child)
                if txt:
                    lines.append(f"{'#' * level} {txt}")
                continue
            if name == "p":
                txt = _text(child)
                if txt:
                    lines.append(txt)
                continue
            if name == "a":
                txt = _text(child)
                href = (child.get("href") or "").strip()
                if txt and len(txt) >= 3:
                    # Keep link text; add href only when it's informative.
                    if href and not href.startswith("#"):
                        lines.append(f"- {txt} ({href})")
                    else:
                        lines.append(f"- {txt}")
                continue
            if name in {"ul", "ol"}:
                for li in child.find_all("li", recursive=False):
                    txt = _text(li)
                    if txt:
                        lines.append(f"- {txt}")
                continue
            if name == "table":
                txt = _emit_table(child)
                if txt:
                    lines.append(txt)
                continue
            if name == "br":
                lines.append("")
                continue
            if is_flat_text_container(child):
                txt = _text(child)
                if txt and len(txt) >= 20:
                    lines.append(txt)
                continue

            walk(child)

    walk(body)

    out = "\n\n".join([ln for ln in lines if ln is not None])
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out
