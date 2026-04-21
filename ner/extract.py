from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from utils import append_jsonl, read_jsonl


@dataclass(frozen=True)
class NerConfig:
    spacy_model: str = "en_core_web_sm"


_SECTION_REF = re.compile(
    r"\b(?:section|sec\.?|s\.)\s*\d+[A-Za-z]?(?:\s*\([0-9a-zA-Z]+\))*"  # Section 4A, Section 2(1)(a)
    r"(?=\W|$)"
    r"|\bOrder\s+[IVXLCDM]+\s+Rule\s+\d+(?:\s*\([0-9a-zA-Z]+\))*"  # Order I Rule 8
    r"(?=\W|$)"
    r"|\bRule\s+\d+(?:\s*\([0-9a-zA-Z]+\))*"  # Rule 8
    r"(?=\W|$)"
    r"|\bSub-?section\s*\(\s*[0-9a-zA-Z]+\s*\)\s+of\s+Section\s+\d+[A-Za-z]?"  # Sub-section (2) of Section 1
    r"(?=\W|$)"
    r"|\bClause\s*\(\s*[0-9a-zA-Z]+\s*\)\s+of\s+Sub-?section\s*\(\s*[0-9a-zA-Z]+\s*\)\s+of\s+Section\s+\d+[A-Za-z]?"  # Clause (b) of Sub-section (1) of Section 7
    r"(?=\W|$)",
    re.IGNORECASE,
)
_DATE = re.compile(
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}\s+(?:Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sep|Sept|September|Oct|October|Nov|November|Dec|December)\s+\d{4}\b",
    re.IGNORECASE,
)
_MONEY = re.compile(r"(?:₹\s?\d[\d,]*(?:\.\d+)?|\b(?:INR|Rs\.?|USD|US\$)\s?\d[\d,]*(?:\.\d+)?)")
# In the provided gold dataset, notifications are typically like "S.O. 432(E)" or "G.S.R. 237(E)".
_NOTIFICATION_CODE = re.compile(r"\b(?:S\.O\.|G\.S\.R\.)\s*\d+(?:\([A-Z]\))?(?=\W|$)", re.IGNORECASE)

_ACT_NAME = re.compile(
    r"\b(?!According\b)(?!Under\b)(?!Per\b)(?!Pursuant\b)(?!As\b)(?!In\b)(?!Of\b)(?!Read\b)(?:"
    r"Code\s+of\s+[A-Z][A-Za-z&\-]+(?:\s+[A-Z][A-Za-z&\-]+){0,6},\s*\d{4}"
    r"|"
    r"[A-Z][A-Za-z&\-]+(?:\s+(?:[A-Z][A-Za-z&\-]+|and|to|of|the|&)){0,12}\s+Act,\s*\d{4}"
    r")(?=\W|$)"
)


def _regex_entities(label: str, pattern: re.Pattern[str], text: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in pattern.finditer(text):
        out.append({"label": label, "text": m.group(0), "start": m.start(), "end": m.end()})
    return out


def _filter_contained_entities(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Prefer longer spans; drop entities fully contained within another entity of the same label.
    by_label: dict[str, list[dict[str, Any]]] = {}
    for e in entities:
        by_label.setdefault(e["label"], []).append(e)

    out: list[dict[str, Any]] = []
    for label, ents in by_label.items():
        ents_sorted = sorted(ents, key=lambda x: ((x["end"] - x["start"]) * -1, x["start"]))
        kept: list[dict[str, Any]] = []
        for e in ents_sorted:
            contained = False
            for k in kept:
                if int(k["start"]) <= int(e["start"]) and int(e["end"]) <= int(k["end"]):
                    contained = True
                    break
            if not contained:
                kept.append(e)
        out.extend(sorted(kept, key=lambda x: (x["start"], x["end"])))
    return out


def _dedupe_entities(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    out = []
    for e in entities:
        key = (e["label"], e["start"], e["end"], e["text"])
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


def _spacy_org_entities(text: str, config: NerConfig) -> list[dict[str, Any]]:
    try:
        import spacy  # type: ignore

        try:
            nlp = spacy.load(config.spacy_model)
        except Exception:
            nlp = spacy.blank("en")
        doc = nlp(text)
        ents = []
        for e in doc.ents:
            if e.label_ == "ORG":
                ents.append({"label": "ORG", "text": e.text, "start": e.start_char, "end": e.end_char})
        if ents:
            return ents
    except Exception:
        pass

    # Fallback: lightweight ORG-ish heuristic (still useful when no spaCy model is installed).
    org_suffixes = r"(?:Ltd\\.?|Limited|LLP|Inc\\.?|Corporation|Company|Ministry|Department|Government|Commission|Authority|Board|Tribunal|Court)"
    org_re = re.compile(rf"\b(?:[A-Z][A-Za-z&\-]+\s+){{1,6}}{org_suffixes}\b")
    out: list[dict[str, Any]] = []
    for m in org_re.finditer(text):
        out.append({"label": "ORG", "text": m.group(0), "start": m.start(), "end": m.end()})

    # Common Indian org pattern: "... of India" (allowing lowercase connectors like "and")
    of_india = re.compile(
        r"\b(?:[A-Z][A-Za-z&\-]+(?:\s+(?:[A-Z][A-Za-z&\-]+|and|&)){0,8}\s+of\s+India)\b"
    )
    for m in of_india.finditer(text):
        out.append({"label": "ORG", "text": m.group(0), "start": m.start(), "end": m.end()})

    # Ministry of X
    ministry = re.compile(r"\bMinistry\s+of\s+[A-Z][A-Za-z&\-]+(?:\s+[A-Z][A-Za-z&\-]+){0,4}\b")
    for m in ministry.finditer(text):
        out.append({"label": "ORG", "text": m.group(0), "start": m.start(), "end": m.end()})

    department = re.compile(r"\bDepartment\s+of\s+[A-Z][A-Za-z&\-]+(?:\s+[A-Z][A-Za-z&\-]+){0,6}\b")
    for m in department.finditer(text):
        out.append({"label": "ORG", "text": m.group(0), "start": m.start(), "end": m.end()})

    registrar = re.compile(r"\bRegistrar\s+of\s+Companies,\s*[A-Z][A-Za-z&\-]+(?:\s+[A-Z][A-Za-z&\-]+){0,6}\b")
    for m in registrar.finditer(text):
        out.append({"label": "ORG", "text": m.group(0), "start": m.start(), "end": m.end()})

    central_board = re.compile(r"\bCentral\s+Board\s+of\s+[A-Z][A-Za-z&\-]+(?:\s+[A-Z][A-Za-z&\-]+){0,6}\b")
    for m in central_board.finditer(text):
        out.append({"label": "ORG", "text": m.group(0), "start": m.start(), "end": m.end()})

    return out


def extract_entities(text: str, config: NerConfig) -> list[dict[str, Any]]:
    entities: list[dict[str, Any]] = []
    entities += _regex_entities("SECTION_REF", _SECTION_REF, text)
    entities += _regex_entities("DATE", _DATE, text)
    entities += _regex_entities("MONEY", _MONEY, text)
    entities += _regex_entities("NOTIFICATION", _NOTIFICATION_CODE, text)
    entities += _regex_entities("ACT_NAME", _ACT_NAME, text)
    entities += _spacy_org_entities(text, config)
    entities = _filter_contained_entities(entities)
    return _dedupe_entities(entities)


def annotate_chunks(project_root: Path, config: NerConfig) -> dict[str, Any]:
    chunks_path = project_root / "chunks" / "chunks.jsonl"
    out_path = project_root / "ner" / "annotations.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("", encoding="utf-8")

    annotated = 0
    for row in read_jsonl(chunks_path):
        chunk_id = row["chunk_id"]
        text = row["text"]

        append_jsonl(out_path, {"chunk_id": chunk_id, "entities": extract_entities(text, config)})
        annotated += 1

    return {"chunks_annotated": annotated, "annotations_path": str(out_path)}
