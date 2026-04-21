from __future__ import annotations

import json
from pathlib import Path

from ner.extract import NerConfig, annotate_chunks


def test_ner_regex_finds_core_labels(tmp_path: Path) -> None:
    project_root = tmp_path
    (project_root / "chunks").mkdir(parents=True, exist_ok=True)

    chunk = {
        "chunk_id": "abc:0001",
        "url": "https://example.com",
        "title": "t",
        "section_path": ["H1"],
        "page_no": None,
        "char_start": 0,
        "char_end": 10,
        "token_estimate": 10,
        "text": "Under section 2(1)(a) of the Companies Act, 1956 on 12 March 2024, the Ministry of Finance paid ₹500.",
    }
    (project_root / "chunks" / "chunks.jsonl").write_text(json.dumps(chunk) + "\n", encoding="utf-8")

    annotate_chunks(project_root, NerConfig(spacy_model="en_core_web_sm"))

    ann_path = project_root / "ner" / "annotations.jsonl"
    rows = [json.loads(l) for l in ann_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert rows and rows[0]["chunk_id"] == "abc:0001"
    labels = {e["label"] for e in rows[0]["entities"]}
    assert {"SECTION_REF", "ACT_NAME", "DATE", "MONEY"}.issubset(labels)

