from __future__ import annotations

import json
from pathlib import Path

from chunker.chunk import ChunkConfig, chunk_all
from utils import url_to_hash


def test_chunking_produces_multiple_chunks(tmp_path: Path) -> None:
    project_root = tmp_path
    (project_root / "normalized").mkdir(parents=True, exist_ok=True)

    url = "https://example.com/doc"
    url_hash = url_to_hash(url)
    doc_path = project_root / "normalized" / f"{url_hash}.md"
    # Enough text for multiple chunks.
    doc_path.write_text("# Title\n\n" + ("word " * 2000) + "\n", encoding="utf-8")

    (project_root / "normalized_index.jsonl").write_text(
        json.dumps(
            {
                "url": url,
                "url_hash": url_hash,
                "source_type": "html",
                "title": "Title",
                "detected_language": "en",
                "char_count": doc_path.stat().st_size,
                "path_to_text": f"normalized/{url_hash}.md",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = chunk_all(project_root, ChunkConfig(min_tokens=120, max_tokens=240, overlap_tokens=30))
    assert result["chunks_created"] >= 2

    chunks_path = project_root / "chunks" / "chunks.jsonl"
    lines = [json.loads(l) for l in chunks_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert all(l["chunk_id"].startswith(f"{url_hash}:") for l in lines)
    assert all(isinstance(l["token_estimate"], int) and l["token_estimate"] > 0 for l in lines)

