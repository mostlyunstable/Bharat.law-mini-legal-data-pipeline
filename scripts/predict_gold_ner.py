from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ner.extract import NerConfig, extract_entities


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Run pipeline NER on gold_ner.jsonl texts to produce predictions.")
    ap.add_argument("--gold", type=Path, default=Path("data/gold_ner.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("ner/pred_gold_ner.jsonl"))
    ap.add_argument("--spacy-model", type=str, default="en_core_web_sm")
    args = ap.parse_args()

    if not args.gold.exists():
        print(f"Gold file not found: {args.gold}")
        return 2

    cfg = NerConfig(spacy_model=args.spacy_model)
    rows = _read_jsonl(args.gold)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with args.out.open("w", encoding="utf-8") as f:
        for r in rows:
            text = r.get("text") or ""
            ents = extract_entities(text, cfg)
            f.write(json.dumps({"text": text, "entities": ents}, ensure_ascii=False) + "\n")

    print(str(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

