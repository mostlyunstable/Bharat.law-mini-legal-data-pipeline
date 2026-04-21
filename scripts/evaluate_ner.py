from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _norm_text(s: str) -> str:
    return re.sub(r"\\s+", " ", s.strip()).lower()


def load_gold(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_pred(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def score(gold_rows: list[dict[str, Any]], pred_rows: list[dict[str, Any]]) -> dict[str, Any]:
    # Prefer chunk_id alignment when available; otherwise fall back to row order.
    gold_has_ids = all(isinstance(r.get("chunk_id"), str) for r in gold_rows)
    pred_has_ids = all(isinstance(r.get("chunk_id"), str) for r in pred_rows)
    if gold_has_ids and pred_has_ids:
        gold_by_key = {r["chunk_id"]: r.get("entities", []) for r in gold_rows}
        pred_by_key = {r["chunk_id"]: r.get("entities", []) for r in pred_rows}
        keys = sorted(set(gold_by_key) | set(pred_by_key))
        aligned = [(k, gold_by_key.get(k, []), pred_by_key.get(k, [])) for k in keys]
    else:
        gold_has_text = all(isinstance(r.get("text"), str) for r in gold_rows)
        pred_has_text = all(isinstance(r.get("text"), str) for r in pred_rows)
        if gold_has_text and pred_has_text:
            gold_by_text = {_norm_text(r["text"]): r.get("entities", []) for r in gold_rows}
            pred_by_text = {_norm_text(r["text"]): r.get("entities", []) for r in pred_rows}
            keys = sorted(set(gold_by_text) | set(pred_by_text))
            aligned = [(k, gold_by_text.get(k, []), pred_by_text.get(k, [])) for k in keys]
        else:
            n = max(len(gold_rows), len(pred_rows))
            aligned = []
            for i in range(n):
                g = gold_rows[i].get("entities", []) if i < len(gold_rows) else []
                p = pred_rows[i].get("entities", []) if i < len(pred_rows) else []
                aligned.append((str(i), g, p))

    per_label_relaxed = defaultdict(lambda: Counter({"tp": 0, "fp": 0, "fn": 0}))
    per_label_strict = defaultdict(lambda: Counter({"tp": 0, "fp": 0, "fn": 0}))

    for key, gold_ents, pred_ents in aligned:
        g_relaxed = {(e["label"], _norm_text(e["text"])) for e in gold_ents}
        p_relaxed = {(e["label"], _norm_text(e["text"])) for e in pred_ents}

        g_strict = {(e["label"], int(e.get("start", -1)), int(e.get("end", -1)), _norm_text(e["text"])) for e in gold_ents}
        p_strict = {(e["label"], int(e.get("start", -1)), int(e.get("end", -1)), _norm_text(e["text"])) for e in pred_ents}

        for label, text in p_relaxed - g_relaxed:
            per_label_relaxed[label]["fp"] += 1
        for label, text in g_relaxed - p_relaxed:
            per_label_relaxed[label]["fn"] += 1
        for label, text in p_relaxed & g_relaxed:
            per_label_relaxed[label]["tp"] += 1

        for label, start, end, text in p_strict - g_strict:
            per_label_strict[label]["fp"] += 1
        for label, start, end, text in g_strict - p_strict:
            per_label_strict[label]["fn"] += 1
        for label, start, end, text in p_strict & g_strict:
            per_label_strict[label]["tp"] += 1

    def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        return prec, rec, f1

    def summarize(per_label: dict[str, Counter]) -> dict[str, Any]:
        tp_total = sum(c["tp"] for c in per_label.values())
        fp_total = sum(c["fp"] for c in per_label.values())
        fn_total = sum(c["fn"] for c in per_label.values())
        micro = prf(tp_total, fp_total, fn_total)

        by_label = {}
        for label, c in sorted(per_label.items()):
            by_label[label] = dict(zip(["precision", "recall", "f1"], prf(c["tp"], c["fp"], c["fn"])))
            by_label[label].update({"tp": c["tp"], "fp": c["fp"], "fn": c["fn"]})

        return {
            "micro": {
                "precision": micro[0],
                "recall": micro[1],
                "f1": micro[2],
                "tp": tp_total,
                "fp": fp_total,
                "fn": fn_total,
            },
            "by_label": by_label,
        }

    return {
        "relaxed": summarize(per_label_relaxed),
        "strict": summarize(per_label_strict),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=Path, default=Path("data/gold_ner.jsonl"))
    ap.add_argument("--pred", type=Path, default=Path("ner/annotations.jsonl"))
    args = ap.parse_args()

    if not args.gold.exists():
        print(f"Gold file not found: {args.gold}")
        return 2
    if not args.pred.exists():
        print(f"Pred file not found: {args.pred}")
        return 2

    result = score(load_gold(args.gold), load_pred(args.pred))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
