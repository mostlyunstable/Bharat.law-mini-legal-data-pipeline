from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _escape(s: str) -> str:
    return html.escape(s, quote=True)


def build_metrics(project_root: Path) -> dict[str, Any]:
    crawl_rows = _read_jsonl(project_root / "crawl_index.jsonl")
    norm_rows = _read_jsonl(project_root / "normalized_index.jsonl")
    chunk_rows = _read_jsonl(project_root / "chunks" / "chunks.jsonl")
    ner_rows = _read_jsonl(project_root / "ner" / "annotations.jsonl")

    status_counts = Counter(str(r.get("status")) for r in crawl_rows)
    pages_crawled = sum(1 for r in crawl_rows if r.get("status") == 200)

    char_counts = [int(r.get("char_count", 0) or 0) for r in norm_rows]
    tokens = [int(r.get("token_estimate", 0) or 0) for r in chunk_rows]
    tokens_sorted = sorted(tokens)
    def pct(p: float) -> float:
        if not tokens_sorted:
            return 0.0
        idx = int(round((p / 100.0) * (len(tokens_sorted) - 1)))
        return float(tokens_sorted[max(0, min(idx, len(tokens_sorted) - 1))])

    within_400_800 = sum(1 for t in tokens if 400 <= t <= 800)
    below_400 = sum(1 for t in tokens if 0 < t < 400)
    above_800 = sum(1 for t in tokens if t > 800)

    entity_counts: Counter[str] = Counter()
    entity_total = 0
    for r in ner_rows:
        for e in r.get("entities", []) or []:
            label = str(e.get("label", ""))
            if label:
                entity_counts[label] += 1
                entity_total += 1

    return {
        "crawl": {"rows": len(crawl_rows), "pages_crawled": pages_crawled, "status_counts": dict(status_counts)},
        "normalize": {
            "docs": len(norm_rows),
            "avg_chars": (sum(char_counts) / len(char_counts)) if char_counts else 0.0,
            "max_chars": max(char_counts) if char_counts else 0,
        },
        "chunk": {
            "chunks": len(chunk_rows),
            "avg_tokens": (sum(tokens) / len(tokens)) if tokens else 0.0,
            "max_tokens": max(tokens) if tokens else 0,
            "within_400_800": within_400_800,
            "below_400": below_400,
            "above_800": above_800,
            "p50_tokens": pct(50),
            "p90_tokens": pct(90),
        },
        "ner": {"chunks_annotated": len(ner_rows), "entities_total": entity_total, "by_label": dict(entity_counts)},
    }


def render_html(metrics: dict[str, Any]) -> str:
    crawl = metrics["crawl"]
    normalize = metrics["normalize"]
    chunk = metrics["chunk"]
    ner = metrics["ner"]

    def kv_table(d: dict[str, Any]) -> str:
        rows = []
        for k, v in d.items():
            rows.append(f"<tr><td>{_escape(str(k))}</td><td>{_escape(str(v))}</td></tr>")
        return "<table><tbody>" + "".join(rows) + "</tbody></table>"

    status_table = kv_table(crawl["status_counts"])
    label_table = kv_table(dict(sorted((ner["by_label"] or {}).items(), key=lambda x: (-x[1], x[0]))))

    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Bharat.Law Pipeline Metrics</title>
    <style>
      :root {{
        --bg: #0b1020;
        --card: #111a33;
        --text: #e6e8ef;
        --muted: #a9b0c3;
        --border: rgba(255,255,255,.12);
      }}
      body {{ margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; background: var(--bg); color: var(--text); }}
      .wrap {{ max-width: 1000px; margin: 28px auto; padding: 0 16px; }}
      h1 {{ font-size: 20px; margin: 0 0 14px; }}
      .grid {{ display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 12px; }}
      .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 14px; }}
      .kpi {{ font-size: 28px; font-weight: 700; margin: 4px 0 2px; }}
      .label {{ color: var(--muted); font-size: 12px; }}
      table {{ width: 100%; border-collapse: collapse; margin-top: 8px; }}
      td {{ border-top: 1px solid var(--border); padding: 8px 6px; font-size: 13px; vertical-align: top; }}
      td:first-child {{ color: var(--muted); width: 50%; }}
      @media (max-width: 720px) {{ .grid {{ grid-template-columns: 1fr; }} }}
    </style>
  </head>
  <body>
    <div class="wrap">
      <h1>Bharat.Law Pipeline Metrics</h1>
      <div class="grid">
        <div class="card">
          <div class="kpi">{crawl["pages_crawled"]}</div>
          <div class="label">Pages crawled (HTTP 200)</div>
          {status_table}
        </div>
        <div class="card">
          <div class="kpi">{normalize["docs"]}</div>
          <div class="label">Docs normalized</div>
          {kv_table({"avg_chars": round(normalize["avg_chars"], 1), "max_chars": normalize["max_chars"]})}
        </div>
        <div class="card">
          <div class="kpi">{chunk["chunks"]}</div>
          <div class="label">Chunks created</div>
          {kv_table({"avg_tokens": round(chunk["avg_tokens"], 1), "p50_tokens": chunk["p50_tokens"], "p90_tokens": chunk["p90_tokens"], "max_tokens": chunk["max_tokens"], "within_400_800": chunk["within_400_800"], "below_400": chunk["below_400"], "above_800": chunk["above_800"]})}
        </div>
        <div class="card">
          <div class="kpi">{ner["entities_total"]}</div>
          <div class="label">Entities extracted</div>
          {label_table}
        </div>
      </div>
      <div class="card" style="margin-top: 12px;">
        <div class="label">Files</div>
        <div style="font-size: 13px; margin-top: 6px; color: var(--muted);">
          crawl_index.jsonl, normalized_index.jsonl, chunks/chunks.jsonl, ner/annotations.jsonl
        </div>
      </div>
    </div>
  </body>
</html>
"""


def write_report(project_root: Path, out_path: Path) -> Path:
    metrics = build_metrics(project_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_html(metrics), encoding="utf-8")
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, default=Path("reports/metrics.html"))
    args = ap.parse_args()

    project_root = args.project_root.resolve()
    out_path = (project_root / args.out).resolve()
    write_report(project_root, out_path)
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
