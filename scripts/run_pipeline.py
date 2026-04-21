from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from chunker.chunk import ChunkConfig, chunk_all
from crawler.crawl import CrawlConfig, crawl
from ner.extract import NerConfig, annotate_chunks
from parsers.normalize import NormalizeConfig, normalize_all
from utils import Metrics


def _load_seed_urls(path: Path) -> dict[str, str]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Mini legal data pipeline (crawl -> normalize -> chunk -> NER)")
    ap.add_argument("--project-root", type=Path, default=Path("."))
    ap.add_argument("--seed-urls", type=Path, default=Path("data/seed_urls.json"))
    ap.add_argument("--max-pages", type=int, default=25)
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--delay", type=float, default=1.0)
    ap.add_argument("--enable-js", action="store_true", help="Use Playwright fallback for JS/blocked pages (optional).")
    ap.add_argument("--skip-crawl", action="store_true", help="Skip crawling and reuse existing raw/crawl_index.jsonl.")
    ap.add_argument("--min-tokens", type=int, default=400)
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--overlap-tokens", type=int, default=80)
    ap.add_argument(
        "--merge-small-docs",
        action="store_true",
        help="Merge multiple short docs together before chunking to reach min-tokens (adds source_urls metadata).",
    )
    ap.add_argument(
        "--drop-small-docs",
        action="store_true",
        help="Drop documents that cannot form a >= min-tokens chunk (helps hit avg 400–800).",
    )
    ap.add_argument("--summary-json", type=Path, default=None)
    ap.add_argument("--html-report", type=Path, default=Path("reports/metrics.html"))
    args = ap.parse_args()

    project_root: Path = args.project_root.resolve()
    seed_urls = _load_seed_urls((project_root / args.seed_urls).resolve())

    crawl_cfg = CrawlConfig(
        max_pages=args.max_pages,
        max_depth=args.max_depth,
        politeness_delay_s=args.delay,
        enable_js=bool(args.enable_js),
    )
    norm_cfg = NormalizeConfig()
    chunk_cfg = ChunkConfig(
        min_tokens=int(args.min_tokens),
        max_tokens=int(args.max_tokens),
        overlap_tokens=int(args.overlap_tokens),
        drop_small_docs=bool(args.drop_small_docs),
        merge_small_docs=bool(args.merge_small_docs),
    )
    ner_cfg = NerConfig()

    if args.skip_crawl:
        crawl_result = {"pages_crawled": 0, "crawl_index": str(project_root / "crawl_index.jsonl")}
    else:
        crawl_result = crawl(seed_urls, project_root, crawl_cfg)
    norm_result = normalize_all(project_root, norm_cfg)
    chunk_result = chunk_all(project_root, chunk_cfg)
    ner_result = annotate_chunks(project_root, ner_cfg)

    # Optional evaluation (only if user provides the file).
    gold_path = project_root / "data" / "gold_ner.jsonl"
    f1: float | None = None
    if gold_path.exists():
        from ner.extract import extract_entities
        from scripts.evaluate_ner import load_gold, score

        gold_rows = load_gold(gold_path)
        pred_rows = [{"text": (r.get("text") or ""), "entities": extract_entities((r.get("text") or ""), ner_cfg)} for r in gold_rows]
        f1 = score(gold_rows, pred_rows)["relaxed"]["micro"]["f1"]

    metrics = Metrics(
        pages_crawled=int(crawl_result["pages_crawled"]),
        chunks_created=int(chunk_result["chunks_created"]),
        avg_tokens_per_chunk=float(chunk_result["avg_tokens_per_chunk"]),
        ner_f1=f1,
    )

    print(f"Pages crawled: {metrics.pages_crawled}")
    print(f"Chunks created: {metrics.chunks_created}")
    print(f"Avg tokens/chunk: {metrics.avg_tokens_per_chunk:.1f}")
    if metrics.ner_f1 is not None:
        print(f"NER F1-score: {metrics.ner_f1:.2f}")
    else:
        print("NER F1-score: (skipped; place gold file at data/gold_ner.jsonl)")

    if args.summary_json:
        args.summary_json.write_text(json.dumps(metrics.__dict__, indent=2), encoding="utf-8")

    if args.html_report:
        from scripts.report_metrics import write_report

        report_path = write_report(project_root, project_root / args.html_report)
        print(f"HTML report: {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
