# Bharat.Law — Mini Legal Data Pipeline

Implements the take-home pipeline described in `Bharat.Law AI Pipeline Project 2.pdf`:

* Crawl a small set of sources (static HTML + PDFs)
* Normalize to clean, structured text (`.md`)
* Chunk into ~400–800 token segments with overlap
* Run hybrid NER (regex + spaCy ORG when available)
* (Optional) evaluate against `data/gold_ner.jsonl`

> **Cross-platform note:** For guaranteed compatibility across all OS and environments, Docker usage is recommended. See the [Docker](#docker-bonus) section below.

## Quickstart

From this folder:

```
./run.sh
```

Outputs:

* `raw/` + `crawl_index.jsonl`
* `normalized/` + `normalized_index.jsonl`
* `chunks/chunks.jsonl`
* `ner/annotations.jsonl`
* `reports/metrics.html`

## Inputs

* Seed URLs: `data/seed_urls.json`
* Gold labels (optional, for evaluation): place `gold_ner.jsonl` at `data/gold_ner.jsonl`

## Run stages

This repo keeps a single entrypoint that runs all stages:

```
python -m scripts.run_pipeline --max-pages 15 --delay 1.5
```

Enable JS rendering fallback (Playwright) for JS-heavy / blocked pages:

```
python -m pip install playwright
playwright install chromium
python -m scripts.run_pipeline --enable-js
```

Re-run processing without re-crawling (useful when tuning normalization/chunking):

```
python -m scripts.run_pipeline --skip-crawl
```

Tune chunk sizing (spec target is 400–800):

```
python -m scripts.run_pipeline --min-tokens 400 --max-tokens 800 --overlap-tokens 80
```

If lots of normalized pages are still too short to form 400+ token chunks, you can drop them:

```
python -m scripts.run_pipeline --drop-small-docs
```

Or, if you want to keep short pages but still reach 400+ token chunks, you can merge multiple short documents together before chunking:

```
python -m scripts.run_pipeline --merge-small-docs
```

The run also writes an HTML metrics report (default):

```
python -m scripts.report_metrics --out reports/metrics.html
```

To evaluate (once you have `data/gold_ner.jsonl`):

```
python -m scripts.predict_gold_ner --gold data/gold_ner.jsonl --out ner/pred_gold_ner.jsonl
python -m scripts.evaluate_ner --gold data/gold_ner.jsonl --pred ner/pred_gold_ner.jsonl
```

The evaluation prints JSON with:

* `relaxed`: matches by `(label, entity text)` (recommended for quick iteration)
* `strict`: additionally requires matching `start/end` character offsets

## Design choices (high level)

* Robots + dedupe + depth-limited BFS crawl with exponential backoff.
* HTML normalization uses `readability-lxml` when possible; otherwise falls back to a direct HTML-to-text pass.
* Chunking splits on paragraph/heading boundaries and tracks `section_path` and `page_no` (for PDFs).
* NER uses:
  * regex: `SECTION_REF`, `DATE`, `MONEY`, `NOTIFICATION`, `ACT_NAME`
  * spaCy: `ORG` if a model is available; otherwise returns none (still produces a valid output file).
* Crawler uses a small denylist and URL-scoring heuristic to prioritize content-like links (articles/PDFs) over auth/navigation pages.
* JS rendering via Playwright is triggered automatically on HTTP 403/429 responses or when the returned HTML is suspiciously small (<2500 chars). Pass `--enable-js` to activate.

## Limitations

* Token estimation is approximate (word-based); swap to `tiktoken` for more accurate counts.
* `ACT_NAME` extraction is regex-heavy; a better approach would be a curated gazetteer + section-aware patterns.
* `spaCy` ORG extraction requires a model to be installed (`python -m spacy download en_core_web_sm`); if unavailable, ORG entities are skipped but output remains valid.

## Docker (bonus)

```
docker build -t bharatlaw-pipeline .
docker run --rm -it -v "$PWD:/app" bharatlaw-pipeline
```

## Unit tests (bonus)

```
python -m pip install -r requirements-dev.txt
pytest -q
```
