# Bharat.Law — Mini Legal Data Pipeline

Implements the take-home pipeline described in `Bharat.Law AI Pipeline Project 2.pdf`:

* Crawl a small set of sources (static HTML + PDFs)
* Normalize to clean, structured text (`.md`)
* Chunk into ~400–800 token segments with overlap
* Run hybrid NER (regex + spaCy ORG when available)
* (Optional) evaluate against `data/gold_ner.jsonl`

> **Cross-platform note:** For guaranteed compatibility across all OS and environments, Docker usage is recommended. See the [Docker](#docker-bonus) section below.

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

For JS rendering fallback (optional — only needed for blocked/JS-heavy pages):

```bash
pip install playwright
playwright install chromium
```

## Quickstart

```bash
./run.sh
```

Outputs:

* `raw/` + `crawl_index.jsonl`
* `normalized/` + `normalized_index.jsonl`
* `chunks/chunks.jsonl`
* `ner/annotations.jsonl`
* `reports/metrics.html`

## Folder structure

```
/project
  /crawler        # BFS crawler with robots.txt, backoff, Playwright fallback
  /parsers        # HTML → markdown, PDF → markdown (PyMuPDF)
  /chunker        # Semantic chunker with overlap and section tracking
  /ner            # Hybrid NER: regex + spaCy
  /data           # seed_urls.json, gold_ner.jsonl (optional)
  /raw            # crawled HTML/PDF files (generated)
  /normalized     # clean .md files (generated)
  /chunks         # chunks.jsonl (generated)
  /scripts        # run_pipeline, evaluate_ner, predict_gold_ner, report_metrics
  /tests          # unit tests
  /reports        # metrics.html (generated)
  README.md
  requirements.txt
  requirements-dev.txt
  run.sh
  Dockerfile
  utils.py
```

## Inputs

* Seed URLs: `data/seed_urls.json`
* Gold labels (optional, for evaluation): place `gold_ner.jsonl` at `data/gold_ner.jsonl`

## Run stages

Single entrypoint that runs all stages:

```bash
python -m scripts.run_pipeline --max-pages 15 --delay 1.5
```

Enable JS rendering fallback for JS-heavy / blocked pages:

```bash
python -m scripts.run_pipeline --enable-js
```

Re-run without re-crawling (useful when tuning normalization/chunking):

```bash
python -m scripts.run_pipeline --skip-crawl
```

Tune chunk sizing (spec target is 400–800 tokens):

```bash
python -m scripts.run_pipeline --min-tokens 400 --max-tokens 800 --overlap-tokens 80
```

Drop documents too short to form a 400+ token chunk:

```bash
python -m scripts.run_pipeline --drop-small-docs
```

Merge short documents together before chunking:

```bash
python -m scripts.run_pipeline --merge-small-docs
```

Generate HTML metrics report:

```bash
python -m scripts.report_metrics --out reports/metrics.html
```

## Evaluation

Once you have `data/gold_ner.jsonl`:

```bash
python -m scripts.predict_gold_ner --gold data/gold_ner.jsonl --out ner/pred_gold_ner.jsonl
python -m scripts.evaluate_ner --gold data/gold_ner.jsonl --pred ner/pred_gold_ner.jsonl
```

Prints JSON with:

* `relaxed`: matches by `(label, entity text)` — recommended for quick iteration
* `strict`: additionally requires matching `start/end` character offsets

## Design choices

* Robots + dedupe + depth-limited BFS crawl with exponential backoff.
* Priority heap scores PDFs and article URLs higher to use crawl budget efficiently.
* HTML normalization uses `readability-lxml` when possible; falls back to a direct HTML-to-text pass if body is too short (<200 chars).
* PDF extraction uses PyMuPDF (`fitz`) with `sort=True` to preserve reading order.
* Chunking splits on paragraph/heading boundaries and tracks `section_path` and `page_no` (for PDFs). Overlapping tail words from previous chunk are prepended for continuity.
* NER uses:
  * regex: `SECTION_REF`, `DATE`, `MONEY`, `NOTIFICATION`, `ACT_NAME`
  * spaCy: `ORG` if `en_core_web_sm` is available; otherwise skipped (output file still produced).
* JS rendering via Playwright triggers automatically on HTTP 403/429 or when returned HTML is suspiciously small (<2500 chars). Pass `--enable-js` to activate.

## Limitations

* Token estimation is word-based (~1 token = 0.75 words); swap to `tiktoken` for more accurate counts.
* `ACT_NAME` extraction is regex-based; a curated gazetteer of known Indian Acts would improve precision.
* `spaCy` ORG extraction requires `en_core_web_sm` (`python -m spacy download en_core_web_sm`); if unavailable, ORG entities are skipped but all other output remains valid.

## Docker (bonus)

```bash
docker build -t bharatlaw-pipeline .
docker run --rm -it -v "$PWD:/app" bharatlaw-pipeline
```

## Unit tests (bonus)

```bash
pip install -r requirements-dev.txt
pytest -q
```
