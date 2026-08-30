# VisualMind

Local-first semantic search over a personal photo collection. Runs entirely
on one machine: no cloud APIs, no photos leaving the filesystem.

Built against a 441-image family photo corpus spanning 2002-2026 - mostly
consumer digicam JPEGs and scans, which turns out to matter (see the
evaluation).

## What it does

Ask for photos in plain language and get back only the ones that exist:

    uv run python scripts/search_gallery.py "someone holding a baby"
    -> 18 results, derived from caption ground truth

    uv run python scripts/search_gallery.py "cat"
    -> 2 results, both depictions - the corpus has no live cats

Output is a self-contained HTML gallery with thumbnails, captions, and the
rank each result held in both underlying indexes.

## How it works

Three models, each doing what it is best at:

| Role            | Model                            | Job                          |
|-----------------|----------------------------------|------------------------------|
| Image embedding | google/siglip2-base-patch16-224  | text-to-image retrieval      |
| Captioning      | Qwen/Qwen3-VL-4B-Instruct        | describe every photo         |
| Text embedding  | BAAI/bge-large-en-v1.5           | semantic search over captions|

Image and caption rankings are fused with Reciprocal Rank Fusion, since
their score scales are not comparable.

### The result count is derived, not fixed

Most retrieval demos return a fixed top-k. That padding hides both failure
and success: a query with four true matches returns eight wrong ones, and a
query with twenty returns only twelve.

VisualMind decides how many results to return from three signals:

1. Full caption term match - every query term appears in the caption
2. Partial term match - at least two terms appear
3. Score gradient - where the similarity curve flattens

When none of these is confident, it says so rather than returning a number
that looks like an answer.

## Results

Measured over nine queries with manual relevance labelling:

| Query type     | Queries | Images returned | Precision |
|----------------|---------|-----------------|-----------|
| Concrete nouns | 8       | 90              | 100%      |
| Relational     | 1       | 18              | 83%       |

Against a fixed top-12 baseline, "dog" returned 10 correct of 12 and missed
10 dogs that exist in the corpus. The derived count returns all 20.

Full methodology, including two hypotheses that were tested and rejected, is
in [evals/retrieval-evaluation.md](evals/retrieval-evaluation.md). Notable
findings:

- A larger, higher-resolution embedding model (SigLIP2 SO400M-384)
  performed measurably worse on this corpus than the base 224px model.
- Absolute similarity scores cannot distinguish "no matches" from "weak
  matches" - a concept with zero instances scored within 12% of one with
  twenty.

## Setup

Requires CUDA. Developed on an RTX 4090 Laptop (16 GB); peak usage is
9.14 GB during captioning.

    uv sync
    uv run python scripts/check_models.py

`check_models.py` reports which models are cached and resolves each pinned
revision against the Hub. Weights live in the shared Hugging Face cache
(`$HF_HOME`), never in this repository - see `configs/models.yaml` for the
registry, including licence per model.

## Pipeline

    scripts/inspect_archive.py      validate the source archive
    scripts/reconcile_manifest.py   match files against provenance records
    scripts/build_catalog.py        normalise metadata and EXIF
    scripts/build_captions.py       VLM captions (resumable)
    scripts/build_embeddings.py     SigLIP2 image index
    scripts/build_caption_embeddings.py  BGE caption index
    scripts/search_hybrid.py        console search
    scripts/search_gallery.py       HTML gallery

Retrieval logic is shared in `src/visualmind/retrieval.py` so the two search
entry points cannot drift apart.

Each stage takes its source path as an argument; nothing is hardcoded. The
catalog matches files to provenance records by SHA-256 as well as by path,
so the archive can be relocated or read across platforms without losing
metadata.

## Data handling

Photos, captions, embeddings, and indexes are all gitignored. Caption text
and lookup CSVs describe private images and are treated as sensitive. The
source archive is read-only and never modified.

## Status

Working: ingestion, captioning, both indexes, hybrid search, derived result
counts, HTML gallery.

Not yet built: near-duplicate detection, face clustering, event grouping,
dynamic taxonomy, web UI.

## Licence

MIT. Note that model licences differ - `configs/models.yaml` records each
one. All models currently in use are Apache 2.0 or MIT.
