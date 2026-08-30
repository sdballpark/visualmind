# VisualMind

Local-first semantic search and duplicate detection over a personal photo
collection. Runs entirely on one machine: no cloud APIs, no photos leaving
the filesystem.

Built against a 441-image family photo corpus spanning 2002-2026 - mostly
consumer digicam JPEGs and scans, which turns out to matter (see the
evaluations).

## What it does

Ask for photos in plain language and get back only the ones that exist:

    uv run python scripts/search_gallery.py "someone holding a baby"
    -> 18 results, derived from caption ground truth

    uv run python scripts/search_gallery.py "cat"
    -> 2 results, both depictions - the corpus has no live cats

Find redundant copies without touching anything:

    uv run python scripts/find_duplicates.py
    -> 6 near-duplicate groups, 15 same-scene groups, nothing deleted

Search output is a self-contained HTML gallery with thumbnails, captions,
and the rank each result held in both underlying indexes.

## How it works

Four models, each doing what it is best at:

| Role             | Model                            | Job                        |
|------------------|----------------------------------|----------------------------|
| Image embedding  | google/siglip2-base-patch16-224  | text-to-image retrieval    |
| Captioning       | Qwen/Qwen3-VL-4B-Instruct        | describe every photo       |
| Text embedding   | BAAI/bge-large-en-v1.5           | search over captions       |
| Visual embedding | facebook/dinov2-base             | image-to-image similarity  |

Text search fuses the image and caption rankings with Reciprocal Rank
Fusion, since their score scales are not comparable. Duplicate detection
uses DINOv2 alone - it has no text tower and answers only "are these the
same photograph".

### The result count is derived, not fixed

Most retrieval demos return a fixed top-k. That padding hides both failure
and success: a query with four true matches returns eight wrong ones, and a
query with twenty returns only twelve.

VisualMind decides how many results to return from three signals:

1. Full caption term match - every query term appears in the caption
2. Partial term match - at least two terms appear
3. Score gradient - where the similarity curve flattens

When none is confident, it says so rather than returning a number that
looks like an answer.

### Duplicate detection runs in three tiers

    EXACT    identical SHA-256
    NEAR     perceptual hash within Hamming distance 6
    SIMILAR  DINOv2 cosine >= 0.92

Each tier catches what the previous one structurally cannot. A re-encoded
copy changes every byte, so hashing misses it; perceptual hashing does not.
The script reports and proposes a keeper. It never moves or deletes.

## Results

Search, over nine queries with manual relevance labelling:

| Query type     | Queries | Images returned | Precision |
|----------------|---------|-----------------|-----------|
| Concrete nouns | 8       | 90              | 100%      |
| Relational     | 1       | 18              | 83%       |

Against a fixed top-12 baseline, "dog" returned 10 correct of 12 and missed
10 dogs that exist in the corpus. The derived count returns all 20.

Duplicates, over the full corpus: 0 exact, 6 near-duplicates, 15 same-scene
groups of which roughly 13 are useful.

Full methodology is in
[evals/retrieval-evaluation.md](evals/retrieval-evaluation.md) and
[evals/duplicate-detection.md](evals/duplicate-detection.md), including
three hypotheses that were tested and rejected:

- A larger, higher-resolution embedding model (SigLIP2 SO400M-384)
  performed measurably worse on this corpus than the base 224px model.
- Compound queries do not underperform bare nouns; that variation was
  sample noise.
- Semantic reranking does not fix relational queries. The text encoder
  ranks subject/object errors mid-pack, not last.

One finding shaped the design more than the rest: absolute similarity
scores cannot distinguish "no matches" from "weak matches". A concept with
zero instances in the corpus scored within 12% of one with twenty.

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

    scripts/inspect_archive.py           validate the source archive
    scripts/reconcile_manifest.py        match files to provenance records
    scripts/build_catalog.py             normalise metadata and EXIF
    scripts/build_captions.py            VLM captions (resumable)
    scripts/build_embeddings.py          SigLIP2 image index
    scripts/build_caption_embeddings.py  BGE caption index
    scripts/build_visual_embeddings.py   DINOv2 visual index
    scripts/find_duplicates.py           three-tier duplicate report
    scripts/search_hybrid.py             console search
    scripts/search_gallery.py            HTML gallery

Retrieval logic is shared in `src/visualmind/retrieval.py` so the two
search entry points cannot drift apart.

Each stage takes its source path as an argument; nothing is hardcoded. The
catalog matches files to provenance records by SHA-256 as well as by path,
so the archive can be relocated or read across platforms without losing
metadata.

## Data handling

Photos, captions, embeddings, and indexes are all gitignored. Caption text
and lookup CSVs describe private images and are treated as sensitive. The
source archive is read-only and never modified. No script deletes or moves
a source file.

## Status

Working: ingestion, captioning, three indexes, hybrid search, derived
result counts, HTML gallery, duplicate detection.

Not yet built: face clustering, event grouping, dynamic taxonomy, web UI.

## Licence

MIT. Note that model licences differ - `configs/models.yaml` records each
one. All models currently in use are Apache 2.0 or MIT.
