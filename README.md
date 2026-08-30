# VisualMind

Local-first semantic search, duplicate detection, and face clustering over
a personal photo collection. Runs entirely on one machine: no cloud APIs,
no photos leaving the filesystem.

Built against a 441-image family photo corpus spanning 2002-2026 - mostly
consumer digicam JPEGs and scans, which turns out to matter (see the
evaluations).

## What it does

Ask for photos in plain language and get back only the ones that exist:

    uv run python scripts/search_gallery.py "someone holding a baby"
    -> 18 results, derived from caption ground truth

    uv run python scripts/search_gallery.py "cat"
    -> 2 results, both depictions - the corpus has no live cats

Filter by who is in the photo:

    uv run python scripts/search_gallery.py "dog" --person lisa
    -> 3 results, from the 73 images containing Lisa

    uv run python scripts/search_gallery.py "" --person casey --person manvi
    -> 16 results, every photo containing both

Find redundant copies without touching anything:

    uv run python scripts/find_duplicates.py
    -> 6 near-duplicate groups, 15 same-scene groups, nothing deleted

Check whether anything is stale:

    uv run python scripts/status.py
    -> per-artifact coverage against the catalog, exit 1 if a rebuild is due

Search output is a self-contained HTML gallery with thumbnails, captions,
and the rank each result held in every underlying index.

## How it works

Five models, each doing what it is best at:

| Role             | Model                            | Job                        |
|------------------|----------------------------------|----------------------------|
| Image embedding  | google/siglip2-base-patch16-224  | text-to-image retrieval    |
| Captioning       | Qwen/Qwen3-VL-4B-Instruct        | describe every photo       |
| Text embedding   | BAAI/bge-large-en-v1.5           | search over captions       |
| Visual embedding | facebook/dinov2-base             | image-to-image similarity  |
| Face analysis    | InsightFace buffalo_l            | detection and recognition  |

Text search fuses the image and caption rankings with Reciprocal Rank
Fusion, since their score scales are not comparable. Duplicate detection
uses DINOv2 alone. Face work is separate again - detection, then
clustering, then names.

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

### People are a filter, not a ranking signal

Faces are detected, clustered into anonymous identities with DBSCAN, and
named once by hand. Names then constrain which images are searched at all.

Names are passed explicitly with `--person`, never parsed out of the query
text. No model has seen the label file, so a name inside a sentence would
have to be guessed at - and guessing which words are names fails in ways
that are hard to explain. `--list-people` shows who is known.

Labels anchor to face IDs rather than cluster IDs, so re-clustering with
different parameters does not scramble who is who.

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

Faces: 1,325 detected across 365 of 441 images, clustered into 52 groups at
DBSCAN eps 0.45, labelled as 44 distinct people covering 81% of faces.
Three clusters merged into one person where the same individual appeared at
different ages.

Duplicates: 0 exact, 6 near-duplicates, 15 same-scene groups of which
roughly 13 are useful.

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

`check_models.py` reports which models are cached, resolves each pinned
revision against the Hub, and shows the licence per model. Weights live in
the shared Hugging Face cache (`$HF_HOME`) or, for InsightFace, in
`~/.insightface`. None of them are in this repository - see
`configs/models.yaml` for the registry.

Install the pre-commit hook before working on this:

    git config core.hooksPath .githooks

## Pipeline

    scripts/inspect_archive.py           validate the source archive
    scripts/reconcile_manifest.py        match files to provenance records
    scripts/build_catalog.py             normalise metadata and EXIF
    scripts/build_captions.py            VLM captions (resumable)
    scripts/build_embeddings.py          SigLIP2 image index
    scripts/build_caption_embeddings.py  BGE caption index
    scripts/build_visual_embeddings.py   DINOv2 visual index
    scripts/build_faces.py               face detection (resumable)
    scripts/cluster_faces.py             DBSCAN clustering + contact sheet
    scripts/label_faces.py               assign names to clusters
    scripts/find_duplicates.py           three-tier duplicate report
    scripts/search_hybrid.py             console search
    scripts/search_gallery.py            HTML gallery
    scripts/status.py                    staleness check

Shared logic lives in `src/visualmind/` - `retrieval.py` for search,
`people.py` for name resolution - so the two search entry points cannot
drift apart.

Each stage takes its source path as an argument; nothing is hardcoded. The
catalog matches files to provenance records by SHA-256 as well as by path,
so the archive can be relocated or read across platforms without losing
metadata.

## Data handling

Photos, captions, embeddings, indexes, face data, and person labels are all
gitignored, and a pre-commit hook refuses to stage them regardless. Face
embeddings are biometric identifiers for identifiable people; caption text
and lookup CSVs describe private photographs.

The hook is local and does not travel with a clone - install it as above.

The source archive is read-only. No script moves or deletes a source file.

## Status

Working: ingestion, captioning, three content indexes, hybrid search,
derived result counts, HTML gallery, duplicate detection, face detection
and clustering, person-filtered search, staleness checking.

Not yet built: event grouping, dynamic taxonomy, web UI, agentic
organisation.

## Licences

The code is MIT.

Model licences differ, and one is restrictive. InsightFace's pretrained
models are licensed for non-commercial research use only; commercial use
requires a separate licence from InsightFace. Everything else in use is
Apache 2.0 or MIT. `configs/models.yaml` records the licence for each
model, and `check_models.py` prints them.
