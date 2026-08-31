"""Shared retrieval logic for VisualMind search entry points.

Both scripts/search_hybrid.py (console) and scripts/search_gallery.py
(HTML) import from here so their behaviour cannot drift apart.

Term matching decides which images are candidates; the caption semantic
score decides their order. Term matching has no notion of subject and
object - "holding" and "baby" both appear in a caption describing a baby
holding a ladle - so it bounds the candidate set rather than ranking it.

`mode` selects the score that orders a matched set: caption score under
"hybrid" and "caption", image score under "image". Hybrid deliberately
orders matched sets by caption score alone - RRF order was measured
against it and scored no better. Outside a matched set, where results
come from the score gradient, "hybrid" fuses both rankings with RRF.

People and events are hard pre-filters applied before any scoring, and
they compose: naming both narrows to their intersection. Several people
must all be present; several events are a union, since an image belongs
to exactly one event.

Names and event references come from explicit arguments, never parsed
out of the query text. No model has seen those label files, and guessing
which words are names fails in ways that are hard to explain.

A filter with no text query returns the whole filtered pool in catalog
order, rather than an order manufactured from an empty embedding.

The score in each results tuple carries no fixed meaning: a matched set
is ordered by the score `mode` selected, everything else by the fused
RRF sum. `score_kind` in the outcome names which of those a caller is
holding, since the scales are an order of magnitude apart and nothing
in the value itself distinguishes them.

An embedding matrix and its lookup are matched by row position and
nothing else, so both are verified against a manifest fingerprint before
a query runs. See verify_index.

Every ordering breaks ties on source path. Equal scores are common -
RRF sums collide whenever two images hold each other's ranks in the two
modalities - and the sets and dicts they arrive in do not iterate in a
stable order across processes, so an untied sort ranks the same query
differently between runs.

Trimming the tail of a matched set, by whichever score ordered it, is
available behind the `trim` flag but is off by default. See
evals/retrieval-evaluation.md.
"""
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from visualmind import events as events_module
from visualmind import people

MODEL_CONFIG = Path("configs/models.yaml")
INDEX_DIR = Path("indexes")

IMAGE_EMB = INDEX_DIR / "siglip2_image_embeddings.npy"
IMAGE_LOOKUP = INDEX_DIR / "siglip2_lookup.csv"
IMAGE_MANIFEST = INDEX_DIR / "siglip2_index.json"
CAPTION_EMB = INDEX_DIR / "caption_embeddings.npy"
CAPTION_LOOKUP = INDEX_DIR / "caption_lookup.csv"
CAPTION_MANIFEST = INDEX_DIR / "caption_index.json"

REBUILD = {
    "image": "scripts/build_embeddings.py",
    "caption": "scripts/build_caption_embeddings.py",
}

RRF_K = 60
MIN_PARTIAL_TERMS = 2
GRADIENT_CEILING = 40
GRADIENT_FLOOR = 0.35
SEMANTIC_DROP = 0.80

# What scale the score in each results tuple is on. The branch that
# produced a result decides this, and the three scales are not
# comparable: RRF sums sit near 0.03, SigLIP cosines near 0.10, BGE
# cosines near 0.7. Named for the modality rather than the model, so
# swapping an encoder in configs/models.yaml cannot turn a stored
# identifier into a lie.
SCORE_CAPTION = "caption_cosine"
SCORE_IMAGE = "image_cosine"
SCORE_FUSED = "rrf_sum"
SCORE_NONE = "none"

STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "with", "and", "or",
    "someone", "somebody", "something", "anyone", "people", "person",
    "photo", "photos", "picture", "pictures", "image", "images",
    "is", "are", "was", "were", "being", "be",
}


def load_config(role):
    config = yaml.safe_load(MODEL_CONFIG.read_text(encoding="utf-8"))
    entry = config["models"][role]
    return entry["repo_id"], entry["revision"]


def read_lookup(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class IndexMismatch(RuntimeError):
    """An embedding matrix and its lookup no longer correspond."""


def lookup_fingerprint(paths):
    """Hash an ordered sequence of source paths.

    Row order is the only thing binding a lookup CSV to its embedding
    matrix, since scores are matched to rows by position. A rebuild that
    reorders the rows without changing their number is invisible to a
    count check and silently attaches every score to the wrong image, so
    the order itself is what gets fingerprinted.

    Both the builders and this module call it, so a drifting second
    implementation cannot raise false mismatches.
    """
    digest = hashlib.sha256()

    for path in paths:
        digest.update(path.encode("utf-8"))
        digest.update(b"\n")

    return digest.hexdigest()


def verify_index(matrix, rows, manifest_path, label):
    """Refuse an embedding matrix that no longer matches its lookup."""
    rebuild = "Rebuild with " + REBUILD[label] + "."

    if matrix.shape[0] != len(rows):
        raise IndexMismatch(
            label + " index: " + str(matrix.shape[0]) + " embedding rows "
            "against " + str(len(rows)) + " lookup rows. " + rebuild
        )

    if not manifest_path.exists():
        raise IndexMismatch(
            label + " index: " + str(manifest_path) + " is missing, so the "
            "embeddings cannot be checked against the lookup. " + rebuild
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest.get("lookup_fingerprint")

    if not expected:
        raise IndexMismatch(
            label + " index: " + str(manifest_path) + " predates lookup "
            "fingerprinting, so the embeddings cannot be checked against "
            "the lookup. " + rebuild
        )

    actual = lookup_fingerprint(row["source_path"] for row in rows)

    if actual != expected:
        raise IndexMismatch(
            label + " index: the lookup has been rebuilt since the "
            "embeddings were. Rows were added, removed, or reordered, so "
            "every score would be read against the wrong image. " + rebuild
        )


def load_index(embeddings_path, lookup_path, manifest_path, label):
    """Load an embedding matrix and its verified lookup."""
    matrix = np.load(embeddings_path)
    rows = read_lookup(lookup_path)

    verify_index(matrix, rows, manifest_path, label)

    return matrix, rows


def image_scores(query):
    matrix, rows = load_index(
        IMAGE_EMB, IMAGE_LOOKUP, IMAGE_MANIFEST, "image"
    )

    repo, revision = load_config("image_embedding")

    processor = AutoProcessor.from_pretrained(repo, revision=revision)
    model = AutoModel.from_pretrained(repo, revision=revision).eval().cuda()

    inputs = processor(
        text=[query],
        padding="max_length",
        return_tensors="pt",
    ).to("cuda")

    with torch.inference_mode():
        result = model.get_text_features(**inputs)

    features = getattr(result, "pooler_output", result)
    features = F.normalize(features.float(), p=2, dim=-1)
    vector = features.cpu().numpy()[0]

    del model
    torch.cuda.empty_cache()

    return matrix @ vector, rows


def caption_scores(query):
    matrix, rows = load_index(
        CAPTION_EMB, CAPTION_LOOKUP, CAPTION_MANIFEST, "caption"
    )

    repo, revision = load_config("text_embedding")

    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
    model = AutoModel.from_pretrained(repo, revision=revision).eval().cuda()

    prefixed = (
        "Represent this sentence for searching relevant passages: " + query
    )

    inputs = tokenizer(
        [prefixed],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to("cuda")

    with torch.inference_mode():
        output = model(**inputs)

    pooled = output.last_hidden_state[:, 0]
    pooled = F.normalize(pooled.float(), p=2, dim=-1)
    vector = pooled.cpu().numpy()[0]

    del model
    torch.cuda.empty_cache()

    return matrix @ vector, rows


def content_terms(query):
    return [
        term for term in re.findall(r"[a-z]+", query.lower())
        if term not in STOPWORDS
    ]


def term_hits(query, lookup, allowed=None):
    """Map each path to how many query content terms its caption contains."""
    terms = content_terms(query)

    if not terms:
        return {}, 0

    hits = {}

    for row in lookup:
        if allowed is not None and row["source_path"] not in allowed:
            continue

        caption = row["caption"].lower()

        count = sum(
            1 for term in terms
            if re.search(r"\b" + term + r"s?\b", caption)
        )

        if count:
            hits[row["source_path"]] = count

    return hits, len(terms)


def gradient_cutoff(scores, floor=GRADIENT_FLOOR, ceiling=GRADIENT_CEILING):
    """Return (cutoff, plateau_found)."""
    ordered = np.sort(scores)[::-1][:ceiling]

    if len(ordered) < 3:
        return len(ordered), False

    drops = ordered[:-1] - ordered[1:]
    first = drops[0]

    if first <= 0:
        return len(ordered), False

    for index, drop in enumerate(drops, start=1):
        if drop < first * floor:
            return index, True

    return len(ordered), False


def rrf(paths, k):
    return {
        path: 1.0 / (k + rank)
        for rank, path in enumerate(paths, start=1)
    }


def semantic_order(paths, score_by_path, drop_ratio, trim):
    """Order a candidate set by one modality's similarity score.

    `score_by_path` decides both the order and, under `trim`, which
    tail entries are discarded, so the two can never disagree.

    Equal scores fall back to path order. Callers pass a list built from
    a set, whose iteration order varies with PYTHONHASHSEED, so without
    a tie-break the same query can rank differently between runs.
    """
    ranked = sorted(paths, key=lambda p: (-score_by_path[p], p))

    if not trim or len(ranked) < 4:
        return ranked, []

    best = score_by_path[ranked[0]]
    worst = score_by_path[ranked[-1]]
    spread = best - worst

    if spread <= 0:
        return ranked, []

    threshold = best - drop_ratio * spread

    kept = [p for p in ranked if score_by_path[p] >= threshold]
    dropped = [p for p in ranked if score_by_path[p] < threshold]

    return kept, dropped


def combine_filters(persons, event_names):
    """Intersect the person and event filters.

    Returns (allowed, people_names, person_counts, event_names,
    event_counts). `allowed` is None when neither filter was requested.
    """
    person_paths, resolved_people, person_counts = people.filter_paths(
        persons or []
    )
    event_paths, resolved_events, event_counts = (
        events_module.filter_paths(event_names or [])
    )

    if person_paths is None:
        allowed = event_paths
    elif event_paths is None:
        allowed = person_paths
    else:
        allowed = person_paths & event_paths

    return (allowed, resolved_people, person_counts,
            resolved_events, event_counts)


def empty_result(cap_lookup, allowed, basis, resolved_people,
                 person_counts, resolved_events, event_counts):
    order = [
        row["source_path"] for row in cap_lookup
        if allowed is None or row["source_path"] in allowed
    ]

    return {
        "results": [(path, 0.0) for path in order],
        "score_kind": SCORE_NONE,
        "basis": basis,
        "matched": set(),
        "hits": {},
        "trimmed": [],
        "total_terms": 0,
        "full_count": 0,
        "partial_count": 0,
        "img_cut": 0,
        "img_plateau": True,
        "cap_cut": 0,
        "cap_plateau": True,
        "low_confidence": False,
        "caption_lookup": cap_lookup,
        "caption_score": {},
        "people": resolved_people,
        "person_counts": person_counts,
        "events": resolved_events,
        "event_counts": event_counts,
        "corpus_size": len(cap_lookup),
        "pool_size": len(order),
        "image_rank": {},
        "caption_rank": {},
    }


def search(query, mode="hybrid", top_k=0, rrf_k=RRF_K,
           gradient_floor=GRADIENT_FLOOR,
           min_partial_terms=MIN_PARTIAL_TERMS,
           semantic_drop=SEMANTIC_DROP,
           trim=False,
           persons=None,
           event_names=None):
    """Run hybrid retrieval and derive a result count."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable.")

    (allowed, resolved_people, person_counts,
     resolved_events, event_counts) = combine_filters(persons, event_names)

    if not content_terms(query):
        cap_lookup = read_lookup(CAPTION_LOOKUP)

        parts = []

        if resolved_people:
            parts.append(" and ".join(resolved_people))

        if resolved_events:
            parts.append(" or ".join(resolved_events))

        if parts:
            basis = "filter only - " + ", within ".join(parts)
        else:
            basis = "no query and no filter"

        return empty_result(
            cap_lookup, allowed, basis, resolved_people, person_counts,
            resolved_events, event_counts,
        )

    img_s, img_lookup = image_scores(query)
    cap_s, cap_lookup = caption_scores(query)

    cap_by_path = {
        row["source_path"]: float(cap_s[i])
        for i, row in enumerate(cap_lookup)
    }
    img_by_path = {
        row["source_path"]: float(img_s[i])
        for i, row in enumerate(img_lookup)
    }

    corpus_size = len(cap_lookup)

    def permitted(path):
        return allowed is None or path in allowed

    img_paths = sorted(
        (row["source_path"] for row in img_lookup
         if permitted(row["source_path"])),
        key=lambda path: (-img_by_path[path], path),
    )
    cap_paths = sorted(
        (row["source_path"] for row in cap_lookup
         if permitted(row["source_path"])),
        key=lambda path: (-cap_by_path[path], path),
    )

    pool = len(cap_paths)

    img_rrf = rrf(img_paths, rrf_k)
    cap_rrf = rrf(cap_paths, rrf_k)

    if mode == "image":
        fused = img_rrf
    elif mode == "caption":
        fused = cap_rrf
    else:
        fused = {
            path: img_rrf.get(path, 0.0) + cap_rrf.get(path, 0.0)
            for path in set(img_rrf) | set(cap_rrf)
        }

    # Matched sets keep caption-score ordering under hybrid: RRF order
    # was measured against it and scored no better - see Finding 7 in
    # evals/retrieval-evaluation.md. --mode image opts out of that
    # default rather than being silently overridden by it.
    if mode == "image":
        match_score, score_label = img_by_path, "image score"
        match_kind = SCORE_IMAGE
    else:
        match_score, score_label = cap_by_path, "caption score"
        match_kind = SCORE_CAPTION

    ordered = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))

    hits, total_terms = term_hits(query, cap_lookup, allowed)
    full = {p for p, n in hits.items() if n == total_terms}

    threshold = min(min_partial_terms, total_terms) if total_terms else 0
    partial = {p for p, n in hits.items() if threshold and n >= threshold}

    cap_subset = np.array(
        [cap_by_path[p] for p in cap_paths]
    ) if cap_paths else np.array([0.0])
    img_subset = np.array(
        [img_by_path[p] for p in img_paths]
    ) if img_paths else np.array([0.0])

    img_cut, img_found = gradient_cutoff(
        img_s if allowed is None else img_subset, gradient_floor
    )
    cap_cut, cap_found = gradient_cutoff(
        cap_s if allowed is None else cap_subset, gradient_floor
    )

    low_confidence = False
    trimmed = []

    if not ordered:
        results = []
        basis = "no images match the filter"
        matched = set()
        score_kind = SCORE_NONE
    elif top_k:
        results = ordered[:top_k]
        basis = "fixed count (--top-k " + str(top_k) + ")"
        matched = set()
        score_kind = SCORE_FUSED
    elif full:
        kept, trimmed = semantic_order(
            list(full), match_score, semantic_drop, trim
        )
        results = [(p, match_score[p]) for p in kept]
        matched = full
        score_kind = match_kind
        basis = ("full caption match - " + str(len(full)) + " of "
                 + str(pool) + " captions contain all "
                 + str(total_terms)
                 + (" term" if total_terms == 1 else " terms"))
    elif partial:
        kept, trimmed = semantic_order(
            list(partial), match_score, semantic_drop, trim
        )
        results = [(p, match_score[p]) for p in kept]
        matched = partial
        score_kind = match_kind
        basis = ("partial caption match - at least " + str(threshold)
                 + " of " + str(total_terms) + " terms")
    else:
        results = ordered[:max(img_cut, cap_cut)]
        basis = "score gradient - no caption mentions these terms"
        matched = set()
        score_kind = SCORE_FUSED
        low_confidence = not (img_found or cap_found)

    if trimmed:
        basis += ", " + str(len(trimmed)) + " trimmed by " + score_label

    return {
        "results": results,
        "score_kind": score_kind,
        "basis": basis,
        "matched": matched,
        "hits": hits,
        "trimmed": trimmed,
        "total_terms": total_terms,
        "full_count": len(full),
        "partial_count": len(partial),
        "img_cut": img_cut,
        "img_plateau": img_found,
        "cap_cut": cap_cut,
        "cap_plateau": cap_found,
        "low_confidence": low_confidence,
        "caption_lookup": cap_lookup,
        "caption_score": cap_by_path,
        "people": resolved_people,
        "person_counts": person_counts,
        "events": resolved_events,
        "event_counts": event_counts,
        "corpus_size": corpus_size,
        "pool_size": pool,
        "image_rank": {p: i for i, p in enumerate(img_paths, start=1)},
        "caption_rank": {p: i for i, p in enumerate(cap_paths, start=1)},
    }
