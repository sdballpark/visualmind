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

Trimming the tail of a matched set, by whichever score ordered it, is
available behind the `trim` flag but is off by default. See
evals/retrieval-evaluation.md.
"""
import csv
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
CAPTION_EMB = INDEX_DIR / "caption_embeddings.npy"
CAPTION_LOOKUP = INDEX_DIR / "caption_lookup.csv"

RRF_K = 60
MIN_PARTIAL_TERMS = 2
GRADIENT_CEILING = 40
GRADIENT_FLOOR = 0.35
SEMANTIC_DROP = 0.80

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


def image_scores(query):
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

    return np.load(IMAGE_EMB) @ vector, read_lookup(IMAGE_LOOKUP)


def caption_scores(query):
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

    return np.load(CAPTION_EMB) @ vector, read_lookup(CAPTION_LOOKUP)


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
    """
    ranked = sorted(paths, key=lambda p: score_by_path[p], reverse=True)

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

    img_paths = [
        img_lookup[i]["source_path"] for i in np.argsort(img_s)[::-1]
        if permitted(img_lookup[i]["source_path"])
    ]
    cap_paths = [
        cap_lookup[i]["source_path"] for i in np.argsort(cap_s)[::-1]
        if permitted(cap_lookup[i]["source_path"])
    ]

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
    else:
        match_score, score_label = cap_by_path, "caption score"

    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)

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
    elif top_k:
        results = ordered[:top_k]
        basis = "fixed count (--top-k " + str(top_k) + ")"
        matched = set()
    elif full:
        kept, trimmed = semantic_order(
            list(full), match_score, semantic_drop, trim
        )
        results = [(p, match_score[p]) for p in kept]
        matched = full
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
        basis = ("partial caption match - at least " + str(threshold)
                 + " of " + str(total_terms) + " terms")
    else:
        results = ordered[:max(img_cut, cap_cut)]
        basis = "score gradient - no caption mentions these terms"
        matched = set()
        low_confidence = not (img_found or cap_found)

    if trimmed:
        basis += ", " + str(len(trimmed)) + " trimmed by " + score_label

    return {
        "results": results,
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
