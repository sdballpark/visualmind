"""Shared retrieval logic for VisualMind search entry points.

Both scripts/search_hybrid.py (console) and scripts/search_gallery.py
(HTML) import from here so their behaviour cannot drift apart.

Term matching decides which images are candidates; the caption semantic
score decides their order. Term matching has no notion of subject and
object - "holding" and "baby" both appear in a caption describing a baby
holding a ladle - so it bounds the candidate set rather than ranking it.

Trimming the tail of a matched set by caption score is available behind
the `trim` flag but is off by default. Measured on a relational query it
raised precision from 83% to 87.5% while discarding a correct match, and
the errors it was meant to catch scored mid-pack rather than last. See
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


def term_hits(query, lookup):
    """Map each path to how many query content terms its caption contains."""
    terms = content_terms(query)

    if not terms:
        return {}, 0

    hits = {}

    for row in lookup:
        caption = row["caption"].lower()

        count = sum(
            1 for term in terms
            if re.search(r"\b" + term + r"s?\b", caption)
        )

        if count:
            hits[row["source_path"]] = count

    return hits, len(terms)


def gradient_cutoff(scores, floor=GRADIENT_FLOOR, ceiling=GRADIENT_CEILING):
    """Return (cutoff, plateau_found).

    plateau_found is False when the curve never flattened, meaning the
    cutoff is the ceiling rather than a decision.
    """
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


def semantic_order(paths, cap_by_path, drop_ratio, trim):
    """Order a candidate set by caption similarity.

    Returns (kept, trimmed). When `trim` is False nothing is discarded,
    which is the default: a silently omitted true match is worse than a
    visible false positive in a personal archive.
    """
    ranked = sorted(paths, key=lambda p: cap_by_path[p], reverse=True)

    if not trim or len(ranked) < 4:
        return ranked, []

    best = cap_by_path[ranked[0]]
    worst = cap_by_path[ranked[-1]]
    spread = best - worst

    if spread <= 0:
        return ranked, []

    threshold = best - drop_ratio * spread

    kept = [p for p in ranked if cap_by_path[p] >= threshold]
    dropped = [p for p in ranked if cap_by_path[p] < threshold]

    return kept, dropped


def search(query, mode="hybrid", top_k=0, rrf_k=RRF_K,
           gradient_floor=GRADIENT_FLOOR,
           min_partial_terms=MIN_PARTIAL_TERMS,
           semantic_drop=SEMANTIC_DROP,
           trim=False):
    """Run hybrid retrieval and derive a result count."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable.")

    img_s, img_lookup = image_scores(query)
    cap_s, cap_lookup = caption_scores(query)

    img_paths = [
        img_lookup[i]["source_path"] for i in np.argsort(img_s)[::-1]
    ]
    cap_paths = [
        cap_lookup[i]["source_path"] for i in np.argsort(cap_s)[::-1]
    ]

    cap_by_path = {
        row["source_path"]: float(cap_s[i])
        for i, row in enumerate(cap_lookup)
    }

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

    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)

    hits, total_terms = term_hits(query, cap_lookup)
    full = {p for p, n in hits.items() if n == total_terms}

    threshold = min(min_partial_terms, total_terms) if total_terms else 0
    partial = {p for p, n in hits.items() if threshold and n >= threshold}

    img_cut, img_found = gradient_cutoff(img_s, gradient_floor)
    cap_cut, cap_found = gradient_cutoff(cap_s, gradient_floor)

    low_confidence = False
    trimmed = []

    if top_k:
        results = ordered[:top_k]
        basis = "fixed count (--top-k " + str(top_k) + ")"
        matched = set()
    elif full:
        kept, trimmed = semantic_order(
            list(full), cap_by_path, semantic_drop, trim
        )
        results = [(p, cap_by_path[p]) for p in kept]
        matched = full
        basis = ("full caption match - " + str(len(full)) + " of "
                 + str(len(cap_lookup)) + " captions contain all "
                 + str(total_terms)
                 + (" term" if total_terms == 1 else " terms"))
    elif partial:
        kept, trimmed = semantic_order(
            list(partial), cap_by_path, semantic_drop, trim
        )
        results = [(p, cap_by_path[p]) for p in kept]
        matched = partial
        basis = ("partial caption match - at least " + str(threshold)
                 + " of " + str(total_terms) + " terms")
    else:
        results = ordered[:max(img_cut, cap_cut)]
        basis = "score gradient - no caption mentions these terms"
        matched = set()
        low_confidence = not (img_found or cap_found)

    if trimmed:
        basis += ", " + str(len(trimmed)) + " trimmed by caption score"

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
        "image_rank": {p: i for i, p in enumerate(img_paths, start=1)},
        "caption_rank": {p: i for i, p in enumerate(cap_paths, start=1)},
    }
