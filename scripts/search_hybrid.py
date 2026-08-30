"""Hybrid search with a derived result count.

Fuses SigLIP2 image retrieval and BGE caption retrieval using Reciprocal
Rank Fusion, then decides how many results to return rather than always
returning k.

Two signals drive the cutoff:

  1. Caption term matching. If the query terms appear literally in the
     captions, those images are ground truth: they set both the count and
     the result set.
  2. Score gradient. A present concept produces a steep similarity decay;
     an absent one produces a plateau of equidistant neighbours. Where the
     curve flattens, the useful results have ended.
"""
import argparse
import csv
import re
import sys
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
STOPWORDS = {"a", "an", "the", "of", "in", "on", "at", "with", "and", "or"}


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


def literal_matches(query, lookup):
    """Paths whose caption contains every content word of the query."""
    terms = [
        term for term in re.findall(r"[a-z]+", query.lower())
        if term not in STOPWORDS
    ]

    if not terms:
        return set()

    matched = set()

    for row in lookup:
        caption = row["caption"].lower()

        if all(re.search(r"\b" + term + r"s?\b", caption) for term in terms):
            matched.add(row["source_path"])

    return matched


def gradient_cutoff(scores, floor=0.35, max_results=40):
    """How many leading results sit on the steep part of the decay."""
    ordered = np.sort(scores)[::-1][:max_results]

    if len(ordered) < 3:
        return len(ordered)

    drops = ordered[:-1] - ordered[1:]
    first = drops[0]

    if first <= 0:
        return len(ordered)

    for index, drop in enumerate(drops, start=1):
        if drop < first * floor:
            return index

    return len(ordered)


def rrf(paths, k):
    return {
        path: 1.0 / (k + rank)
        for rank, path in enumerate(paths, start=1)
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Force a fixed result count. Default 0 derives the count.",
    )
    parser.add_argument("--rrf-k", type=int, default=RRF_K)
    parser.add_argument("--gradient-floor", type=float, default=0.35)
    parser.add_argument(
        "--mode",
        choices=["hybrid", "image", "caption"],
        default="hybrid",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable.")

    print()
    print("=" * 76)
    print("HYBRID SEARCH - " + args.query)
    print("=" * 76)

    img_s, img_lookup = image_scores(args.query)
    cap_s, cap_lookup = caption_scores(args.query)

    img_order = np.argsort(img_s)[::-1]
    cap_order = np.argsort(cap_s)[::-1]

    img_paths = [img_lookup[i]["source_path"] for i in img_order]
    cap_paths = [cap_lookup[i]["source_path"] for i in cap_order]

    img_rrf = rrf(img_paths, args.rrf_k)
    cap_rrf = rrf(cap_paths, args.rrf_k)

    if args.mode == "image":
        fused = img_rrf
    elif args.mode == "caption":
        fused = cap_rrf
    else:
        fused = {
            path: img_rrf.get(path, 0.0) + cap_rrf.get(path, 0.0)
            for path in set(img_rrf) | set(cap_rrf)
        }

    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)

    exact = literal_matches(args.query, cap_lookup)
    img_cut = gradient_cutoff(img_s, args.gradient_floor)
    cap_cut = gradient_cutoff(cap_s, args.gradient_floor)

    if args.top_k:
        results = ordered[:args.top_k]
        basis = "fixed (--top-k)"
    elif exact:
        # Caption matches are ground truth: they define the result set,
        # not merely its size. Rank them among themselves by fused score.
        results = [item for item in ordered if item[0] in exact]
        basis = "caption term match"
    else:
        results = ordered[:max(img_cut, cap_cut)]
        basis = "score gradient"

    print()
    print("-" * 76)
    print("RESULT COUNT")
    print("-" * 76)
    print("Caption term matches: " + str(len(exact)))
    print("Image gradient cut:   " + str(img_cut))
    print("Caption gradient cut: " + str(cap_cut))
    print("Returning:            " + str(len(results)) + "  (" + basis + ")")

    if not exact and max(img_cut, cap_cut) <= 2:
        print()
        print("NOTE: no caption mentions this concept and both score curves")
        print("      are flat. This query may have no matches in the corpus.")

    by_path = {r["source_path"]: r for r in cap_lookup}
    img_rank = {p: i for i, p in enumerate(img_paths, start=1)}
    cap_rank = {p: i for i, p in enumerate(cap_paths, start=1)}

    print()

    for rank, (path, score) in enumerate(results, start=1):
        row = by_path.get(path, {})
        name = row.get("filename", Path(path).name)
        caption = row.get("caption", "")
        mark = "*" if path in exact else " "

        print("#" + str(rank) + mark + "  " + name)
        print("     rrf=" + format(score, ".5f")
              + "  image_rank=" + str(img_rank.get(path, "-"))
              + "  caption_rank=" + str(cap_rank.get(path, "-")))
        print("     " + caption[:150])
        print()

    if exact:
        print("* = query terms appear literally in the caption")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
