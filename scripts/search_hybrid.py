"""Hybrid search: fuse SigLIP2 image retrieval with BGE caption retrieval.

Scores from the two indexes are not comparable (SigLIP sits near 0.10,
BGE near 0.7), so they are combined with Reciprocal Rank Fusion, which
uses only ordering.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from transformers import (
    AutoModel,
    AutoProcessor,
    AutoTokenizer,
)

MODEL_CONFIG = Path("configs/models.yaml")
INDEX_DIR = Path("indexes")

IMAGE_EMB = INDEX_DIR / "siglip2_image_embeddings.npy"
IMAGE_LOOKUP = INDEX_DIR / "siglip2_lookup.csv"
CAPTION_EMB = INDEX_DIR / "caption_embeddings.npy"
CAPTION_LOOKUP = INDEX_DIR / "caption_lookup.csv"

RRF_K = 60


def load_config(role: str):
    config = yaml.safe_load(MODEL_CONFIG.read_text(encoding="utf-8"))
    entry = config["models"][role]
    return entry["repo_id"], entry["revision"]


def read_lookup(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def image_scores(query: str) -> tuple[np.ndarray, list[dict]]:
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


def caption_scores(query: str) -> tuple[np.ndarray, list[dict]]:
    repo, revision = load_config("text_embedding")

    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
    model = AutoModel.from_pretrained(repo, revision=revision).eval().cuda()

    # BGE recommends an instruction prefix on the query side only.
    prefixed = f"Represent this sentence for searching relevant passages: {query}"

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


def rrf(ranked_paths: list[str], k: int) -> dict[str, float]:
    return {
        path: 1.0 / (k + rank)
        for rank, path in enumerate(ranked_paths, start=1)
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--rrf-k", type=int, default=RRF_K)
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
    print(f"HYBRID SEARCH - {args.query}")
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

    by_path = {r["source_path"]: r for r in cap_lookup}
    img_rank = {p: i for i, p in enumerate(img_paths, start=1)}
    cap_rank = {p: i for i, p in enumerate(cap_paths, start=1)}

    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)

    print(f"\nMode: {args.mode}   RRF k={args.rrf_k}\n")

    for rank, (path, score) in enumerate(ordered[:args.top_k], start=1):
        row = by_path.get(path, {})
        name = row.get("filename", Path(path).name)
        caption = row.get("caption", "")

        print(f"#{rank:<3} {name}")
        print(f"     rrf={score:.5f}  "
              f"image_rank={img_rank.get(path, '-'):<4} "
              f"caption_rank={cap_rank.get(path, '-')}")
        print(f"     {caption[:150]}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
