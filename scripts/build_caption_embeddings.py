"""Embed VLM captions with a text encoder for hybrid retrieval.

Reads data/metadata/captions.csv, writes indexes/caption_embeddings.npy
and indexes/caption_lookup.csv.
"""
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from transformers import AutoModel, AutoTokenizer

from visualmind.retrieval import lookup_fingerprint

CAPTIONS = Path("data/metadata/captions.csv")
MODEL_CONFIG = Path("configs/models.yaml")
INDEX_DIR = Path("indexes")
EMBEDDINGS_PATH = INDEX_DIR / "caption_embeddings.npy"
LOOKUP_PATH = INDEX_DIR / "caption_lookup.csv"
INDEX_INFO_PATH = INDEX_DIR / "caption_index.json"

BATCH_SIZE = 32


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)

    return digest.hexdigest()


def load_model_config():
    config = yaml.safe_load(MODEL_CONFIG.read_text(encoding="utf-8"))
    entry = config["models"]["text_embedding"]
    return entry["repo_id"], entry["revision"]


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable.")

    repo, revision = load_model_config()

    with CAPTIONS.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    print()
    print("=" * 76)
    print("VISUALMIND - BUILD CAPTION EMBEDDING INDEX")
    print("=" * 76)
    print(f"Captions:        {len(rows)}")
    print(f"Model:           {repo}")
    print(f"Revision:        {revision}")
    print(f"GPU:             {torch.cuda.get_device_name(0)}")
    print(f"Batch size:      {BATCH_SIZE}")

    print("\nLoading model from shared HF cache...")
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
    model = AutoModel.from_pretrained(repo, revision=revision).eval().cuda()

    vectors = []

    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start:start + BATCH_SIZE]
        texts = [r["caption"] for r in batch]

        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to("cuda")

        with torch.inference_mode():
            output = model(**inputs)

        pooled = output.last_hidden_state[:, 0]
        pooled = F.normalize(pooled.float(), p=2, dim=-1)

        vectors.append(pooled.cpu().numpy())

        done = min(start + BATCH_SIZE, len(rows))
        print(f"  {done}/{len(rows)}")

    matrix = np.vstack(vectors).astype(np.float32)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_PATH, matrix)

    with LOOKUP_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source_path", "filename", "caption"],
        )
        writer.writeheader()

        for row in rows:
            writer.writerow({
                "source_path": row["source_path"],
                "filename": row["filename"],
                "caption": row["caption"],
            })

    norms = np.linalg.norm(matrix, axis=1)

    index_info = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": repo,
        "revision": revision,
        "captions": str(CAPTIONS),
        "captions_sha256": sha256_file(CAPTIONS),
        "caption_count": int(matrix.shape[0]),
        "embedding_dimension": int(matrix.shape[1]),
        "dtype": str(matrix.dtype),
        "minimum_l2_norm": float(norms.min()),
        "maximum_l2_norm": float(norms.max()),

        # Binds this matrix to the exact lookup row order written
        # above. retrieval.py refuses the pair if they diverge.
        "lookup_fingerprint": lookup_fingerprint(
            row["source_path"] for row in rows
        ),
    }

    INDEX_INFO_PATH.write_text(
        json.dumps(index_info, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("-" * 76)
    print("CAPTION INDEX SUMMARY")
    print("-" * 76)
    print(f"Embedding matrix:   {matrix.shape}")
    print(f"Datatype:           {matrix.dtype}")
    print(f"Minimum L2 norm:    {norms.min():.6f}")
    print(f"Maximum L2 norm:    {norms.max():.6f}")
    print(f"GPU peak memory:    {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    print(f"\nEmbeddings: {EMBEDDINGS_PATH.resolve()}")
    print(f"Lookup:     {LOOKUP_PATH.resolve()}")
    print(f"Manifest:   {INDEX_INFO_PATH.resolve()}")
    print()
    print("=" * 76)
    print("CAPTION INDEX COMPLETE")
    print("=" * 76)

    return 0


if __name__ == "__main__":
    sys.exit(main())
