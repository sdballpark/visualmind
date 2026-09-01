"""Build a DINOv2 visual embedding index for near-duplicate detection.

DINOv2 has no text tower. This index answers image-to-image questions
only - "which of these are the same photograph" - and is deliberately
separate from the SigLIP2 index used for text search.

Writes indexes/dinov2_embeddings.npy, indexes/dinov2_lookup.csv and
indexes/dinov2_index.json.
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
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from visualmind.retrieval import lookup_fingerprint

CATALOG = Path("data/metadata/image_catalog.csv")
MODEL_CONFIG = Path("configs/models.yaml")
INDEX_DIR = Path("indexes")
EMBEDDINGS_PATH = INDEX_DIR / "dinov2_embeddings.npy"
LOOKUP_PATH = INDEX_DIR / "dinov2_lookup.csv"
INDEX_INFO_PATH = INDEX_DIR / "dinov2_index.json"

BATCH_SIZE = 32


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)

    return digest.hexdigest()


def load_model_config():
    config = yaml.safe_load(MODEL_CONFIG.read_text(encoding="utf-8"))
    entry = config["models"]["visual_embedding"]
    return entry["repo_id"], entry["revision"]


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable.")

    repo, revision = load_model_config()

    with CATALOG.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    print()
    print("=" * 76)
    print("VISUALMIND - BUILD DINOV2 VISUAL INDEX")
    print("=" * 76)
    print("Images:          " + str(len(rows)))
    print("Model:           " + repo)
    print("Revision:        " + revision)
    print("GPU:             " + torch.cuda.get_device_name(0))
    print("Batch size:      " + str(BATCH_SIZE))

    print("\nLoading model from shared HF cache...")
    processor = AutoImageProcessor.from_pretrained(repo, revision=revision)
    model = AutoModel.from_pretrained(repo, revision=revision).eval().cuda()

    vectors = []
    kept = []
    failures = 0

    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start:start + BATCH_SIZE]
        images = []

        for row in batch:
            try:
                with Image.open(row["source_path"]) as image:
                    images.append(image.convert("RGB"))
                kept.append(row)
            except Exception as error:
                print("  SKIP " + row["filename"] + ": " + str(error))
                failures += 1

        if not images:
            continue

        inputs = processor(images=images, return_tensors="pt").to("cuda")

        with torch.inference_mode():
            output = model(**inputs)

        # DINOv2 exposes a CLS token in pooler_output; fall back to the
        # first token of the sequence if a revision omits it.
        pooled = getattr(output, "pooler_output", None)

        if pooled is None:
            pooled = output.last_hidden_state[:, 0]

        pooled = F.normalize(pooled.float(), p=2, dim=-1)
        vectors.append(pooled.cpu().numpy())

        done = min(start + BATCH_SIZE, len(rows))
        print("  " + str(done) + "/" + str(len(rows)))

    matrix = np.vstack(vectors).astype(np.float32)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_PATH, matrix)

    with LOOKUP_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_path", "filename", "sha256",
                "width", "height", "bytes", "best_exif_date",
            ],
        )
        writer.writeheader()

        for row in kept:
            writer.writerow({
                "source_path": row["source_path"],
                "filename": row["filename"],
                "sha256": row.get("sha256", ""),
                "width": row.get("width", ""),
                "height": row.get("height", ""),
                "bytes": row.get("bytes", ""),
                "best_exif_date": row.get("best_exif_date", ""),
            })

    norms = np.linalg.norm(matrix, axis=1)

    index_info = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": repo,
        "revision": revision,
        "catalog": str(CATALOG),
        "catalog_sha256": sha256_file(CATALOG),
        "image_count": int(matrix.shape[0]),
        "embedding_dimension": int(matrix.shape[1]),
        "dtype": str(matrix.dtype),
        "minimum_l2_norm": float(norms.min()),
        "maximum_l2_norm": float(norms.max()),

        # Over `kept`, not `rows`. This builder skips an image it cannot
        # open, so the lookup and the matrix hold only the images that
        # survived - fingerprinting the catalog order instead would
        # describe an index that was never written.
        "lookup_fingerprint": lookup_fingerprint(
            row["source_path"] for row in kept
        ),
    }

    INDEX_INFO_PATH.write_text(
        json.dumps(index_info, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("-" * 76)
    print("VISUAL INDEX SUMMARY")
    print("-" * 76)
    print("Embedding matrix:   " + str(matrix.shape))
    print("Datatype:           " + str(matrix.dtype))
    print("Failed images:      " + str(failures))
    print("Minimum L2 norm:    " + format(norms.min(), ".6f"))
    print("Maximum L2 norm:    " + format(norms.max(), ".6f"))
    print("GPU peak memory:    "
          + format(torch.cuda.max_memory_allocated() / 1e9, ".2f") + " GB")
    print("\nEmbeddings: " + str(EMBEDDINGS_PATH.resolve()))
    print("Lookup:     " + str(LOOKUP_PATH.resolve()))
    print("Manifest:   " + str(INDEX_INFO_PATH.resolve()))
    print()
    print("=" * 76)
    print("VISUAL INDEX COMPLETE")
    print("=" * 76)

    return 0


if __name__ == "__main__":
    sys.exit(main())
