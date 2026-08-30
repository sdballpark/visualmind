from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from transformers import AutoModel, AutoProcessor


register_heif_opener()

CATALOG = Path("data/metadata/image_catalog.csv")
MODEL_CONFIG = Path("configs/models.yaml")

INDEX_DIR = Path("indexes")
EMBEDDINGS_PATH = INDEX_DIR / "siglip2_image_embeddings.npy"
LOOKUP_PATH = INDEX_DIR / "siglip2_lookup.csv"
INDEX_INFO_PATH = INDEX_DIR / "siglip2_index.json"

BATCH_SIZE = 32


def get_feature_tensor(value):
    if isinstance(value, torch.Tensor):
        return value

    if hasattr(value, "pooler_output"):
        return value.pooler_output

    raise TypeError(
        f"Unexpected feature result: {type(value)}"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def load_model_config():
    config = yaml.safe_load(
        MODEL_CONFIG.read_text(encoding="utf-8")
    )

    entry = config["models"]["image_embedding"]

    return entry["repo_id"], entry["revision"]


def load_catalog():
    with CATALOG.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as handle:
        return list(csv.DictReader(handle))


def load_image(path: Path):
    with Image.open(path) as raw:
        image = ImageOps.exif_transpose(raw)
        return image.convert("RGB")


def main():
    print()
    print("=" * 76)
    print("VISUALMIND - BUILD SIGLIP2 IMAGE EMBEDDING INDEX")
    print("=" * 76)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    device = torch.device("cuda")

    model_id, revision = load_model_config()
    rows = load_catalog()

    print(f"Images:          {len(rows):,}")
    print(f"Model:           {model_id}")
    print(f"Revision:        {revision}")
    print(f"GPU:             {torch.cuda.get_device_name(0)}")
    print(f"Batch size:      {BATCH_SIZE}")
    print()

    print("Loading processor from shared HF cache...")

    processor = AutoProcessor.from_pretrained(
        model_id,
        revision=revision,
        local_files_only=True,
    )

    print("Loading model from shared HF cache...")

    model = AutoModel.from_pretrained(
        model_id,
        revision=revision,
        dtype=torch.float16,
        local_files_only=True,
    )

    model = model.to(device)
    model.eval()

    all_embeddings = []

    total_batches = (
        len(rows) + BATCH_SIZE - 1
    ) // BATCH_SIZE

    for batch_number, start in enumerate(
        range(0, len(rows), BATCH_SIZE),
        start=1,
    ):
        batch_rows = rows[start:start + BATCH_SIZE]

        images = [
            load_image(Path(row["source_path"]))
            for row in batch_rows
        ]

        inputs = processor(
            images=images,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(device)
            if isinstance(value, torch.Tensor)
            else value
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            result = model.get_image_features(**inputs)

        features = get_feature_tensor(result)

        # Normalize in FP32 for numerical stability.
        features = F.normalize(
            features.float(),
            p=2,
            dim=-1,
        )

        all_embeddings.append(
            features.cpu().numpy()
        )

        print(
            f"Batch {batch_number:>2}/{total_batches}  "
            f"images {start + 1:>3}-"
            f"{start + len(batch_rows):>3}"
        )

    embedding_matrix = np.concatenate(
        all_embeddings,
        axis=0,
    ).astype(np.float32)

    INDEX_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        EMBEDDINGS_PATH,
        embedding_matrix,
    )

    lookup_fields = [
        "row_index",
        "image_id",
        "source_path",
        "relative_path",
        "filename",
        "provenance_status",
        "gmail_directory_year",
        "gmail_directory_month",
        "gmail_message_date",
        "gmail_subject",
        "best_exif_date",
    ]

    with LOOKUP_PATH.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=lookup_fields,
        )

        writer.writeheader()

        for index, row in enumerate(rows):
            writer.writerow(
                {
                    "row_index": index,
                    "image_id": row["image_id"],
                    "source_path": row["source_path"],
                    "relative_path": row["relative_path"],
                    "filename": row["filename"],
                    "provenance_status":
                        row["provenance_status"],
                    "gmail_directory_year":
                        row["gmail_directory_year"],
                    "gmail_directory_month":
                        row["gmail_directory_month"],
                    "gmail_message_date":
                        row["gmail_message_date"],
                    "gmail_subject":
                        row["gmail_subject"],
                    "best_exif_date":
                        row["best_exif_date"],
                }
            )

    norms = np.linalg.norm(
        embedding_matrix,
        axis=1,
    )

    index_info = {
        "created_utc":
            datetime.now(timezone.utc).isoformat(),

        "model_id": model_id,
        "revision": revision,

        "catalog": str(CATALOG),
        "catalog_sha256":
            sha256_file(CATALOG),

        "image_count":
            int(embedding_matrix.shape[0]),

        "embedding_dimension":
            int(embedding_matrix.shape[1]),

        "dtype":
            str(embedding_matrix.dtype),

        "minimum_l2_norm":
            float(norms.min()),

        "maximum_l2_norm":
            float(norms.max()),
    }

    INDEX_INFO_PATH.write_text(
        json.dumps(
            index_info,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("-" * 76)
    print("EMBEDDING INDEX SUMMARY")
    print("-" * 76)

    print(
        f"Embedding matrix:   "
        f"{embedding_matrix.shape}"
    )

    print(
        f"Datatype:           "
        f"{embedding_matrix.dtype}"
    )

    print(
        f"Minimum L2 norm:    "
        f"{norms.min():.6f}"
    )

    print(
        f"Maximum L2 norm:    "
        f"{norms.max():.6f}"
    )

    print(
        f"GPU peak memory:    "
        f"{torch.cuda.max_memory_allocated()/1024**3:.2f} GB"
    )

    print()
    print(f"Embeddings: {EMBEDDINGS_PATH.resolve()}")
    print(f"Lookup:     {LOOKUP_PATH.resolve()}")
    print(f"Metadata:   {INDEX_INFO_PATH.resolve()}")

    print()
    print("=" * 76)
    print("SIGLIP2 IMAGE INDEX COMPLETE")
    print("=" * 76)
    print()


if __name__ == "__main__":
    main()
