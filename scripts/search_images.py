from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from transformers import AutoModel, AutoProcessor


MODEL_CONFIG = Path("configs/models.yaml")
EMBEDDINGS_PATH = Path(
    "indexes/siglip2_image_embeddings.npy"
)
LOOKUP_PATH = Path(
    "indexes/siglip2_lookup.csv"
)


def get_feature_tensor(value):
    if isinstance(value, torch.Tensor):
        return value

    if hasattr(value, "pooler_output"):
        return value.pooler_output

    raise TypeError(
        f"Unexpected feature result: {type(value)}"
    )


def load_model_config():
    config = yaml.safe_load(
        MODEL_CONFIG.read_text(
            encoding="utf-8"
        )
    )

    entry = config["models"]["image_embedding"]

    return entry["repo_id"], entry["revision"]


def load_lookup():
    with LOOKUP_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Search VisualMind images "
            "using natural language."
        )
    )

    parser.add_argument(
        "query",
        help="Natural-language image query.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable.")

    device = torch.device("cuda")

    embeddings = np.load(
        EMBEDDINGS_PATH
    )

    lookup = load_lookup()

    if len(lookup) != len(embeddings):
        raise RuntimeError(
            "Lookup and embedding counts differ."
        )

    model_id, revision = load_model_config()

    processor = AutoProcessor.from_pretrained(
        model_id,
        revision=revision,
        local_files_only=True,
    )

    model = AutoModel.from_pretrained(
        model_id,
        revision=revision,
        dtype=torch.float16,
        local_files_only=True,
    ).to(device)

    model.eval()

    # SigLIP2 was trained with lowercase text and
    # max-length padding. HF also recommends the
    # "This is a photo of ..." prompt form.
    query = args.query.strip().lower()

    prompt = f"This is a photo of {query}."

    inputs = processor(
        text=[prompt],
        padding="max_length",
        max_length=64,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        result = model.get_text_features(
            **inputs
        )

    text_features = get_feature_tensor(
        result
    )

    text_features = F.normalize(
        text_features.float(),
        p=2,
        dim=-1,
    )

    query_vector = (
        text_features[0]
        .cpu()
        .numpy()
    )

    scores = embeddings @ query_vector

    top_k = min(
        args.top_k,
        len(scores),
    )

    indices = np.argsort(scores)[::-1][:top_k]

    print()
    print("=" * 76)
    print("VISUALMIND - SEMANTIC IMAGE SEARCH")
    print("=" * 76)
    print(f"Query:  {args.query}")
    print(f"Prompt: {prompt}")
    print()

    for rank, index in enumerate(
        indices,
        start=1,
    ):
        row = lookup[index]

        print(
            f"{rank:>2}. "
            f"score={scores[index]:.4f}"
        )

        print(
            f"    File:    "
            f"{row['filename']}"
        )

        print(
            f"    Year:    "
            f"{row['gmail_directory_year']}"
        )

        if row["gmail_subject"]:
            print(
                f"    Subject: "
                f"{row['gmail_subject']}"
            )

        if row["best_exif_date"]:
            print(
                f"    EXIF:    "
                f"{row['best_exif_date']}"
            )

        print(
            f"    Path:    "
            f"{row['source_path']}"
        )

        print()

    print("=" * 76)
    print("SEARCH COMPLETE")
    print("=" * 76)
    print()


if __name__ == "__main__":
    main()
