from __future__ import annotations

import argparse
import base64
import csv
import html
import io
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageOps
from transformers import AutoModel, AutoProcessor


MODEL_CONFIG = Path("configs/models.yaml")
EMBEDDINGS_PATH = Path("indexes/siglip2_image_embeddings.npy")
LOOKUP_PATH = Path("indexes/siglip2_lookup.csv")
OUTPUT_ROOT = Path("outputs/search")


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
        MODEL_CONFIG.read_text(encoding="utf-8")
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


def make_thumbnail_data_uri(path: Path) -> str:
    try:
        with Image.open(path) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
            image.thumbnail((420, 420))

            buffer = io.BytesIO()

            image.save(
                buffer,
                format="JPEG",
                quality=85,
                optimize=True,
            )

            encoded = base64.b64encode(
                buffer.getvalue()
            ).decode("ascii")

            return f"data:image/jpeg;base64,{encoded}"

    except Exception as exc:
        print(f"Thumbnail error: {path} - {exc}")
        return ""


def safe_name(value: str) -> str:
    value = value.lower().strip()

    value = re.sub(
        r"[^a-z0-9]+",
        "-",
        value,
    )

    return value.strip("-") or "search"


def main():
    parser = argparse.ArgumentParser(
        description="Create VisualMind semantic search gallery."
    )

    parser.add_argument("query")

    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable.")

    device = torch.device("cuda")

    embeddings = np.load(EMBEDDINGS_PATH)
    lookup = load_lookup()

    if len(lookup) != len(embeddings):
        raise RuntimeError(
            "Embedding and lookup counts differ."
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
        result = model.get_text_features(**inputs)

    features = get_feature_tensor(result)

    features = F.normalize(
        features.float(),
        p=2,
        dim=-1,
    )

    query_vector = features[0].cpu().numpy()

    scores = embeddings @ query_vector

    top_k = min(args.top_k, len(scores))

    indices = np.argsort(scores)[::-1][:top_k]

    cards = []

    for rank, index in enumerate(indices, start=1):
        row = lookup[index]

        image_path = Path(row["source_path"])

        thumbnail = make_thumbnail_data_uri(
            image_path
        )

        cards.append(
            f"""
            <article class="card">
                <div class="rank">#{rank}</div>

                <img
                    src="{thumbnail}"
                    alt="{html.escape(row['filename'])}"
                >

                <div class="content">

                    <div class="score">
                        Similarity: {scores[index]:.4f}
                    </div>

                    <h3>
                        {html.escape(row['filename'])}
                    </h3>

                    <p>
                        <strong>Gmail year:</strong>
                        {html.escape(row['gmail_directory_year'])}
                    </p>

                    <p>
                        <strong>EXIF:</strong>
                        {html.escape(row['best_exif_date'] or 'Unavailable')}
                    </p>

                    <p>
                        <strong>Subject:</strong>
                        {html.escape(row['gmail_subject'] or 'Unavailable')}
                    </p>

                    <p class="path">
                        {html.escape(row['source_path'])}
                    </p>

                </div>
            </article>
            """
        )

    page = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>VisualMind Search - {html.escape(args.query)}</title>

<style>

body {{
    font-family: Arial, sans-serif;
    margin: 30px;
    background: #f4f4f4;
}}

h1 {{
    margin-bottom: 4px;
}}

.subtitle {{
    margin-bottom: 30px;
}}

.grid {{
    display: grid;
    grid-template-columns:
        repeat(auto-fill, minmax(300px, 1fr));
    gap: 24px;
}}

.card {{
    background: white;
    border-radius: 10px;
    overflow: hidden;
    position: relative;
    box-shadow:
        0 2px 8px rgba(0,0,0,.12);
}}

.card img {{
    display: block;
    width: 100%;
    height: 280px;
    object-fit: contain;
    background: #111;
}}

.content {{
    padding: 16px;
}}

.rank {{
    position: absolute;
    top: 10px;
    left: 10px;
    background: black;
    color: white;
    padding: 6px 9px;
    border-radius: 6px;
    font-weight: bold;
}}

.score {{
    font-weight: bold;
}}

.path {{
    font-size: 11px;
    color: #666;
    word-break: break-all;
}}

</style>
</head>

<body>

<h1>VisualMind Semantic Search</h1>

<div class="subtitle">
<strong>Query:</strong>
{html.escape(args.query)}

<br>

<strong>Model:</strong>
{html.escape(model_id)}

<br>

<strong>Revision:</strong>
{html.escape(revision)}
</div>

<div class="grid">
{''.join(cards)}
</div>

</body>
</html>
"""

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_ROOT
        / f"{safe_name(args.query)}.html"
    )

    output_path.write_text(
        page,
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("VISUALMIND SEARCH GALLERY CREATED")
    print("=" * 72)
    print(f"Query:   {args.query}")
    print(f"Results: {top_k}")
    print(f"Gallery: {output_path.resolve()}")
    print("=" * 72)
    print()

    print(
        "Open automatically with:"
    )

    print(
        f'Start-Process "{output_path.resolve()}"'
    )


if __name__ == "__main__":
    main()
