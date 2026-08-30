"""Hybrid search with an HTML thumbnail gallery.

Fuses SigLIP2 image retrieval and BGE caption retrieval with Reciprocal
Rank Fusion, derives how many results to return, and writes a
self-contained HTML page with embedded thumbnails.

Output pages contain base64 copies of the source photos. outputs/ is
gitignored for that reason.
"""
import argparse
import base64
import csv
import html
import io
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer

MODEL_CONFIG = Path("configs/models.yaml")
INDEX_DIR = Path("indexes")
OUTPUT_ROOT = Path("outputs/search")

IMAGE_EMB = INDEX_DIR / "siglip2_image_embeddings.npy"
IMAGE_LOOKUP = INDEX_DIR / "siglip2_lookup.csv"
CAPTION_EMB = INDEX_DIR / "caption_embeddings.npy"
CAPTION_LOOKUP = INDEX_DIR / "caption_lookup.csv"

RRF_K = 60
THUMBNAIL = (420, 420)
STOPWORDS = {"a", "an", "the", "of", "in", "on", "at", "with", "and", "or"}


def load_config(role):
    config = yaml.safe_load(MODEL_CONFIG.read_text(encoding="utf-8"))
    entry = config["models"][role]
    return entry["repo_id"], entry["revision"]


def read_lookup(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def safe_name(text):
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "query"


def thumbnail_data_uri(path):
    try:
        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail(THUMBNAIL)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=82)
    except Exception:
        return ""

    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return "data:image/jpeg;base64," + encoded


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


PAGE_CSS = """
body { font-family: system-ui, sans-serif; margin: 30px; background: #f4f4f4;
       color: #222; }
h1 { margin-bottom: 4px; }
.meta { margin-bottom: 8px; color: #444; font-size: 14px; }
.basis { display: inline-block; padding: 4px 10px; border-radius: 4px;
         background: #222; color: #fff; font-size: 13px; margin: 10px 0 24px; }
.warn { background: #8a1f1f; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
        gap: 22px; }
.card { background: #fff; border-radius: 10px; overflow: hidden;
        position: relative; box-shadow: 0 2px 8px rgba(0,0,0,.12); }
.card img { display: block; width: 100%; height: 280px; object-fit: contain;
            background: #111; }
.content { padding: 14px 16px 18px; }
.rank { position: absolute; top: 10px; left: 10px; background: #000; color: #fff;
        padding: 5px 9px; border-radius: 6px; font-weight: 600; font-size: 13px; }
.exact { position: absolute; top: 10px; right: 10px; background: #1f6f3f;
         color: #fff; padding: 5px 9px; border-radius: 6px; font-size: 12px; }
.name { font-weight: 600; margin-bottom: 6px; word-break: break-all; }
.ranks { font-size: 12px; color: #666; margin-bottom: 10px; }
.caption { font-size: 13px; line-height: 1.45; color: #333; }
.exif { font-size: 12px; color: #666; margin-top: 10px; }
"""


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

    img_s, img_lookup = image_scores(args.query)
    cap_s, cap_lookup = caption_scores(args.query)

    img_paths = [
        img_lookup[i]["source_path"] for i in np.argsort(img_s)[::-1]
    ]
    cap_paths = [
        cap_lookup[i]["source_path"] for i in np.argsort(cap_s)[::-1]
    ]

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
        basis = "fixed count (--top-k " + str(args.top_k) + ")"
        warn = False
    elif exact:
        results = [item for item in ordered if item[0] in exact]
        basis = ("caption term match - "
                 + str(len(exact)) + " images in corpus")
        warn = False
    else:
        results = ordered[:max(img_cut, cap_cut)]
        basis = ("score gradient - no caption mentions these terms")
        warn = max(img_cut, cap_cut) <= 2

    by_path = {r["source_path"]: r for r in cap_lookup}
    img_rank = {p: i for i, p in enumerate(img_paths, start=1)}
    cap_rank = {p: i for i, p in enumerate(cap_paths, start=1)}

    cards = []

    for rank, (path, score) in enumerate(results, start=1):
        row = by_path.get(path, {})
        name = row.get("filename", Path(path).name)
        caption = row.get("caption", "")
        thumb = thumbnail_data_uri(Path(path))

        badge = ""

        if path in exact:
            badge = '<div class="exact">term match</div>'

        cards.append(
            '<article class="card">'
            + '<div class="rank">#' + str(rank) + '</div>'
            + badge
            + '<img src="' + thumb + '" alt="' + html.escape(name) + '">'
            + '<div class="content">'
            + '<div class="name">' + html.escape(name) + '</div>'
            + '<div class="ranks">rrf ' + format(score, ".5f")
            + ' &nbsp;|&nbsp; image #' + str(img_rank.get(path, "-"))
            + ' &nbsp;|&nbsp; caption #' + str(cap_rank.get(path, "-"))
            + '</div>'
            + '<div class="caption">' + html.escape(caption) + '</div>'
            + '</div></article>'
        )

    if not cards:
        cards.append('<p>No results.</p>')

    page = (
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        + '<title>VisualMind - ' + html.escape(args.query) + '</title>'
        + '<style>' + PAGE_CSS + '</style></head><body>'
        + '<h1>' + html.escape(args.query) + '</h1>'
        + '<div class="meta">mode: ' + args.mode
        + ' &nbsp;|&nbsp; RRF k=' + str(args.rrf_k)
        + ' &nbsp;|&nbsp; image gradient cut ' + str(img_cut)
        + ' &nbsp;|&nbsp; caption gradient cut ' + str(cap_cut)
        + '</div>'
        + '<div class="basis' + (' warn' if warn else '') + '">'
        + 'returning ' + str(len(results)) + ' - ' + html.escape(basis)
        + '</div>'
        + '<div class="grid">' + "".join(cards) + '</div>'
        + '</body></html>'
    )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_ROOT / (safe_name(args.query) + ".html")
    output_path.write_text(page, encoding="utf-8")

    print()
    print("=" * 72)
    print("VISUALMIND SEARCH GALLERY")
    print("=" * 72)
    print("Query:    " + args.query)
    print("Mode:     " + args.mode)
    print("Returned: " + str(len(results)) + "  (" + basis + ")")
    print("Gallery:  " + str(output_path.resolve()))
    print("=" * 72)
    print()
    print('Open with: explorer.exe "$(wslpath -w '
          + str(output_path) + ')"')

    return 0


if __name__ == "__main__":
    sys.exit(main())
