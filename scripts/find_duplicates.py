"""Find duplicate and near-duplicate images in three tiers.

  EXACT      identical SHA-256. Byte-for-byte the same file.
  NEAR       perceptual hash within a small Hamming distance. Resized,
             recompressed, or lightly edited copies of one photograph.
  SIMILAR    DINOv2 cosine similarity above a threshold. Different
             photographs of the same scene - burst frames, retakes.

This script reports only. It never moves, renames, or deletes anything.
Any action on its output is a separate, human-approved step.

Writes data/metadata/duplicate_groups.csv.
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image

CATALOG = Path("data/metadata/image_catalog.csv")
INDEX_DIR = Path("indexes")
EMBEDDINGS_PATH = INDEX_DIR / "dinov2_embeddings.npy"
LOOKUP_PATH = INDEX_DIR / "dinov2_lookup.csv"
OUTPUT = Path("data/metadata/duplicate_groups.csv")

PHASH_DISTANCE = 6
COSINE_THRESHOLD = 0.92


def read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def megapixels(row):
    try:
        return int(row.get("width") or 0) * int(row.get("height") or 0)
    except ValueError:
        return 0


def file_bytes(row):
    try:
        return int(row.get("bytes") or 0)
    except ValueError:
        return 0


def best_of(rows):
    """Pick a keeper: most pixels, then largest file, then earliest EXIF."""
    return sorted(
        rows,
        key=lambda r: (
            megapixels(r),
            file_bytes(r),
            r.get("best_exif_date") or "9999",
        ),
        reverse=True,
    )[0]


class Union:
    def __init__(self, items):
        self.parent = {item: item for item in items}

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def join(self, left, right):
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[b] = a

    def groups(self):
        out = defaultdict(list)
        for item in self.parent:
            out[self.find(item)].append(item)
        return [g for g in out.values() if len(g) > 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phash-distance", type=int, default=PHASH_DISTANCE)
    parser.add_argument("--cosine", type=float, default=COSINE_THRESHOLD)
    parser.add_argument(
        "--show",
        type=int,
        default=10,
        help="How many groups to print per tier.",
    )
    args = parser.parse_args()

    lookup = read_csv(LOOKUP_PATH)
    embeddings = np.load(EMBEDDINGS_PATH)

    if len(lookup) != len(embeddings):
        raise RuntimeError("Embedding and lookup counts differ.")

    by_path = {row["source_path"]: row for row in lookup}
    paths = [row["source_path"] for row in lookup]
    index_of = {path: i for i, path in enumerate(paths)}

    print()
    print("=" * 76)
    print("VISUALMIND - DUPLICATE DETECTION")
    print("=" * 76)
    print("Images:           " + str(len(paths)))
    print("pHash distance:   <= " + str(args.phash_distance))
    print("Cosine threshold: >= " + str(args.cosine))

    # Tier 1 - exact SHA-256
    by_sha = defaultdict(list)

    for row in lookup:
        sha = (row.get("sha256") or "").strip().lower()
        if sha:
            by_sha[sha].append(row["source_path"])

    exact_groups = [g for g in by_sha.values() if len(g) > 1]
    exact_members = {p for g in exact_groups for p in g}

    print("\nComputing perceptual hashes...")

    hashes = {}

    for number, path in enumerate(paths, start=1):
        try:
            with Image.open(path) as image:
                hashes[path] = imagehash.phash(image.convert("RGB"))
        except Exception:
            continue

        if number % 100 == 0:
            print("  " + str(number) + "/" + str(len(paths)))

    # Tier 2 - perceptual hash within distance
    near = Union(list(hashes))
    hashed = list(hashes)

    for i in range(len(hashed)):
        for j in range(i + 1, len(hashed)):
            left, right = hashed[i], hashed[j]

            if hashes[left] - hashes[right] <= args.phash_distance:
                near.join(left, right)

    near_groups = [
        g for g in near.groups()
        if not set(g) <= exact_members
    ]
    near_members = {p for g in near_groups for p in g} | exact_members

    # Tier 3 - DINOv2 cosine
    sims = embeddings @ embeddings.T
    np.fill_diagonal(sims, 0.0)

    similar = Union(paths)
    pairs = np.argwhere(sims >= args.cosine)

    for i, j in pairs:
        if i < j:
            similar.join(paths[i], paths[j])

    similar_groups = [
        g for g in similar.groups()
        if not set(g) <= near_members
    ]

    print()
    print("-" * 76)
    print("SUMMARY")
    print("-" * 76)
    print("EXACT groups (identical SHA-256):     " + str(len(exact_groups)))
    print("NEAR groups (perceptual hash):        " + str(len(near_groups)))
    print("SIMILAR groups (DINOv2 cosine):       "
          + str(len(similar_groups)))

    redundant = sum(len(g) - 1 for g in exact_groups + near_groups)
    print("Redundant files in EXACT/NEAR:        " + str(redundant))

    rows_out = []

    for tier, groups in (
        ("EXACT", exact_groups),
        ("NEAR", near_groups),
        ("SIMILAR", similar_groups),
    ):
        for number, group in enumerate(groups, start=1):
            members = [by_path[p] for p in group]
            keeper = best_of(members)

            for row in members:
                rows_out.append({
                    "tier": tier,
                    "group": tier.lower() + "-" + str(number),
                    "keep": "1" if row is keeper else "0",
                    "filename": row["filename"],
                    "width": row.get("width", ""),
                    "height": row.get("height", ""),
                    "bytes": row.get("bytes", ""),
                    "best_exif_date": row.get("best_exif_date", ""),
                    "source_path": row["source_path"],
                })

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "tier", "group", "keep", "filename",
                "width", "height", "bytes", "best_exif_date",
                "source_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows_out)

    for tier, groups in (
        ("EXACT", exact_groups),
        ("NEAR", near_groups),
        ("SIMILAR", similar_groups),
    ):
        if not groups:
            continue

        print()
        print("-" * 76)
        print(tier + " - showing " + str(min(args.show, len(groups)))
              + " of " + str(len(groups)))
        print("-" * 76)

        for group in groups[:args.show]:
            members = [by_path[p] for p in group]
            keeper = best_of(members)

            for row in members:
                mark = "KEEP" if row is keeper else "    "
                size = str(row.get("width", "?")) + "x" + str(
                    row.get("height", "?"))
                print("  " + mark + "  " + row["filename"][:44].ljust(46)
                      + size)

            print()

    print("Report: " + str(OUTPUT.resolve()))
    print()
    print("This script reports only. Nothing was moved or deleted.")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
