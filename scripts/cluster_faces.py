"""Cluster face embeddings into anonymous person identities.

DBSCAN over cosine distance. k-means is wrong here: the number of people
in the corpus is unknown, and every face would be forced into a cluster.
DBSCAN infers the count and leaves genuinely ambiguous faces unassigned
rather than guessing.

Produces Person_001..Person_NNN and an HTML contact sheet of cropped
faces per cluster, so clusters can be labelled by looking rather than by
reading filenames.

Expect over-splitting: the same person at 5 and at 30 years old will
often land in separate clusters. That is the model working as designed -
merging is a labelling decision, not a clustering one.

Writes data/metadata/face_clusters.csv and outputs/faces/clusters.html.
Both describe identifiable people. Gitignored, blocked by the hook.
"""
import argparse
import base64
import csv
import html
import io
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN

from visualmind.text import counted, plural

INDEX_DIR = Path("indexes")
EMBEDDINGS_PATH = INDEX_DIR / "face_embeddings.npy"
LOOKUP_PATH = INDEX_DIR / "face_lookup.csv"
OUTPUT_CSV = Path("data/metadata/face_clusters.csv")
OUTPUT_HTML = Path("outputs/faces/clusters.html")

EPS = 0.45
MIN_SAMPLES = 3
THUMB = 96
SHEET_LIMIT = 24

PAGE_CSS = """
body { font-family: system-ui, sans-serif; margin: 28px; background: #f4f4f4;
       color: #222; }
h1 { margin-bottom: 2px; }
.sub { color: #555; font-size: 14px; margin-bottom: 24px; }
.cluster { background: #fff; border-radius: 10px; padding: 16px 18px 18px;
           margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,.10); }
.cluster h2 { margin: 0 0 4px; font-size: 17px; }
.cluster .stats { color: #666; font-size: 13px; margin-bottom: 12px; }
.faces { display: flex; flex-wrap: wrap; gap: 8px; }
.faces img { width: 96px; height: 96px; object-fit: cover; border-radius: 6px;
             background: #111; }
.noise { background: #fdf6e3; }
"""


def read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def crop_data_uri(row, cache):
    path = row["source_path"]

    if path not in cache:
        cache[path] = cv2.imread(path)

    image = cache[path]

    if image is None:
        return ""

    h, w = image.shape[:2]
    x1 = max(0, int(row["x1"]))
    y1 = max(0, int(row["y1"]))
    x2 = min(w, int(row["x2"]))
    y2 = min(h, int(row["y2"]))

    if x2 <= x1 or y2 <= y1:
        return ""

    crop = image[y1:y2, x1:x2]
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    pil = Image.fromarray(crop)
    pil.thumbnail((THUMB * 2, THUMB * 2))

    buffer = io.BytesIO()
    pil.save(buffer, format="JPEG", quality=80)

    return "data:image/jpeg;base64," + base64.b64encode(
        buffer.getvalue()).decode("ascii")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eps", type=float, default=EPS,
                        help="DBSCAN cosine distance. Lower splits more.")
    parser.add_argument("--min-samples", type=int, default=MIN_SAMPLES)
    parser.add_argument("--sheet-limit", type=int, default=SHEET_LIMIT,
                        help="Max face crops shown per cluster.")
    parser.add_argument("--no-html", action="store_true")
    args = parser.parse_args()

    rows = read_csv(LOOKUP_PATH)
    vectors = np.load(EMBEDDINGS_PATH)

    if len(rows) != len(vectors):
        raise RuntimeError("Face lookup and embedding counts differ.")

    print()
    print("=" * 76)
    print("VISUALMIND - FACE CLUSTERING")
    print("=" * 76)
    print("Faces:        " + str(len(rows)))
    print("eps:          " + str(args.eps))
    print("min_samples:  " + str(args.min_samples))

    labels = DBSCAN(
        eps=args.eps,
        min_samples=args.min_samples,
        metric="cosine",
    ).fit_predict(vectors)

    counts = Counter(labels)
    noise = counts.get(-1, 0)
    clusters = sorted(c for c in counts if c != -1)

    # Rename by size: Person_001 is the most photographed.
    by_size = sorted(clusters, key=lambda c: counts[c], reverse=True)
    name_of = {
        c: "Person_" + str(i).zfill(3)
        for i, c in enumerate(by_size, start=1)
    }
    name_of[-1] = "unassigned"

    print()
    print("Clusters found:   " + str(len(clusters)))
    print("Unassigned faces: " + str(noise)
          + "  (" + format(100 * noise / len(rows), ".1f") + "%)")

    members = defaultdict(list)

    for row, label in zip(rows, labels):
        members[label].append(row)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "person", "face_id", "source_path", "filename",
                "x1", "y1", "x2", "y2", "det_score", "sex", "age",
            ],
        )
        writer.writeheader()

        for label in by_size + [-1]:
            for row in members.get(label, []):
                writer.writerow({
                    "person": name_of[label],
                    "face_id": row["face_id"],
                    "source_path": row["source_path"],
                    "filename": row["filename"],
                    "x1": row["x1"], "y1": row["y1"],
                    "x2": row["x2"], "y2": row["y2"],
                    "det_score": row["det_score"],
                    "sex": row["sex"], "age": row["age"],
                })

    print()
    print("-" * 76)
    print("LARGEST CLUSTERS")
    print("-" * 76)

    for label in by_size[:15]:
        group = members[label]
        images = len({r["source_path"] for r in group})
        ages = [int(r["age"]) for r in group if r["age"].isdigit()]
        sexes = Counter(r["sex"] for r in group)
        dominant = sexes.most_common(1)[0][0] if sexes else "?"

        age_note = ""

        if ages:
            age_note = ("  age " + str(min(ages)) + "-" + str(max(ages))
                        + " (median " + str(int(np.median(ages))) + ")")

        print(name_of[label].ljust(14) + str(len(group)).rjust(4)
              + " " + plural(len(group), "face") + " in "
              + str(images).rjust(3) + " " + plural(images, "image")
              + "  " + dominant + age_note)

    if not args.no_html:
        print()
        print("Building contact sheet...")

        cache = {}
        blocks = []

        for label in by_size + [-1]:
            group = members.get(label, [])

            if not group:
                continue

            group = sorted(
                group,
                key=lambda r: float(r["det_score"]),
                reverse=True,
            )

            images = len({r["source_path"] for r in group})
            shown = group[:args.sheet_limit]

            crops = "".join(
                '<img src="' + crop_data_uri(r, cache) + '" alt="face">'
                for r in shown
            )

            extra = ""

            if len(group) > len(shown):
                extra = (" - showing " + str(len(shown))
                         + " highest-confidence")

            css = "cluster noise" if label == -1 else "cluster"

            blocks.append(
                '<div class="' + css + '">'
                + "<h2>" + html.escape(name_of[label]) + "</h2>"
                + '<div class="stats">' + counted(len(group), "face")
                + " in " + counted(images, "image") + extra + "</div>"
                + '<div class="faces">' + crops + "</div></div>"
            )

            # Release decoded images periodically; full-size frames are large.
            if len(cache) > 60:
                cache.clear()

        page = (
            '<!DOCTYPE html><html><head><meta charset="utf-8">'
            + "<title>VisualMind - face clusters</title>"
            + "<style>" + PAGE_CSS + "</style></head><body>"
            + "<h1>Face clusters</h1>"
            + '<div class="sub">' + str(len(clusters)) + " clusters from "
            + str(len(rows)) + " faces, eps=" + str(args.eps)
            + ", min_samples=" + str(args.min_samples)
            + ". Unassigned faces are shown last.</div>"
            + "".join(blocks)
            + "</body></html>"
        )

        OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_HTML.write_text(page, encoding="utf-8")

        print("Contact sheet: " + str(OUTPUT_HTML.resolve()))

    print()
    print("Clusters: " + str(OUTPUT_CSV.resolve()))
    print()
    print("These files identify real people. Gitignored and hook-blocked.")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
