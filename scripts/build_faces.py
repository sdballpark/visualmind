"""Detect faces across the catalog and store their embeddings.

Uses InsightFace buffalo_l: SCRFD for detection, ArcFace for 512-d
recognition embeddings. Detection and embedding happen here; clustering
is a separate step so it can be re-run with different parameters without
redoing this pass.

Writes indexes/face_embeddings.npy, indexes/face_lookup.csv, and
indexes/face_scanned.csv. The last records every image the scan looked
at, including the ones with no faces - without it, "no faces found" and
"never scanned" are indistinguishable, and a staleness check cannot tell
a complete index from a partial one.

Face data is biometric and identifies real people. All three files are
gitignored and blocked by the pre-commit hook.

Resumable: images already scanned are skipped.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

CATALOG = Path("data/metadata/image_catalog.csv")
INDEX_DIR = Path("indexes")
EMBEDDINGS_PATH = INDEX_DIR / "face_embeddings.npy"
LOOKUP_PATH = INDEX_DIR / "face_lookup.csv"
SCANNED_PATH = INDEX_DIR / "face_scanned.csv"

MIN_DET_SCORE = 0.60
MIN_FACE_PIXELS = 40

FIELDNAMES = [
    "face_id", "source_path", "filename",
    "x1", "y1", "x2", "y2", "width", "height",
    "det_score", "sex", "age",
]

SCANNED_FIELDNAMES = ["source_path", "filename", "faces", "status"]


def read_csv(path):
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min-score", type=float, default=MIN_DET_SCORE)
    parser.add_argument("--min-pixels", type=int, default=MIN_FACE_PIXELS)
    parser.add_argument("--det-size", type=int, default=640)
    args = parser.parse_args()

    from insightface.app import FaceAnalysis

    rows = read_csv(CATALOG)

    existing_rows = read_csv(LOOKUP_PATH)
    scanned_rows = read_csv(SCANNED_PATH)

    existing_vectors = None

    if existing_rows and EMBEDDINGS_PATH.exists():
        existing_vectors = np.load(EMBEDDINGS_PATH)

    done = {r["source_path"] for r in scanned_rows}

    if not scanned_rows and existing_rows:
        done = {r["source_path"] for r in existing_rows}

    todo = [r for r in rows if r["source_path"] not in done]

    if args.limit:
        todo = todo[:args.limit]

    print()
    print("=" * 76)
    print("VISUALMIND - FACE DETECTION AND EMBEDDING")
    print("=" * 76)
    print("Catalog images:   " + str(len(rows)))
    print("Already scanned:  " + str(len(done)))
    print("To scan:          " + str(len(todo)))
    print("Min det score:    " + str(args.min_score))
    print("Min face pixels:  " + str(args.min_pixels))

    if not todo:
        print("\nNothing to do.")
        return 0

    print("\nLoading buffalo_l...")
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(args.det_size, args.det_size))

    new_rows = []
    new_vectors = []
    new_scanned = []

    next_id = len(existing_rows)

    unreadable = 0
    rejected = 0
    started = time.time()

    for number, row in enumerate(todo, start=1):
        path = row["source_path"]
        image = cv2.imread(path)

        if image is None:
            unreadable += 1
            new_scanned.append({
                "source_path": path,
                "filename": row["filename"],
                "faces": "0",
                "status": "unreadable",
            })
            continue

        found = 0

        for face in app.get(image):
            if face.det_score < args.min_score:
                rejected += 1
                continue

            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            width = x2 - x1
            height = y2 - y1

            if width < args.min_pixels or height < args.min_pixels:
                rejected += 1
                continue

            vector = face.embedding.astype(np.float32)
            norm = np.linalg.norm(vector)

            if norm == 0:
                rejected += 1
                continue

            new_vectors.append(vector / norm)

            new_rows.append({
                "face_id": str(next_id),
                "source_path": path,
                "filename": row["filename"],
                "x1": str(x1), "y1": str(y1),
                "x2": str(x2), "y2": str(y2),
                "width": str(width), "height": str(height),
                "det_score": format(float(face.det_score), ".4f"),
                "sex": str(face.sex),
                "age": str(face.age),
            })

            next_id += 1
            found += 1

        new_scanned.append({
            "source_path": path,
            "filename": row["filename"],
            "faces": str(found),
            "status": "ok",
        })

        if number % 50 == 0 or number == len(todo):
            rate = (time.time() - started) / number
            left = (len(todo) - number) * rate
            print("  " + str(number) + "/" + str(len(todo))
                  + "  " + str(len(new_rows)) + " faces"
                  + "  " + format(rate, ".2f") + "s/image"
                  + "  ~" + format(left / 60, ".0f") + " min left")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    all_scanned = scanned_rows + new_scanned

    with SCANNED_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCANNED_FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_scanned)

    if new_rows:
        matrix = np.vstack(new_vectors).astype(np.float32)

        if existing_vectors is not None:
            matrix = np.vstack([existing_vectors, matrix])

        all_rows = existing_rows + new_rows

        np.save(EMBEDDINGS_PATH, matrix)

        with LOOKUP_PATH.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(all_rows)

        shape = str(matrix.shape)
    else:
        all_rows = existing_rows
        shape = "unchanged"

    with_faces = len({r["source_path"] for r in all_rows})
    elapsed = time.time() - started

    print()
    print("-" * 76)
    print("FACE INDEX SUMMARY")
    print("-" * 76)
    print("Faces this run:      " + str(len(new_rows)))
    print("Faces total:         " + str(len(all_rows)))
    print("Images scanned:      " + str(len(all_scanned)) + " of "
          + str(len(rows)))
    print("Images with faces:   " + str(with_faces))
    print("Images with none:    " + str(len(all_scanned) - with_faces))
    print("Rejected (low score or too small): " + str(rejected))
    print("Unreadable images:   " + str(unreadable))
    print("Embedding matrix:    " + shape)
    print("Elapsed:             " + format(elapsed / 60, ".1f") + " min")
    print()
    print("Embeddings:  " + str(EMBEDDINGS_PATH.resolve()))
    print("Lookup:      " + str(LOOKUP_PATH.resolve()))
    print("Scan record: " + str(SCANNED_PATH.resolve()))
    print()
    print("Biometric data. Gitignored and blocked by the pre-commit hook.")
    print()
    print("=" * 76)
    print("FACE INDEX COMPLETE")
    print("=" * 76)

    return 0


if __name__ == "__main__":
    sys.exit(main())
