#!/usr/bin/env python3
"""Build the synthetic corpus the README screenshots are taken from.

Every photograph in the real archive is private, and a screenshot leaks
more than faces: 417 of the 441 captions describe people, the item page
sets that caption at 21px, the people links carry real names, and the
diagnostics panel carries a filename and a sha. There is no safe subset
of the real corpus large enough to fill a grid, so the screenshots use
this instead - drawn images, captions written to describe what was
drawn, and invented people and events.

The captions have to match the pictures. A screenshot of "Every caption
here mentions the query term" over images with no dog in them would be
the same quiet lie the interface is built to avoid, so the dog scenes
have dogs in them.

This writes only the parts a corpus cannot derive: the images, the
catalog, the captions, the people and the events. Everything else comes
from the real builders, run against this root. Every path in the code is
relative, so pointing the pipeline and the app at a different corpus is
a matter of where they are run from, not a flag:

    uv run python scripts/build_demo_corpus.py /tmp/demo
    cp -r configs /tmp/demo/

    cd /tmp/demo
    for stage in build_thumbnails build_palette build_embeddings \
                 build_caption_embeddings build_visual_embeddings \
                 find_duplicates; do
        PYTHONPATH=<repo>/src python <repo>/scripts/$stage.py
    done

    PYTHONPATH=<repo>/src python <repo>/scripts/serve.py

Then run the frontend dev server against it and screenshot. Confirm the
API is serving this corpus and not the archive before capturing
anything - /palette reports 60 marks here and 441 there.

Two mistakes are worth not repeating, because both reached a screenshot
before they were caught. Capture times must use EXIF's own separators;
api.captured_at parses "%Y:%m:%d %H:%M:%S", and dashes silently put
every image in the undated segment, which flattens the strip into an
even comb. And captions must vary: one caption repeated across eight
dogs gives eight identical embeddings, and a similarity curve made of
flat runs has no edge for the gradient to cut on - every fallback query
returned the ceiling.
"""
import argparse
import csv
import hashlib
import json
import math
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

from PIL import Image, ImageDraw

SEED = 20260901

# Warm, flat, poster-like. Deliberately not photographic: these are
# drawings and should read as drawings.
SKY = [(176, 205, 224), (198, 216, 228), (162, 193, 214)]
GRASS = [(138, 163, 106), (120, 148, 94), (152, 172, 118)]
SAND = [(224, 205, 172), (214, 192, 158)]
SEA = [(122, 160, 172), (104, 145, 162)]
WARM = [(196, 118, 84), (208, 146, 92), (176, 96, 74)]
DEEP = [(74, 88, 96), (92, 104, 108), (58, 70, 78)]
CREAM = (240, 233, 220)

SHAPES = [(4, 3), (3, 4), (16, 9), (1, 1), (3, 2), (2, 3)]

# Named, so a caption can say the colour that was actually drawn. One
# caption repeated across every dog left the caption embeddings in flat
# staircases, and a similarity curve with no edge in it never finds a
# gradient cut - every fallback query returned the ceiling.
NAMED_WARM = [("rust", (196, 118, 84)), ("amber", (208, 146, 92)),
              ("brick", (176, 96, 74)), ("ochre", (198, 152, 84))]
NAMED_SKY = [("pale blue", (176, 205, 224)), ("overcast", (198, 216, 228)),
             ("clear blue", (162, 193, 214))]
NAMED_GRASS = [("green", (138, 163, 106)), ("deep green", (120, 148, 94)),
               ("sunlit", (152, 172, 118))]


def pick(named, r):
    return named[r.randrange(len(named))]


def a(word):
    """The right indefinite article. "A amber dog" would ship in a
    screenshot, and the captions are the thing being read."""
    return ("an " if word[0] in "aeiou" else "a ") + word


def rnd(seq, r):
    return seq[r.randrange(len(seq))]


# ---------------------------------------------------------------- scenes


def scene_dog(d, w, h, r):
    d.rectangle([0, 0, w, int(h * 0.62)], fill=rnd(SKY, r))
    d.rectangle([0, int(h * 0.62), w, h], fill=rnd(GRASS, r))
    cx, cy = w * 0.5, h * 0.66
    s = min(w, h) * 0.20
    coat, body = pick(NAMED_WARM, r)
    sky_name, _ = pick(NAMED_SKY, r)
    grass_name, _ = pick(NAMED_GRASS, r)
    d.ellipse([cx - s * 1.4, cy - s * 0.6, cx + s * 0.9, cy + s * 0.7], fill=body)
    d.ellipse([cx + s * 0.5, cy - s * 1.5, cx + s * 1.8, cy - s * 0.2], fill=body)
    # A drooping ear, not a point: the pointed version read as a party hat.
    darker = tuple(max(0, c - 26) for c in body)
    d.ellipse([cx + s * 0.52, cy - s * 1.45, cx + s * 0.96, cy - s * 0.45],
              fill=darker)
    # Muzzle, so the head reads as facing the viewer.
    d.ellipse([cx + s * 1.28, cy - s * 0.92, cx + s * 1.92, cy - s * 0.34],
              fill=(232, 214, 190))
    d.ellipse([cx + s * 1.68, cy - s * 0.80, cx + s * 1.88, cy - s * 0.62],
              fill=(52, 46, 46))
    for i in range(4):
        x = cx - s * 1.2 + i * s * 0.62
        d.rectangle([x, cy + s * 0.3, x + s * 0.26, cy + s * 1.15], fill=darker)
    d.line([(cx - s * 1.4, cy - s * 0.3), (cx - s * 2.1, cy - s * 1.0)],
           fill=body, width=int(s * 0.22))
    d.ellipse([cx + s * 1.34, cy - s * 1.12, cx + s * 1.52, cy - s * 0.94],
              fill=(40, 40, 44))
    pose = r.choice(["stands", "waits", "sits"])
    return (a(coat).capitalize() + " dog " + pose + " on " + grass_name
            + " grass under " + a(sky_name) + " sky.")


def scene_beach(d, w, h, r):
    d.rectangle([0, 0, w, int(h * 0.45)], fill=rnd(SKY, r))
    d.rectangle([0, int(h * 0.45), w, int(h * 0.68)], fill=rnd(SEA, r))
    d.rectangle([0, int(h * 0.68), w, h], fill=rnd(SAND, r))
    sun = min(w, h) * 0.09
    d.ellipse([w * 0.72, h * 0.10, w * 0.72 + sun, h * 0.10 + sun],
              fill=(236, 214, 168))
    for i in range(5):
        y = h * (0.48 + i * 0.04)
        d.line([(w * 0.08 * i, y), (w * 0.08 * i + w * 0.16, y)],
               fill=CREAM, width=max(2, int(h * 0.006)))
    weather = r.choice(["a low sun", "a bright sun", "a hazy sun"])
    return ("A beach of pale sand with the sea behind it and " + weather
            + " over the water.")


def scene_cake(d, w, h, r):
    d.rectangle([0, 0, w, h], fill=CREAM)
    cx, cy = w * 0.5, h * 0.66
    bw, bh = min(w, h) * 0.44, min(w, h) * 0.20
    d.rectangle([cx - bw, cy, cx + bw, cy + bh], fill=rnd(WARM, r))
    d.rectangle([cx - bw * 0.72, cy - bh * 0.9, cx + bw * 0.72, cy],
                fill=(226, 196, 172))
    candles = r.choice([3, 4, 5, 6])
    for i in range(candles):
        x = cx - bw * 0.5 + i * bw * 0.25
        d.rectangle([x, cy - bh * 1.7, x + bw * 0.06, cy - bh * 0.9],
                    fill=(210, 214, 216))
        d.ellipse([x - bw * 0.02, cy - bh * 2.0, x + bw * 0.08, cy - bh * 1.6],
                  fill=(230, 176, 96))
    words = {3: "three", 4: "four", 5: "five", 6: "six"}
    return ("A birthday cake with " + words[candles]
            + " lit candles on a plain background.")


def scene_tree(d, w, h, r):
    d.rectangle([0, 0, w, h], fill=rnd(SKY, r))
    cx = w * 0.5
    base, top = h * 0.82, h * 0.16
    green = (96, 122, 88)
    for i in range(3):
        y0 = top + (base - top) * (i * 0.28)
        y1 = top + (base - top) * (0.42 + i * 0.28)
        spread = min(w, h) * (0.16 + i * 0.09)
        d.polygon([(cx, y0), (cx - spread, y1), (cx + spread, y1)], fill=green)
    d.rectangle([cx - w * 0.02, base, cx + w * 0.02, base + h * 0.07],
                fill=(122, 96, 74))
    baubles = r.choice([7, 9, 11])
    for _ in range(baubles):
        bx = cx + r.uniform(-0.18, 0.18) * w
        by = r.uniform(0.3, 0.78) * h
        s = min(w, h) * 0.022
        d.ellipse([bx, by, bx + s, by + s], fill=rnd(WARM, r))
    d.polygon([(cx, top - h * 0.05), (cx - w * 0.03, top + h * 0.01),
               (cx + w * 0.03, top + h * 0.01)], fill=(230, 200, 120))
    return ("A christmas tree decorated with " + str(baubles)
            + " baubles and a star on top.")


def scene_pool(d, w, h, r):
    d.rectangle([0, 0, w, h], fill=(214, 205, 190))
    d.rectangle([w * 0.08, h * 0.22, w * 0.92, h * 0.86], fill=(116, 166, 184))
    for i in range(4):
        y = h * (0.32 + i * 0.15)
        d.line([(w * 0.12, y), (w * 0.88, y)], fill=(150, 192, 204),
               width=max(2, int(h * 0.012)))
    return ("A rectangular swimming pool with " + str(4)
            + " lane markings, seen from above on a pale deck.")


def scene_car(d, w, h, r):
    d.rectangle([0, 0, w, int(h * 0.66)], fill=rnd(SKY, r))
    d.rectangle([0, int(h * 0.66), w, h], fill=(122, 118, 116))
    cx, cy = w * 0.5, h * 0.62
    s = min(w, h) * 0.24
    shade, red = r.choice([("red", (178, 74, 62)),
                           ("dark red", (152, 60, 52)),
                           ("bright red", (198, 88, 70))])
    d.rectangle([cx - s * 1.5, cy - s * 0.35, cx + s * 1.5, cy + s * 0.35],
                fill=red)
    d.polygon([(cx - s * 0.85, cy - s * 0.35), (cx - s * 0.5, cy - s * 0.95),
               (cx + s * 0.6, cy - s * 0.95), (cx + s * 0.9, cy - s * 0.35)],
              fill=red)
    for dx in (-0.85, 0.85):
        d.ellipse([cx + s * dx - s * 0.3, cy + s * 0.2,
                   cx + s * dx + s * 0.3, cy + s * 0.8], fill=(46, 46, 50))
    return ("A " + shade + " car parked on a grey road under an open sky.")


def scene_arch(d, w, h, r):
    d.rectangle([0, 0, w, int(h * 0.7)], fill=rnd(SKY, r))
    d.rectangle([0, int(h * 0.7), w, h], fill=rnd(GRASS, r))
    cx = w * 0.5
    aw, ay = min(w, h) * 0.30, h * 0.24
    d.arc([cx - aw, ay, cx + aw, ay + aw * 1.7], 180, 360,
          fill=(150, 160, 128), width=max(4, int(min(w, h) * 0.035)))
    for _ in range(14):
        t = r.uniform(0, math.pi)
        px = cx - aw * math.cos(t)
        py = ay + aw * 0.85 - aw * 0.85 * math.sin(t)
        s = min(w, h) * 0.026
        d.ellipse([px - s, py - s, px + s, py + s], fill=rnd(WARM, r))
    for dx in (-0.09, 0.09):
        fx = cx + w * dx
        d.ellipse([fx - w * 0.026, h * 0.60, fx + w * 0.026, h * 0.66],
                  fill=(206, 186, 162))
        d.polygon([(fx - w * 0.035, h * 0.86), (fx + w * 0.035, h * 0.86),
                   (fx + w * 0.022, h * 0.65), (fx - w * 0.022, h * 0.65)],
                  fill=CREAM if dx < 0 else (92, 104, 116))
    return ("Two figures standing under a flower arch on " 
            + pick(NAMED_GRASS, r)[0] + " grass.")


def scene_hills(d, w, h, r):
    d.rectangle([0, 0, w, h], fill=rnd(SKY, r))
    sun = min(w, h) * 0.10
    d.ellipse([w * 0.18, h * 0.14, w * 0.18 + sun, h * 0.14 + sun],
              fill=(232, 206, 158))
    for i, col in enumerate([(128, 142, 126), (104, 120, 110), (84, 98, 94)]):
        y = h * (0.52 + i * 0.13)
        d.polygon([(-w * 0.1 + i * w * 0.3, h), (w * 0.35 + i * w * 0.3, y),
                   (w * 0.8 + i * w * 0.3, h)], fill=col)
    d.rectangle([0, int(h * 0.9), w, h], fill=(96, 110, 100))
    return ("Rolling " + r.choice(["green", "grey-green", "hazy"])
            + " hills under a wide sky with a low sun.")


def scene_balloons(d, w, h, r):
    d.rectangle([0, 0, w, h], fill=CREAM)
    count = r.choice([5, 6, 7])
    for i in range(count):
        bx = w * (0.14 + i * 0.12) + r.uniform(-0.02, 0.02) * w
        by = h * r.uniform(0.16, 0.42)
        s = min(w, h) * r.uniform(0.07, 0.11)
        d.ellipse([bx - s, by - s * 1.2, bx + s, by + s * 1.2], fill=rnd(WARM, r))
        d.line([(bx, by + s * 1.2), (bx + r.uniform(-8, 8), h * 0.82)],
               fill=(150, 142, 130), width=2)
    return ("A cluster of " + str(count)
            + " party balloons on strings against a cream background.")


def scene_glasses(d, w, h, r):
    d.rectangle([0, 0, w, h], fill=rnd(SKY, r))
    cx, cy = w * 0.5, h * 0.52
    s = min(w, h) * 0.26
    d.ellipse([cx - s, cy - s * 1.15, cx + s, cy + s * 1.15],
              fill=(224, 198, 172))
    for dx in (-0.42, 0.42):
        d.ellipse([cx + s * dx - s * 0.34, cy - s * 0.42,
                   cx + s * dx + s * 0.34, cy + s * 0.16], fill=(56, 60, 68))
    d.line([(cx - s * 0.08, cy - s * 0.14), (cx + s * 0.08, cy - s * 0.14)],
           fill=(56, 60, 68), width=max(3, int(s * 0.1)))
    d.arc([cx - s * 0.4, cy + s * 0.3, cx + s * 0.4, cy + s * 0.8],
          0, 180, fill=(150, 104, 92), width=max(3, int(s * 0.08)))
    return ("A figure wearing dark sunglasses, drawn face-on against "
            + a(pick(NAMED_SKY, r)[0]) + " background.")


SCENES = [
    (scene_dog, 8), (scene_beach, 7), (scene_cake, 6), (scene_tree, 5),
    (scene_pool, 5), (scene_car, 6), (scene_arch, 5), (scene_hills, 6),
    (scene_balloons, 6), (scene_glasses, 6),
]

# Invented, and recognisably so. The test fixtures already use this
# register, so a reader who meets both sees the same convention.
PEOPLE = ["Ada Fixture", "Bo Sample", "Cy Placeholder", "Dev Example"]
EVENTS = [
    ("event-01", "Fixture Picnic"),
    ("event-02", "Sample Trip"),
    ("event-03", "Example Party"),
]


def occupied(root):
    """Whether `root` already holds a corpus this would overwrite.

    The catalog and the captions are the two files this script rewrites
    wholesale, and in the real tree they describe private photographs.
    Pointing this at the repository by accident would replace them with
    sixty drawings, and the originals are not reconstructable from
    anything else in the tree.
    """
    return [
        path for path in (
            root / "data" / "metadata" / "image_catalog.csv",
            root / "data" / "metadata" / "captions.csv",
        )
        if path.exists()
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        type=Path,
        help="Directory to write the corpus into. Not the repository.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite a corpus that is already there.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    source = root / "source"

    existing = occupied(root)

    if existing and not args.force:
        print()
        print("There is already a corpus at " + str(root) + ":")
        print()

        for path in existing:
            print("  " + str(path.relative_to(root)))

        print()
        print("This script rewrites both. If that is a real corpus, the")
        print("captions are not reconstructable from anything else here.")
        print("Choose an empty directory, or pass --force.")
        print()
        return 1

    r = random.Random(SEED)
    source.mkdir(parents=True, exist_ok=True)
    (root / "data" / "metadata").mkdir(parents=True, exist_ok=True)

    plan = []
    for fn, count in SCENES:
        plan.extend([fn] * count)
    r.shuffle(plan)

    catalog, captions, face_rows, event_rows = [], [], [], []
    labels = {name: [] for name in PEOPLE}
    start = datetime(2019, 4, 6, 11, 0, 0)
    face_id = 0

    for index, fn in enumerate(plan, start=1):
        ratio = rnd(SHAPES, r)
        long_edge = r.choice([1400, 1600, 1800])
        if ratio[0] >= ratio[1]:
            w = long_edge
            h = int(long_edge * ratio[1] / ratio[0])
        else:
            h = long_edge
            w = int(long_edge * ratio[0] / ratio[1])

        image = Image.new("RGB", (w, h), CREAM)
        caption = fn(ImageDraw.Draw(image), w, h, r)

        name = "demo-" + str(index).zfill(3) + ".png"
        path = source / name
        image.save(path)

        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()

        # Two thirds carry a capture time, so the strip shows both a
        # dated span and an undated segment, as the real one does.
        dated = index % 3 != 0
        when = start + timedelta(days=index * 11, hours=index % 7)
        # EXIF's own separators. api.captured_at parses "%Y:%m:%d
        # %H:%M:%S"; dashes fail that parse and every image lands in
        # the undated segment, which flattens the whole strip.
        captured = when.strftime("%Y:%m:%d %H:%M:%S") if dated else ""

        catalog.append({
            "image_id": sha, "source_path": str(path),
            "relative_path": name, "filename": name, "extension": ".png",
            "file_bytes": str(len(raw)), "sha256": sha,
            "provenance_status": "synthetic",
            "gmail_directory_year": "", "gmail_directory_month": "",
            "gmail_message_date": "", "gmail_from": "", "gmail_subject": "",
            "gmail_message_id": "", "gmail_thread_id": "",
            "original_filename": name, "width": str(w), "height": str(h),
            "image_format": "PNG", "image_mode": "RGB",
            "has_exif": "False", "exif_datetime_original": "",
            "exif_datetime_digitized": "", "exif_datetime_modified": "",
            "best_exif_date": captured, "camera_make": "",
            "camera_model": "", "orientation": "", "image_error": "",
        })
        captions.append({
            "source_path": str(path), "filename": name, "caption": caption,
            "model": "synthetic", "revision": "demo",
        })

        # People on roughly a third of the images, so the item page and
        # the person filter both have something to show.
        if index % 3 == 1:
            who = PEOPLE[index % len(PEOPLE)]
            face_id += 1
            fid = "f" + str(face_id).zfill(4)
            face_rows.append({
                "person": who, "face_id": fid, "source_path": str(path),
                "filename": name, "x1": "10", "y1": "10", "x2": "90",
                "y2": "90", "det_score": "0.90", "sex": "", "age": "",
            })
            labels[who].append(fid)

            if index % 6 == 1:
                second = PEOPLE[(index + 1) % len(PEOPLE)]
                face_id += 1
                fid2 = "f" + str(face_id).zfill(4)
                face_rows.append({
                    "person": second, "face_id": fid2,
                    "source_path": str(path), "filename": name,
                    "x1": "110", "y1": "10", "x2": "190", "y2": "90",
                    "det_score": "0.88", "sex": "", "age": "",
                })
                labels[second].append(fid2)

        if dated:
            eid, ename = EVENTS[index % len(EVENTS)]
            event_rows.append({
                "event_id": eid, "event_name": ename,
                "event_start": captured, "event_end": captured,
                "image_count": "0", "date_source": "synthetic",
                "source_path": str(path), "filename": name,
                "capture_time": captured, "gmail_subject": "",
                "gmail_thread_id": "",
            })
        else:
            event_rows.append({
                # The real corpus names this bucket rather than leaving
                # it blank, and the item page renders the name.
                "event_id": "unassigned", "event_name": "unassigned",
                "event_start": "", "event_end": "", "image_count": "0",
                "date_source": "none", "source_path": str(path),
                "filename": name, "capture_time": "",
                "gmail_subject": "", "gmail_thread_id": "",
            })

    counts = {}
    for row in event_rows:
        counts[row["event_id"]] = counts.get(row["event_id"], 0) + 1
    for row in event_rows:
        row["image_count"] = str(counts[row["event_id"]])

    def write(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    meta = root / "data" / "metadata"
    write(meta / "image_catalog.csv", catalog)
    write(meta / "captions.csv", captions)
    write(meta / "face_clusters.csv", face_rows)
    write(meta / "events.csv", event_rows)
    (meta / "person_labels.json").write_text(
        json.dumps(labels, indent=2) + "\n", encoding="utf-8"
    )

    print("images   :", len(catalog))
    print("captions :", len(captions))
    print("faces    :", len(face_rows), "over", len(PEOPLE), "people")
    print("events   :", len({r["event_id"] for r in event_rows}))
    print("root     :", root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
