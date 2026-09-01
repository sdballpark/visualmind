#!/usr/bin/env python3
"""Derive a dominant hue and a lightness per image from grid thumbnails.

Reads thumbnails/manifest.csv and writes thumbnails/palette.csv. Source
images are never opened and no model is loaded: the grid thumbnail
already is the downsample, so this is one cheap pass over files that
exist.

The UI renders the whole corpus as a density strip above the photo grid,
one mark per image coloured from its photo. A mean colour cannot do
that. Averaging a family photo gives brown, and averaging 441 of them
gives 441 near-identical browns - the strip would be a flat band. So the
hue comes from the dominant colour rather than the average: sample the
thumbnail, throw away the pixels that carry no hue at all, and take the
most common hue among what is left.

Lightness is measured separately, over every pixel, and is not part of
the hue decision. The strip draws hue as colour and lightness as mark
height, so a dark 2003 flash photo and a bright 2024 beach shot stay
distinguishable even when both are warm.

Hue is stored as real degrees. Quantising it to a constrained palette is
an art-direction decision and belongs in the frontend, where it can be
changed without regenerating anything.

Resumable: a sha already in the palette is skipped. An image whose
thumbnail cannot be read records a status row rather than stopping the
run, the same way build_thumbnails.py does.

Exit code is 1 if any thumbnail was unreadable.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

THUMBNAILS = Path("thumbnails")
MANIFEST = THUMBNAILS / "manifest.csv"
GRID = THUMBNAILS / "grid"
PALETTE = THUMBNAILS / "palette.csv"

FIELDNAMES = ["sha256", "hue", "lightness", "status"]

# The thumbnail is already small; this is about how many pixels the
# histogram needs, not about decode cost.
SAMPLE = 64

# Below these, a pixel carries no usable hue. A near-grey pixel has a
# hue but it is numerical noise, and a near-black one is worse: tiny
# channel differences swing it across the whole circle.
MIN_SATURATION = 0.15
MIN_VALUE = 0.15

# Coarse buckets to find where the mass is. The stored hue is then the
# circular mean of the pixels in that bucket, so the bucket width never
# reaches the output.
BINS = 36

RULE = "=" * 76
THIN = "-" * 76


def rgb_to_hsv(rgb):
    """Hue in degrees, saturation and value in 0-1, from float RGB.

    Done here rather than through PIL's HSV mode, which rounds hue to
    256 steps. The output is meant to be a real angle.
    """
    highest = rgb.max(axis=-1)
    lowest = rgb.min(axis=-1)
    spread = highest - lowest

    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    safe = np.where(spread == 0, 1.0, spread)

    hue = np.select(
        [highest == red, highest == green, highest == blue],
        [
            ((green - blue) / safe) % 6.0,
            ((blue - red) / safe) + 2.0,
            ((red - green) / safe) + 4.0,
        ],
        default=0.0,
    ) * 60.0

    hue = np.where(spread == 0, 0.0, hue % 360.0)
    saturation = np.where(
        highest == 0, 0.0, spread / np.where(highest == 0, 1.0, highest)
    )

    return hue, saturation, highest


def perceptual_lightness(rgb):
    """Mean CIE L* over every pixel, scaled to 0-1.

    Relative luminance alone clusters most photographs near the bottom
    of the range, which would make the strip's mark heights nearly
    uniform. L* is the perceptual scale, so the heights spread the way
    an eye expects.
    """
    linear = np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )

    luminance = float(
        (linear * np.array([0.2126, 0.7152, 0.0722])).sum(axis=-1).mean()
    )

    if luminance <= 0.008856:
        lightness = 903.3 * luminance
    else:
        lightness = 116.0 * luminance ** (1.0 / 3.0) - 16.0

    return max(0.0, min(1.0, lightness / 100.0))


def circular_mean(degrees):
    """Mean of angles, so hues either side of 0 do not average to cyan."""
    radians = np.radians(degrees)

    return float(
        np.degrees(
            np.arctan2(np.sin(radians).mean(), np.cos(radians).mean())
        ) % 360.0
    )


def dominant_hue(hue, saturation, value):
    """The most common hue among pixels that carry one, or None.

    None means the image is grey, black or white throughout - a real
    answer, not a failure. The frontend renders those marks neutral.
    """
    carries_hue = (saturation >= MIN_SATURATION) & (value >= MIN_VALUE)
    hues = hue[carries_hue]

    if hues.size == 0:
        return None

    counts, _ = np.histogram(hues, bins=BINS, range=(0.0, 360.0))
    centre = (int(counts.argmax()) + 0.5) * (360.0 / BINS)

    # Circular distance, so a peak sitting on 0 collects both sides.
    offset = np.abs((hues - centre + 180.0) % 360.0 - 180.0)

    return circular_mean(hues[offset <= (360.0 / BINS)])


def extract(image, sample=SAMPLE):
    """Return (hue in degrees or None, lightness in 0-1)."""
    small = image.convert("RGB")
    small.thumbnail((sample, sample), Image.LANCZOS)

    rgb = np.asarray(small, dtype=np.float64) / 255.0
    hue, saturation, value = rgb_to_hsv(rgb)

    return dominant_hue(hue, saturation, value), perceptual_lightness(rgb)


def read_rows(path):
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def row_for(sha, hue, lightness, status):
    return {
        "sha256": sha,
        "hue": "" if hue is None else format(hue, ".3f"),
        "lightness": "" if lightness is None else format(lightness, ".5f"),
        "status": status,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract every image, ignoring the existing palette.",
    )
    args = parser.parse_args()

    manifest = read_rows(MANIFEST)

    if not manifest:
        print("No thumbnail manifest at " + str(MANIFEST)
              + ". Run build_thumbnails.py.")
        return 1

    recorded = {} if args.force else {
        row["sha256"]: row for row in read_rows(PALETTE)
    }

    print()
    print(RULE)
    print("VISUALMIND - BUILD PALETTE")
    print(RULE)
    print("Thumbnails:      " + str(len(manifest)))
    print("Already built:   " + str(len(recorded)))
    print("Sample:          " + str(SAMPLE) + "px, "
          + str(BINS) + " hue buckets")
    print()

    rows = []
    extracted = 0
    skipped = 0
    failures = 0
    achromatic = 0

    for entry in manifest:
        sha = entry["sha256"]

        if sha in recorded:
            skipped += 1
            rows.append(recorded[sha])
            continue

        source = GRID / (sha + ".jpg")

        try:
            with Image.open(source) as image:
                hue, lightness = extract(image)
        except Exception as error:
            print("  SKIP " + sha[:12] + ": " + str(error))
            failures += 1
            rows.append(row_for(sha, None, None, "unreadable"))
            continue

        if hue is None:
            achromatic += 1

        extracted += 1
        rows.append(row_for(sha, hue, lightness, "ok"))

        if extracted % 100 == 0:
            print("  " + str(extracted) + "/" + str(len(manifest)))

    PALETTE.parent.mkdir(parents=True, exist_ok=True)

    with PALETTE.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(THIN)
    print("PALETTE SUMMARY")
    print(THIN)
    print("Extracted:          " + str(extracted))
    print("Skipped (existing): " + str(skipped))
    print("No hue (grey):      " + str(achromatic))
    print("Unreadable:         " + str(failures))
    print("Palette rows:       " + str(len(rows)))
    print()
    print("Palette:  " + str(PALETTE.resolve()))
    print()
    print(RULE)
    print("PALETTE COMPLETE")
    print(RULE)

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
