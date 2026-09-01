#!/usr/bin/env python3
"""Generate grid and lightbox thumbnails for every image in the catalog.

Writes thumbnails/grid/ and thumbnails/lightbox/ as JPEG, plus
thumbnails/manifest.csv. All of it is gitignored and regenerable.

Outputs are keyed by the catalog's sha256 rather than the filename.
Filenames are not unique in this corpus - the duplicate-detection work
established that - so a filename key would silently overwrite.

Aspect ratio is preserved and nothing is cropped, because the UI uses a
justified grid that needs true ratios. Images smaller than a target are
left at their own size rather than enlarged.

EXIF orientation is applied at generation time. A large share of this
corpus is scans and 19 catalog rows carry orientation 6, which a viewer
reading the raw pixels would show on its side.

JPEG sources are drafted before decoding, so the decoder downscales
during the read. The corpus contains 45 MP images, and decoding one in
full to produce a 400px thumbnail is most of the cost of the run.

Resumable: a sha with both outputs on disk is skipped. An unreadable
image records a status row rather than stopping the run, the same way
build_faces.py does, so "failed" and "never attempted" stay distinct.

Exit code is 1 if any image was unreadable.
"""
import argparse
import csv
import sys
from pathlib import Path

from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

CATALOG = Path("data/metadata/image_catalog.csv")
THUMBNAILS = Path("thumbnails")
MANIFEST = THUMBNAILS / "manifest.csv"

# kind -> (longest edge in pixels, JPEG quality)
SIZES = {
    "grid": (400, 82),
    "lightbox": (1600, 88),
}

FIELDNAMES = [
    "sha256",
    "source_path",
    "grid_width",
    "grid_height",
    "lightbox_width",
    "lightbox_height",
    "status",
]

RULE = "=" * 76
THIN = "-" * 76


def thumbnail_size(width, height, longest):
    """Target size for a longest-edge fit, aspect preserved.

    Never enlarges: a source already inside the box keeps its own size,
    so a 200px scan does not become a soft 400px one.
    """
    if width <= 0 or height <= 0:
        raise ValueError("image has no area: " + str((width, height)))

    if max(width, height) <= longest:
        return width, height

    scale = longest / max(width, height)

    return max(1, round(width * scale)), max(1, round(height * scale))


def output_path(kind, sha):
    return THUMBNAILS / kind / (sha + ".jpg")


def pending_kinds(sha, force=False):
    """Which outputs still need generating for this sha."""
    if force:
        return list(SIZES)

    return [kind for kind in SIZES if not output_path(kind, sha).exists()]


def draft_longest(kinds):
    """The largest edge any pending output needs.

    One decode serves both sizes, so the draft has to keep enough
    resolution for the biggest of them.
    """
    return max(SIZES[kind][0] for kind in kinds)


def read_rows(path, encoding="utf-8"):
    if not path.exists():
        return []

    with path.open("r", encoding=encoding, newline="") as handle:
        return list(csv.DictReader(handle))


def existing_size(path):
    """Dimensions of an output already on disk, or blanks if unreadable."""
    try:
        with Image.open(path) as image:
            return image.size
    except Exception:
        return "", ""


def render(source, sha, kinds):
    """Decode once, write every pending size. Returns {kind: (w, h)}."""
    with Image.open(source) as raw:
        # No-op for PNG, GIF and TIFF; the win is on JPEG and MPO, which
        # are 366 of the 441 images here.
        raw.draft("RGB", (draft_longest(kinds),) * 2)
        image = ImageOps.exif_transpose(raw).convert("RGB")

    written = {}

    for kind in kinds:
        longest, quality = SIZES[kind]
        size = thumbnail_size(image.width, image.height, longest)

        resized = (
            image if size == image.size
            else image.resize(size, Image.LANCZOS)
        )

        target = output_path(kind, sha)
        target.parent.mkdir(parents=True, exist_ok=True)
        resized.save(target, "JPEG", quality=quality, optimize=True)

        written[kind] = size

    return written


def row_for(sha, source, sizes, status):
    grid = sizes.get("grid", ("", ""))
    lightbox = sizes.get("lightbox", ("", ""))

    return {
        "sha256": sha,
        "source_path": source,
        "grid_width": grid[0],
        "grid_height": grid[1],
        "lightbox_width": lightbox[0],
        "lightbox_height": lightbox[1],
        "status": status,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate every thumbnail, ignoring what is on disk.",
    )
    args = parser.parse_args()

    register_heif_opener()

    catalog = read_rows(CATALOG, encoding="utf-8-sig")

    if not catalog:
        print("No catalog at " + str(CATALOG) + ". Run build_catalog.py.")
        return 1

    recorded = {row["source_path"]: row for row in read_rows(MANIFEST)}

    print()
    print(RULE)
    print("VISUALMIND - BUILD THUMBNAILS")
    print(RULE)
    print("Catalog images:  " + str(len(catalog)))
    print("Sizes:           " + ", ".join(
        kind + " " + str(longest) + "px q" + str(quality)
        for kind, (longest, quality) in SIZES.items()
    ))
    print("Already built:   " + ("0 (--force)" if args.force
                                 else str(len(recorded))))
    print()

    rows = []
    generated = 0
    skipped = 0
    failures = 0

    for position, entry in enumerate(catalog, start=1):
        sha = entry["sha256"]
        source = entry["source_path"]
        kinds = pending_kinds(sha, args.force)

        if not kinds:
            skipped += 1
            carried = recorded.get(source)

            # A sha can be shared by two catalog rows, and the manifest
            # may have been deleted while the images survived, so a
            # skipped image still has to produce a row.
            rows.append(carried if carried else row_for(
                sha,
                source,
                {kind: existing_size(output_path(kind, sha))
                 for kind in SIZES},
                "ok",
            ))
            continue

        try:
            written = render(Path(source), sha, kinds)
        except Exception as error:
            print("  SKIP " + entry["filename"] + ": " + str(error))
            failures += 1
            rows.append(row_for(sha, source, {}, "unreadable"))
            continue

        for kind in SIZES:
            if kind not in written:
                written[kind] = existing_size(output_path(kind, sha))

        generated += 1
        rows.append(row_for(sha, source, written, "ok"))

        if generated % 50 == 0:
            print("  " + str(position) + "/" + str(len(catalog)))

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    with MANIFEST.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    on_disk = sum(
        1 for kind in SIZES
        for _ in (THUMBNAILS / kind).glob("*.jpg")
    )

    print()
    print(THIN)
    print("THUMBNAIL SUMMARY")
    print(THIN)
    print("Generated:          " + str(generated))
    print("Skipped (existing): " + str(skipped))
    print("Unreadable:         " + str(failures))
    print("Files on disk:      " + str(on_disk))
    print("Manifest rows:      " + str(len(rows)))
    print()
    print("Grid:      " + str((THUMBNAILS / "grid").resolve()))
    print("Lightbox:  " + str((THUMBNAILS / "lightbox").resolve()))
    print("Manifest:  " + str(MANIFEST.resolve()))
    print()
    print(RULE)
    print("THUMBNAILS COMPLETE")
    print(RULE)

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
