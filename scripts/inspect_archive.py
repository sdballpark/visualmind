from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from pillow_heif import register_heif_opener


register_heif_opener()

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".heic", ".heif",
    ".gif", ".webp", ".tif", ".tiff"
}


def human_size(size_bytes: int) -> str:
    value = float(size_bytes)

    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:,.2f} {unit}"
        value /= 1024

    return f"{size_bytes:,} B"


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def inspect_image(path: Path) -> tuple[bool, bool, str | None]:
    try:
        with Image.open(path) as image:
            image.load()
            exif = image.getexif()
            return True, bool(exif), None

    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return False, False, str(exc)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect the Gmail family-photo archive."
    )

    parser.add_argument(
        "source",
        type=Path,
        help="Path to gmail-family-photo-downloader output directory."
    )

    args = parser.parse_args()

    source = args.source.expanduser().resolve()

    if not source.exists():
        raise SystemExit(f"ERROR: Source does not exist: {source}")

    if not source.is_dir():
        raise SystemExit(f"ERROR: Source is not a directory: {source}")

    manifest_path = source / "manifest.csv"

    image_files = sorted(
        path
        for path in source.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    format_counts = Counter()
    year_counts = Counter()

    total_bytes = 0
    readable = 0
    exif_count = 0
    unreadable: list[tuple[Path, str]] = []

    print()
    print("=" * 72)
    print("VISUALMIND - GMAIL PHOTO ARCHIVE INSPECTION")
    print("=" * 72)
    print(f"Source:   {source}")
    print(f"Manifest: {manifest_path}")
    print()

    for number, path in enumerate(image_files, start=1):

        format_counts[path.suffix.lower()] += 1

        relative = path.relative_to(source)

        if relative.parts:
            candidate_year = relative.parts[0]

            if candidate_year.isdigit() and len(candidate_year) == 4:
                year_counts[candidate_year] += 1

        try:
            total_bytes += path.stat().st_size
        except OSError:
            pass

        ok, has_exif, error = inspect_image(path)

        if ok:
            readable += 1

            if has_exif:
                exif_count += 1
        else:
            unreadable.append(
                (path, error or "Unknown image error")
            )

        if number % 100 == 0:
            print(
                f"Inspected {number:,} / "
                f"{len(image_files):,} images..."
            )

    manifest_rows = read_manifest(manifest_path)

    print()
    print("-" * 72)
    print("ARCHIVE SUMMARY")
    print("-" * 72)

    print(f"Image files:          {len(image_files):,}")
    print(f"Total image storage:  {human_size(total_bytes)}")
    print(f"Readable images:      {readable:,}")
    print(f"Unreadable images:    {len(unreadable):,}")
    print(f"Images with EXIF:     {exif_count:,}")
    print(f"Manifest rows:        {len(manifest_rows):,}")

    if image_files:
        coverage = (exif_count / len(image_files)) * 100
        print(f"EXIF coverage:        {coverage:.1f}%")

    print()
    print("-" * 72)
    print("IMAGE FORMATS")
    print("-" * 72)

    for extension, count in format_counts.most_common():
        print(f"{extension:10} {count:>8,}")

    print()
    print("-" * 72)
    print("IMAGES BY GMAIL DIRECTORY YEAR")
    print("-" * 72)

    for year in sorted(year_counts):
        print(f"{year:10} {year_counts[year]:>8,}")

    if unreadable:
        print()
        print("-" * 72)
        print("UNREADABLE / INVALID IMAGES")
        print("-" * 72)

        for path, error in unreadable[:25]:
            print(path)
            print(f"    {error}")

        if len(unreadable) > 25:
            remaining = len(unreadable) - 25
            print(f"... plus {remaining:,} more.")

    print()
    print("=" * 72)
    print("INSPECTION COMPLETE")
    print("=" * 72)
    print()


if __name__ == "__main__":
    main()
