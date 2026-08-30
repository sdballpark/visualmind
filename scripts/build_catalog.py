from __future__ import annotations

import argparse
import csv
import hashlib
import os
from collections import Counter
from pathlib import Path

from PIL import Image, ExifTags
from pillow_heif import register_heif_opener


register_heif_opener()

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".heic", ".heif",
    ".gif", ".webp", ".tif", ".tiff"
}


def normalize_path(path: Path | str) -> str:
    return os.path.normcase(os.path.abspath(str(path)))


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def exif_value(exif, tag: int) -> str:
    value = exif.get(tag)

    if value is None:
        return ""

    return str(value).strip()


def inspect_metadata(path: Path) -> dict[str, str | int]:
    with Image.open(path) as image:
        width, height = image.size
        image_format = image.format or ""
        mode = image.mode or ""

        exif = image.getexif()

        # IFD0 metadata
        datetime_modified = exif_value(
            exif,
            ExifTags.Base.DateTime
        )

        camera_make = exif_value(
            exif,
            ExifTags.Base.Make
        )

        camera_model = exif_value(
            exif,
            ExifTags.Base.Model
        )

        orientation = exif_value(
            exif,
            ExifTags.Base.Orientation
        )

        # DateTimeOriginal and DateTimeDigitized normally live
        # inside the dedicated Exif sub-IFD.
        try:
            exif_ifd = exif.get_ifd(ExifTags.IFD.Exif)
        except (KeyError, TypeError, ValueError):
            exif_ifd = {}

        datetime_original = exif_value(
            exif_ifd,
            0x9003
        )

        datetime_digitized = exif_value(
            exif_ifd,
            0x9004
        )

        best_capture_date = (
            datetime_original
            or datetime_digitized
            or datetime_modified
        )

        return {
            "width": width,
            "height": height,
            "image_format": image_format,
            "image_mode": mode,
            "has_exif": 1 if bool(exif) else 0,
            "exif_datetime_original": datetime_original,
            "exif_datetime_digitized": datetime_digitized,
            "exif_datetime_modified": datetime_modified,
            "best_exif_date": best_capture_date,
            "camera_make": camera_make,
            "camera_model": camera_model,
            "orientation": orientation,
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build normalized VisualMind image catalog."
    )

    parser.add_argument(
        "source",
        type=Path,
        help="Gmail downloader output directory.",
    )

    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    manifest_path = source / "manifest.csv"

    output_path = (
        Path("data")
        / "metadata"
        / "image_catalog.csv"
    )

    manifest_rows = read_manifest(manifest_path)

    manifest_by_path: dict[str, dict[str, str]] = {}

    for row in manifest_rows:
        path = row.get("downloaded_path", "").strip()

        if path:
            manifest_by_path[normalize_path(path)] = row

    manifest_by_sha: dict[str, dict[str, str]] = {}

    for row in manifest_rows:
        sha = row.get("sha256", "").strip().lower()

        if sha:
            manifest_by_sha[sha] = row

    image_files = sorted(
        path
        for path in source.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    rows: list[dict[str, str | int]] = []

    provenance_counts = Counter()
    format_counts = Counter()

    print()
    print("=" * 76)
    print("VISUALMIND - BUILD NORMALIZED IMAGE CATALOG")
    print("=" * 76)

    for number, path in enumerate(image_files, start=1):
        normalized = normalize_path(path)
        manifest = manifest_by_path.get(normalized)

        if manifest is None:
            manifest = manifest_by_sha.get(sha256_file(path))

        relative = path.relative_to(source)

        gmail_year = (
            relative.parts[0]
            if len(relative.parts) >= 1
            else ""
        )

        gmail_month = (
            relative.parts[1]
            if len(relative.parts) >= 2
            else ""
        )

        if manifest:
            provenance_status = "MANIFEST"
            sha256 = manifest.get("sha256", "").strip()
        else:
            provenance_status = "UNTRACKED"
            sha256 = sha256_file(path)

        provenance_counts[provenance_status] += 1

        try:
            metadata = inspect_metadata(path)
        except Exception as exc:
            metadata = {
                "width": "",
                "height": "",
                "image_format": "",
                "image_mode": "",
                "has_exif": 0,
                "exif_datetime_original": "",
                "exif_datetime_digitized": "",
                "exif_datetime_modified": "",
                "best_exif_date": "",
                "camera_make": "",
                "camera_model": "",
                "orientation": "",
            }

            image_error = str(exc)
        else:
            image_error = ""

        format_counts[path.suffix.lower()] += 1

        rows.append(
            {
                "image_id": sha256,
                "source_path": str(path),
                "relative_path": str(relative),
                "filename": path.name,
                "extension": path.suffix.lower(),
                "file_bytes": path.stat().st_size,
                "sha256": sha256,
                "provenance_status": provenance_status,

                "gmail_directory_year": gmail_year,
                "gmail_directory_month": gmail_month,

                "gmail_message_date":
                    manifest.get("message_date", "")
                    if manifest else "",

                "gmail_from":
                    manifest.get("from", "")
                    if manifest else "",

                "gmail_subject":
                    manifest.get("subject", "")
                    if manifest else "",

                "gmail_message_id":
                    manifest.get("gmail_message_id", "")
                    if manifest else "",

                "gmail_thread_id":
                    manifest.get("gmail_thread_id", "")
                    if manifest else "",

                "original_filename":
                    manifest.get("original_filename", "")
                    if manifest else "",

                **metadata,

                "image_error": image_error,
            }
        )

        if number % 100 == 0:
            print(
                f"Processed {number:,} / "
                f"{len(image_files):,} images..."
            )

    fieldnames = [
        "image_id",
        "source_path",
        "relative_path",
        "filename",
        "extension",
        "file_bytes",
        "sha256",
        "provenance_status",

        "gmail_directory_year",
        "gmail_directory_month",
        "gmail_message_date",
        "gmail_from",
        "gmail_subject",
        "gmail_message_id",
        "gmail_thread_id",
        "original_filename",

        "width",
        "height",
        "image_format",
        "image_mode",
        "has_exif",

        "exif_datetime_original",
        "exif_datetime_digitized",
        "exif_datetime_modified",
        "best_exif_date",

        "camera_make",
        "camera_model",
        "orientation",

        "image_error",
    ]

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    exif_original_count = sum(
        1
        for row in rows
        if row["exif_datetime_original"]
    )

    print()
    print("-" * 76)
    print("CATALOG SUMMARY")
    print("-" * 76)

    print(f"Catalog records:                {len(rows):,}")
    print(
        f"Manifest provenance:            "
        f"{provenance_counts['MANIFEST']:,}"
    )
    print(
        f"Untracked provenance:           "
        f"{provenance_counts['UNTRACKED']:,}"
    )
    print(
        f"EXIF DateTimeOriginal present:  "
        f"{exif_original_count:,}"
    )

    print()
    print("Catalog written to:")
    print(output_path.resolve())

    print()
    print("=" * 76)
    print("CATALOG COMPLETE")
    print("=" * 76)
    print()


if __name__ == "__main__":
    main()
