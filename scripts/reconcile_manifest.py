from __future__ import annotations

import argparse
import csv
import hashlib
import os
from collections import Counter
from pathlib import Path


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


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to gmail-family-photo-downloader output directory.",
    )
    args = parser.parse_args()
    source = args.source.expanduser().resolve()

    manifest_path = source / "manifest.csv"

    report_dir = Path("data") / "metadata" / "reconciliation"

    manifest_rows = read_manifest(manifest_path)

    image_files = sorted(
        path
        for path in source.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    file_by_normalized_path = {
        normalize_path(path): path
        for path in image_files
    }

    manifest_by_path: dict[str, list[dict[str, str]]] = {}

    for row in manifest_rows:
        downloaded_path = row.get("downloaded_path", "").strip()

        if not downloaded_path:
            continue

        normalized = normalize_path(downloaded_path)
        manifest_by_path.setdefault(normalized, []).append(row)

    file_paths = set(file_by_normalized_path)
    manifest_paths = set(manifest_by_path)

    matched_paths = file_paths & manifest_paths
    files_without_manifest = file_paths - manifest_paths
    manifest_without_file = manifest_paths - file_paths

    duplicate_manifest_paths = {
        path: rows
        for path, rows in manifest_by_path.items()
        if len(rows) > 1
    }

    hash_counts = Counter(
        row.get("sha256", "").strip().lower()
        for row in manifest_rows
        if row.get("sha256", "").strip()
    )

    duplicate_manifest_hashes = {
        digest: count
        for digest, count in hash_counts.items()
        if count > 1
    }

    manifest_hashes = set(hash_counts)

    orphan_rows: list[dict[str, str]] = []

    print()
    print("=" * 76)
    print("VISUALMIND - MANIFEST RECONCILIATION")
    print("=" * 76)

    print(f"Image files on disk:             {len(image_files):,}")
    print(f"Manifest rows:                   {len(manifest_rows):,}")
    print(f"Unique manifest paths:           {len(manifest_paths):,}")
    print(f"Matched file/path records:       {len(matched_paths):,}")
    print(f"Files without manifest path:     {len(files_without_manifest):,}")
    print(f"Manifest paths without file:     {len(manifest_without_file):,}")
    print(f"Duplicate manifest paths:        {len(duplicate_manifest_paths):,}")
    print(f"Duplicate manifest SHA-256s:     {len(duplicate_manifest_hashes):,}")

    if files_without_manifest:
        print()
        print("-" * 76)
        print("HASHING FILES WITHOUT MANIFEST PATHS")
        print("-" * 76)

        for number, normalized in enumerate(
            sorted(files_without_manifest),
            start=1,
        ):
            path = file_by_normalized_path[normalized]

            try:
                digest = sha256_file(path)
            except OSError as exc:
                orphan_rows.append(
                    {
                        "path": str(path),
                        "sha256": "",
                        "status": "hash_error",
                        "detail": str(exc),
                    }
                )
                continue

            if digest.lower() in manifest_hashes:
                status = "HASH_EXISTS_IN_MANIFEST"
                detail = (
                    "Content is already represented by a manifest SHA-256 "
                    "but under another downloaded_path."
                )
            else:
                status = "NO_MANIFEST_RECORD"
                detail = (
                    "File path and SHA-256 are both absent from manifest."
                )

            orphan_rows.append(
                {
                    "path": str(path),
                    "sha256": digest,
                    "status": status,
                    "detail": detail,
                }
            )

            print(
                f"{number:>3}/{len(files_without_manifest)} "
                f"{status:24} {path.name}"
            )

    missing_file_rows: list[dict[str, str]] = []

    for normalized in sorted(manifest_without_file):
        for row in manifest_by_path[normalized]:
            missing_file_rows.append(
                {
                    "downloaded_path": row.get("downloaded_path", ""),
                    "original_filename": row.get("original_filename", ""),
                    "sha256": row.get("sha256", ""),
                    "message_date": row.get("message_date", ""),
                    "subject": row.get("subject", ""),
                    "gmail_message_id": row.get("gmail_message_id", ""),
                }
            )

    duplicate_path_rows: list[dict[str, str]] = []

    for normalized, rows in duplicate_manifest_paths.items():
        for row in rows:
            duplicate_path_rows.append(
                {
                    "downloaded_path": row.get("downloaded_path", ""),
                    "sha256": row.get("sha256", ""),
                    "message_date": row.get("message_date", ""),
                    "subject": row.get("subject", ""),
                    "gmail_message_id": row.get("gmail_message_id", ""),
                }
            )

    write_csv(
        report_dir / "files_without_manifest.csv",
        orphan_rows,
        ["path", "sha256", "status", "detail"],
    )

    write_csv(
        report_dir / "manifest_without_file.csv",
        missing_file_rows,
        [
            "downloaded_path",
            "original_filename",
            "sha256",
            "message_date",
            "subject",
            "gmail_message_id",
        ],
    )

    write_csv(
        report_dir / "duplicate_manifest_paths.csv",
        duplicate_path_rows,
        [
            "downloaded_path",
            "sha256",
            "message_date",
            "subject",
            "gmail_message_id",
        ],
    )

    truly_untracked = sum(
        1
        for row in orphan_rows
        if row["status"] == "NO_MANIFEST_RECORD"
    )

    hash_represented = sum(
        1
        for row in orphan_rows
        if row["status"] == "HASH_EXISTS_IN_MANIFEST"
    )

    print()
    print("-" * 76)
    print("RECONCILIATION RESULT")
    print("-" * 76)

    print(f"Files fully matched by path:     {len(matched_paths):,}")
    print(f"Untracked files:                 {truly_untracked:,}")
    print(f"Path mismatch / hash represented:{hash_represented:>6,}")
    print(f"Manifest references missing file:{len(missing_file_rows):>6,}")

    print()
    print("Reports written to:")
    print(report_dir.resolve())

    print()
    print("=" * 76)
    print("RECONCILIATION COMPLETE")
    print("=" * 76)
    print()


if __name__ == "__main__":
    main()
