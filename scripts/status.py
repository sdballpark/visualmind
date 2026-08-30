"""Report pipeline state: what has been built, and what is stale.

Every derived artifact is compared against the catalog it was built from.
Adding photos to the archive invalidates the catalog; rebuilding the
catalog invalidates every index. Nothing else in the pipeline notices, so
this script is the check that does.

Exit code is 1 if anything is stale, so it can gate a rebuild.
"""
import csv
import sys
from pathlib import Path

CATALOG = Path("data/metadata/image_catalog.csv")
CAPTIONS = Path("data/metadata/captions.csv")
DUPLICATES = Path("data/metadata/duplicate_groups.csv")
INDEX_DIR = Path("indexes")

ARTIFACTS = [
    ("catalog", CATALOG, "source_path", None),
    ("captions", CAPTIONS, "source_path", "build_captions.py"),
    ("siglip2 index", INDEX_DIR / "siglip2_lookup.csv", "source_path",
     "build_embeddings.py"),
    ("caption index", INDEX_DIR / "caption_lookup.csv", "source_path",
     "build_caption_embeddings.py"),
    ("dinov2 index", INDEX_DIR / "dinov2_lookup.csv", "source_path",
     "build_visual_embeddings.py"),
]


def read_paths(path):
    """Return the set of source_path values in a CSV, or None if absent."""
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["source_path"] for row in csv.DictReader(handle)}


def age(path):
    if not path.exists():
        return "-"

    seconds = Path(path).stat().st_mtime
    import time
    delta = time.time() - seconds

    if delta < 3600:
        return str(int(delta / 60)) + "m ago"

    if delta < 86400:
        return str(int(delta / 3600)) + "h ago"

    return str(int(delta / 86400)) + "d ago"


def main():
    print()
    print("=" * 76)
    print("VISUALMIND - PIPELINE STATUS")
    print("=" * 76)

    catalog_paths = read_paths(CATALOG)

    if catalog_paths is None:
        print()
        print("No catalog. Run build_catalog.py first.")
        print()
        return 1

    print()
    print("Catalog: " + str(len(catalog_paths)) + " images, built "
          + age(CATALOG))
    print()
    print("-" * 76)
    print("ARTIFACT".ljust(18) + "COUNT".ljust(10) + "BUILT".ljust(12)
          + "STATUS")
    print("-" * 76)

    stale = []

    for name, path, _field, script in ARTIFACTS:
        paths = read_paths(path)

        if paths is None:
            print(name.ljust(18) + "-".ljust(10) + "-".ljust(12)
                  + "MISSING")

            if script:
                stale.append((name, script, "never built"))

            continue

        count = str(len(paths))
        built = age(path)

        if name == "catalog":
            print(name.ljust(18) + count.ljust(10) + built.ljust(12) + "OK")
            continue

        missing = catalog_paths - paths
        orphaned = paths - catalog_paths

        if not missing and not orphaned:
            status = "OK"
        else:
            parts = []

            if missing:
                parts.append(str(len(missing)) + " not covered")

            if orphaned:
                parts.append(str(len(orphaned)) + " no longer in catalog")

            status = "STALE - " + ", ".join(parts)
            stale.append((name, script, status))

        print(name.ljust(18) + count.ljust(10) + built.ljust(12) + status)

    # Duplicate report is derived from the DINOv2 index, not the catalog.
    dupes = read_paths(DUPLICATES)

    print()

    if dupes is None:
        print("Duplicate report: not built (find_duplicates.py)")
    else:
        dino = read_paths(INDEX_DIR / "dinov2_lookup.csv") or set()
        outside = dupes - dino

        note = "OK" if not outside else (
            str(len(outside)) + " entries outside the current index")
        print("Duplicate report: " + str(len(dupes)) + " grouped entries, "
              + "built " + age(DUPLICATES) + " - " + note)

    print()

    if not stale:
        print("Everything is current.")
        print()
        return 0

    print("-" * 76)
    print("REBUILD NEEDED")
    print("-" * 76)

    for name, script, reason in stale:
        print("  " + name.ljust(18) + reason)
        print("  " + " " * 18 + "uv run python scripts/" + str(script))
        print()

    return 1


if __name__ == "__main__":
    sys.exit(main())
