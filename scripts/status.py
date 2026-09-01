"""Report pipeline state: what has been built, and what is stale.

Every derived artifact is compared against the catalog it was built
from. Adding photos invalidates the catalog; rebuilding the catalog
invalidates every index. Nothing else in the pipeline notices, so this
script is the check that does.

Face coverage is measured against the scan record, not the face index.
An image with no faces in it is correctly absent from the index but was
still scanned, and treating that as a gap would report false staleness
on 76 of 441 images in this corpus.

The same record keeps a row for an image the builder could not read, so
a failure stays distinguishable from an image never attempted. That row
counts as covered, correctly - it was attempted - which means coverage
alone would report "441 OK" while part of the corpus is broken. Non-ok
rows are therefore counted and reported on their own.

Exit code is 1 if anything is stale. Unreadable sources do not set it:
rebuilding cannot clear them, so failing here would leave the check
permanently red with nothing to act on but the source files.
"""
import csv
import json
import sys
import time
from pathlib import Path

CATALOG = Path("data/metadata/image_catalog.csv")
CAPTIONS = Path("data/metadata/captions.csv")
DUPLICATES = Path("data/metadata/duplicate_groups.csv")
CLUSTERS = Path("data/metadata/face_clusters.csv")
LABELS = Path("data/metadata/person_labels.json")
EVENTS = Path("data/metadata/events.csv")
INDEX_DIR = Path("indexes")
THUMBNAIL_MANIFEST = Path("thumbnails") / "manifest.csv"

COVERAGE = [
    ("captions", CAPTIONS, "build_captions.py"),
    ("siglip2 index", INDEX_DIR / "siglip2_lookup.csv",
     "build_embeddings.py"),
    ("caption index", INDEX_DIR / "caption_lookup.csv",
     "build_caption_embeddings.py"),
    ("dinov2 index", INDEX_DIR / "dinov2_lookup.csv",
     "build_visual_embeddings.py"),
    ("face scan", INDEX_DIR / "face_scanned.csv", "build_faces.py"),
    ("thumbnails", THUMBNAIL_MANIFEST, "build_thumbnails.py"),
]


def read_paths(path):
    """Set of source_path values in a CSV, or None if the file is absent."""
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["source_path"] for row in csv.DictReader(handle)}


def read_rows(path):
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def failures(path):
    """Non-ok status rows: counts by state, and a few names to look at.

    build_faces.py and build_thumbnails.py both record a status column.
    Anything other than "ok" is a source the builder reached and could
    not use.
    """
    counts = {}
    names = []

    for row in read_rows(path):
        state = row.get("status")

        if not state or state == "ok":
            continue

        counts[state] = counts.get(state, 0) + 1

        if len(names) < 5:
            names.append(Path(row["source_path"]).name)

    return counts, names


def failure_note(counts):
    """Render non-ok counts for the status column."""
    return ", ".join(
        str(count) + " " + state
        for state, count in sorted(counts.items())
    )


def age(path):
    if not path.exists():
        return "-"

    delta = time.time() - Path(path).stat().st_mtime

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
    print("ARTIFACT".ljust(18) + "COVERS".ljust(10) + "BUILT".ljust(12)
          + "STATUS")
    print("-" * 76)

    stale = []
    broken = []

    for name, path, script in COVERAGE:
        paths = read_paths(path)

        if paths is None:
            print(name.ljust(18) + "-".ljust(10) + "-".ljust(12) + "MISSING")
            stale.append((name, script, "never built"))
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

        # A failed image is covered but not usable, so the coverage
        # count cannot carry this and the status column has to.
        counts, names = failures(path)

        if counts:
            status = status + ", " + failure_note(counts)
            broken.append((name, counts, names))

        print(name.ljust(18) + str(len(paths)).ljust(10)
              + age(path).ljust(12) + status)

    print()
    print("-" * 76)
    print("DERIVED FROM INDEXES")
    print("-" * 76)

    # Duplicate groups come from the DINOv2 index, not the catalog.
    dupes = read_paths(DUPLICATES)

    if dupes is None:
        print("duplicates       not built".ljust(40)
              + "find_duplicates.py")
    else:
        dino = read_paths(INDEX_DIR / "dinov2_lookup.csv") or set()
        outside = dupes - dino
        note = "OK" if not outside else (
            str(len(outside)) + " entries outside the index")

        rows = read_rows(DUPLICATES)
        groups = len({r["group"] for r in rows})

        print("duplicates       " + str(groups) + " groups, "
              + str(len(rows)) + " entries, built " + age(DUPLICATES)
              + " - " + note)

        if outside:
            stale.append(("duplicates", "find_duplicates.py", note))

    # Face clusters come from the face index.
    face_index = read_paths(INDEX_DIR / "face_lookup.csv")
    cluster_rows = read_rows(CLUSTERS)

    if not cluster_rows:
        print("face clusters    not built".ljust(40) + "cluster_faces.py")
    else:
        clustered = {r["face_id"] for r in cluster_rows}
        face_rows = read_rows(INDEX_DIR / "face_lookup.csv")
        known_faces = {r["face_id"] for r in face_rows}

        uncovered = known_faces - clustered
        note = "OK" if not uncovered else (
            str(len(uncovered)) + " faces not clustered")

        people_count = len({
            r["person"] for r in cluster_rows
            if r["person"] != "unassigned"
        })

        print("face clusters    " + str(people_count) + " clusters over "
              + str(len(clustered)) + " faces, built " + age(CLUSTERS)
              + " - " + note)

        if uncovered:
            stale.append(("face clusters", "cluster_faces.py", note))

    # Events come from the catalog's EXIF, not from an index.
    event_rows = read_rows(EVENTS)

    if not event_rows:
        print("events           not built".ljust(40) + "build_events.py")
    else:
        covered = {r["source_path"] for r in event_rows}
        missing = catalog_paths - covered
        placed = {r["source_path"] for r in event_rows
                  if r["event_id"] != "unassigned"}
        count = len({r["event_id"] for r in event_rows
                     if r["event_id"] != "unassigned"})

        note = "OK" if not missing else (
            str(len(missing)) + " catalog images not covered")

        print("events           " + str(count) + " events over "
              + str(len(placed)) + " images, " + str(len(covered) - len(placed))
              + " unassigned, built " + age(EVENTS) + " - " + note)

        if missing:
            stale.append(("events", "build_events.py", note))

    if LABELS.exists():
        labels = json.loads(LABELS.read_text(encoding="utf-8"))
        labelled_faces = sum(len(v) for v in labels.values())

        total_faces = len(face_index or [])

        if face_index is not None:
            face_rows = read_rows(INDEX_DIR / "face_lookup.csv")
            total_faces = len(face_rows)

        pct = ""

        if total_faces:
            pct = ("  (" + format(100 * labelled_faces / total_faces, ".0f")
                   + "% of faces)")

        print("person labels    " + str(len(labels)) + " people, "
              + str(labelled_faces) + " faces" + pct
              + ", built " + age(LABELS))
    else:
        print("person labels    not built".ljust(40) + "label_faces.py")

    print()

    if broken:
        print("-" * 76)
        print("UNREADABLE SOURCES")
        print("-" * 76)

        for name, counts, names in broken:
            print("  " + name.ljust(18) + failure_note(counts))
            print("  " + " " * 18 + ", ".join(names)
                  + (", ..." if sum(counts.values()) > len(names) else ""))
            print()

        print("  Rebuilding will not clear these. The builder reached each")
        print("  file and recorded that it could not be used, so the source")
        print("  is what needs attention - not the artifact.")
        print()

    if not stale:
        if broken:
            total = sum(sum(c.values()) for _, c, _ in broken)
            print("Nothing is stale. " + str(total)
                  + " source(s) could not be read - see above.")
        else:
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
