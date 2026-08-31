"""The manifest check that binds an embedding matrix to its lookup.

Scores are matched to lookup rows by position, so a lookup rebuilt out
of step with its .npy attaches every score to the wrong image. A row
count alone does not catch it: the dangerous case keeps the count and
changes the order.
"""
import csv
import json

import numpy as np
import pytest

from visualmind import retrieval

PATHS = ["a.jpg", "b.jpg", "c.jpg"]


def build(tmp_path, paths, matrix_rows=None, manifest="fingerprint",
          fingerprint_over=None):
    """Write an index triple, each part independently corruptible."""
    embeddings = tmp_path / "emb.npy"
    lookup = tmp_path / "lookup.csv"
    manifest_path = tmp_path / "manifest.json"

    rows = len(paths) if matrix_rows is None else matrix_rows
    np.save(embeddings, np.zeros((rows, 4), dtype=np.float32))

    with lookup.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_path", "filename"])
        writer.writeheader()

        for path in paths:
            writer.writerow({"source_path": path, "filename": path})

    if manifest == "fingerprint":
        body = {
            "lookup_fingerprint": retrieval.lookup_fingerprint(
                fingerprint_over or paths
            )
        }
        manifest_path.write_text(json.dumps(body), encoding="utf-8")
    elif manifest == "legacy":
        manifest_path.write_text(json.dumps({"image_count": 3}),
                                 encoding="utf-8")

    return embeddings, lookup, manifest_path


def test_fingerprint_is_order_sensitive():
    """The point of hashing the column rather than counting it."""
    assert (retrieval.lookup_fingerprint(["a", "b"])
            != retrieval.lookup_fingerprint(["b", "a"]))


def test_matching_index_loads(tmp_path):
    matrix, rows = retrieval.load_index(
        *build(tmp_path, PATHS), "caption"
    )

    assert matrix.shape[0] == 3
    assert [row["source_path"] for row in rows] == PATHS


def test_reordered_lookup_is_refused(tmp_path):
    """The silent case: same rows, same count, different order.

    Nothing else in the pipeline notices this, and every score would be
    read against the wrong image.
    """
    parts = build(tmp_path, PATHS, fingerprint_over=list(reversed(PATHS)))

    with pytest.raises(retrieval.IndexMismatch) as error:
        retrieval.load_index(*parts, "caption")

    assert "reordered" in str(error.value)
    assert "scripts/build_caption_embeddings.py" in str(error.value)


def test_row_count_mismatch_is_refused(tmp_path):
    parts = build(tmp_path, PATHS, matrix_rows=2)

    with pytest.raises(retrieval.IndexMismatch) as error:
        retrieval.load_index(*parts, "image")

    assert "2 embedding rows against 3 lookup rows" in str(error.value)
    assert "scripts/build_embeddings.py" in str(error.value)


def test_missing_manifest_is_refused(tmp_path):
    parts = build(tmp_path, PATHS, manifest=None)

    with pytest.raises(retrieval.IndexMismatch) as error:
        retrieval.load_index(*parts, "image")

    assert "is missing" in str(error.value)


def test_manifest_without_a_fingerprint_is_refused(tmp_path):
    """An index built before fingerprinting cannot be vouched for.

    Accepting it would mean the check silently does nothing on exactly
    the indexes most likely to have drifted.
    """
    parts = build(tmp_path, PATHS, manifest="legacy")

    with pytest.raises(retrieval.IndexMismatch) as error:
        retrieval.load_index(*parts, "caption")

    assert "predates lookup fingerprinting" in str(error.value)
