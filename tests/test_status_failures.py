"""Unreadable-source reporting in scripts/status.py.

build_faces.py and build_thumbnails.py record a row for every image they
reached, failures included, so a failure counts as covered. That is the
right coverage answer and the wrong headline: without a separate count,
the table reads "441 OK" while part of the corpus is unusable.
"""
import csv
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "status.py"

_spec = importlib.util.spec_from_file_location("status", SCRIPT)
status = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(status)

THUMBNAIL_FIELDS = ["sha256", "source_path", "status"]
FACE_FIELDS = ["source_path", "filename", "faces", "status"]


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    return path


def thumbnail_rows(states):
    return [
        {"sha256": str(i), "source_path": "/photos/" + str(i) + ".jpg",
         "status": state}
        for i, state in enumerate(states)
    ]


def test_an_artifact_without_a_status_column_reports_nothing(tmp_path):
    """captions.csv and the lookups have no status column."""
    path = write_csv(
        tmp_path / "captions.csv",
        ["source_path", "caption"],
        [{"source_path": "/photos/1.jpg", "caption": "a dog"}],
    )

    assert status.failures(path) == ({}, [])


def test_an_all_ok_artifact_reports_nothing(tmp_path):
    path = write_csv(tmp_path / "m.csv", THUMBNAIL_FIELDS,
                     thumbnail_rows(["ok", "ok", "ok"]))

    assert status.failures(path) == ({}, [])


def test_a_missing_artifact_reports_nothing(tmp_path):
    """Never built is not the same as built and broken."""
    assert status.failures(tmp_path / "absent.csv") == ({}, [])


def test_failures_are_counted_and_named(tmp_path):
    path = write_csv(tmp_path / "m.csv", THUMBNAIL_FIELDS,
                     thumbnail_rows(["ok", "unreadable", "ok", "unreadable"]))

    counts, names = status.failures(path)

    assert counts == {"unreadable": 2}
    assert names == ["1.jpg", "3.jpg"]


def test_failure_kinds_are_grouped_separately(tmp_path):
    """A future builder may record more than one way to fail."""
    path = write_csv(
        tmp_path / "m.csv", THUMBNAIL_FIELDS,
        thumbnail_rows(["ok", "unreadable", "truncated", "unreadable"]),
    )

    counts, _ = status.failures(path)

    assert counts == {"unreadable": 2, "truncated": 1}


def test_named_examples_are_capped(tmp_path):
    """The report names a few files, it does not dump the corpus."""
    path = write_csv(tmp_path / "m.csv", THUMBNAIL_FIELDS,
                     thumbnail_rows(["unreadable"] * 40))

    counts, names = status.failures(path)

    assert counts == {"unreadable": 40}
    assert len(names) == 5


def test_the_face_scan_layout_works_too(tmp_path):
    """face_scanned.csv carries different columns, same semantics."""
    path = write_csv(tmp_path / "face_scanned.csv", FACE_FIELDS, [
        {"source_path": "/photos/a.jpg", "filename": "a.jpg",
         "faces": "2", "status": "ok"},
        {"source_path": "/photos/b.jpg", "filename": "b.jpg",
         "faces": "0", "status": "unreadable"},
    ])

    assert status.failures(path) == ({"unreadable": 1}, ["b.jpg"])


def test_zero_faces_is_not_a_failure(tmp_path):
    """An image with no faces was scanned successfully.

    Treating it as a gap is the false-staleness case the module
    docstring already warns about for coverage.
    """
    path = write_csv(tmp_path / "face_scanned.csv", FACE_FIELDS, [
        {"source_path": "/photos/a.jpg", "filename": "a.jpg",
         "faces": "0", "status": "ok"},
    ])

    assert status.failures(path) == ({}, [])


@pytest.mark.parametrize("counts,expected", [
    ({"unreadable": 3}, "3 unreadable"),
    ({"unreadable": 2, "truncated": 1}, "1 truncated, 2 unreadable"),
    ({}, ""),
])
def test_the_status_column_note_reads_in_a_stable_order(counts, expected):
    assert status.failure_note(counts) == expected


def test_both_status_recording_artifacts_are_covered():
    """The two builders that can fail are both in the coverage table."""
    names = [name for name, _, _ in status.COVERAGE]

    assert "face scan" in names
    assert "thumbnails" in names
