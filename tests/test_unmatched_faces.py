"""Faces the clusterer detected but could not place with anyone.

index() maps labelled faces to people, so a face DBSCAN marked as noise
carries no name and never reaches the item page. A photograph with three
detected faces and two named people then reads as a photograph with two
people - incomplete rather than wrong, and the one place this interface
stayed silent about its own uncertainty.
"""
import csv
import json

import pytest

from visualmind import people

FIELDS = ["person", "face_id", "source_path", "filename",
          "x1", "y1", "x2", "y2", "det_score", "sex", "age"]


def write_clusters(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()

        for person, face_id, source in rows:
            writer.writerow({
                "person": person, "face_id": face_id,
                "source_path": source, "filename": source,
                "x1": "0", "y1": "0", "x2": "10", "y2": "10",
                "det_score": "0.85", "sex": "", "age": "",
            })


@pytest.fixture
def clusters(tmp_path, monkeypatch):
    """Install a synthetic cluster file and its labels."""
    def install(rows, labels):
        path = tmp_path / "face_clusters.csv"
        write_clusters(path, rows)

        labels_path = tmp_path / "person_labels.json"
        labels_path.write_text(json.dumps(labels), encoding="utf-8")

        monkeypatch.setattr(people, "CLUSTERS", path)
        monkeypatch.setattr(people, "LABELS", labels_path)

    return install


def test_a_frame_where_everything_was_placed_reports_nothing(clusters):
    clusters(
        [("Person_001", "f1", "a.jpg"), ("Person_002", "f2", "a.jpg")],
        {"Ada": ["f1"], "Bo": ["f2"]},
    )

    assert people.unmatched() == {}


def test_an_unplaced_face_is_counted(clusters):
    """The reported case: three faces, two names, one unplaced."""
    clusters(
        [("Person_001", "f1", "a.jpg"),
         ("unassigned", "f2", "a.jpg"),
         ("Person_002", "f3", "a.jpg")],
        {"Ada": ["f1"], "Bo": ["f3"]},
    )

    assert people.unmatched() == {"a.jpg": 1}

    images, _ = people.index()
    assert sorted(images) == ["Ada", "Bo"]


def test_unplaced_faces_are_counted_per_image(clusters):
    clusters(
        [("unassigned", "f1", "a.jpg"),
         ("unassigned", "f2", "a.jpg"),
         ("unassigned", "f3", "b.jpg"),
         ("Person_001", "f4", "c.jpg")],
        {"Ada": ["f4"]},
    )

    assert people.unmatched() == {"a.jpg": 2, "b.jpg": 1}


def test_a_placed_face_is_never_counted_even_when_unlabelled(clusters):
    """Clustering placed it; nobody named the cluster.

    That is a labelling gap, not a clustering one, and this count claims
    only the second. The corpus has none today, and the distinction is
    worth holding on to if it ever does.
    """
    clusters(
        [("Person_007", "f1", "a.jpg"), ("unassigned", "f2", "a.jpg")],
        {},
    )

    assert people.unmatched() == {"a.jpg": 1}


def test_nothing_is_reported_without_a_cluster_file(monkeypatch, tmp_path):
    monkeypatch.setattr(people, "CLUSTERS", tmp_path / "absent.csv")
    monkeypatch.setattr(people, "LABELS", tmp_path / "absent.json")

    assert people.unmatched() == {}


def test_only_the_count_travels(clusters):
    """No name, no distance, no face id.

    Every unplaced face has a nearest labelled neighbour and a distance
    to it. Clustering already declined to act on them, and putting a
    name beside the face here would invite the trust the rejection
    withheld - the same error as a search presenting a gradient guess
    as a match.
    """
    clusters(
        [("unassigned", "f2", "a.jpg"), ("Person_001", "f1", "a.jpg")],
        {"Ada": ["f1"]},
    )

    value = people.unmatched()["a.jpg"]

    assert isinstance(value, int)
    assert value == 1
