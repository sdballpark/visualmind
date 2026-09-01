"""Source-fingerprint reporting in scripts/status.py.

Coverage compares row counts, and a source can be rewritten without its
row count changing. That is not hypothetical: the caption index reported
"441 OK" while it held embeddings of caption text that had since been
regenerated at a higher token cap. 441 rows against a 441-image catalog
is the right coverage answer to the wrong question.

Where a builder records what it was built from, that fingerprint is
compared too. This is a report, not a gate - retrieval.verify_index is
the check that refuses, on the different and unsafe-to-get-wrong
question of whether a matrix and its lookup still line up. Nothing here
may raise.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "status.py"

_spec = importlib.util.spec_from_file_location("status", SCRIPT)
status = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(status)

SOURCE = b"source_path,caption\n/photos/1.jpg,a dog\n"
FIELD = "captions_sha256"


def install(monkeypatch, tmp_path, *, recorded="match", manifest=True,
            source=SOURCE, field=FIELD, body=None):
    """Point one artifact's fingerprint entry at files under tmp_path.

    `recorded` is the sha the manifest claims: "match" for the real one,
    any other string for a stale or absent claim.
    """
    source_path = tmp_path / "captions.csv"

    if source is not None:
        source_path.write_bytes(source)

    manifest_path = tmp_path / "caption_index.json"

    if manifest:
        if body is None:
            sha = (
                hashlib.sha256(source or b"").hexdigest()
                if recorded == "match" else recorded
            )
            body = json.dumps(
                {"caption_count": 441, field: sha} if sha else
                {"caption_count": 441}
            )

        manifest_path.write_text(body, encoding="utf-8")

    monkeypatch.setattr(status, "SOURCE_FINGERPRINTS", {
        "caption index": (manifest_path, source_path, FIELD),
    })

    return manifest_path, source_path


# ---------------------------------------------------------------- states


def test_a_matching_fingerprint_says_nothing(monkeypatch, tmp_path):
    """The whole point is that a current index stays quiet."""
    install(monkeypatch, tmp_path)

    assert status.source_state("caption index") == (None, False)


def test_a_superseded_source_is_reported_and_stale(monkeypatch, tmp_path):
    """The gap this exists to close.

    The manifest records the sha of a captions.csv that has since been
    rewritten. Row counts are untouched, so only this sees it.
    """
    install(monkeypatch, tmp_path, recorded="0" * 64)

    note, stale = status.source_state("caption index")

    assert note == "built from superseded captions.csv"
    assert stale is True


def test_a_missing_manifest_is_stale(monkeypatch, tmp_path):
    """An index built before fingerprinting. Retrieval refuses it."""
    install(monkeypatch, tmp_path, manifest=False)

    note, stale = status.source_state("caption index")

    assert "no manifest" in note
    assert stale is True


def test_a_manifest_without_the_field_is_stale(monkeypatch, tmp_path):
    install(monkeypatch, tmp_path, body=json.dumps({"caption_count": 441}))

    note, stale = status.source_state("caption index")

    assert note == "caption_index.json records no " + FIELD
    assert stale is True


def test_a_missing_source_is_stale(monkeypatch, tmp_path):
    """Nothing to compare against, and nothing to rebuild from."""
    install(monkeypatch, tmp_path, source=None)

    note, stale = status.source_state("caption index")

    assert note == "source captions.csv is missing"
    assert stale is True


def test_an_unreadable_manifest_is_reported_rather_than_raised(
        monkeypatch, tmp_path):
    """A report that dies on a truncated JSON file is not a report."""
    install(monkeypatch, tmp_path, body="{not json")

    note, stale = status.source_state("caption index")

    assert note == "caption_index.json is unreadable"
    assert stale is True


@pytest.mark.parametrize("body", ["", "{not json", "[]", "null", "{}"])
def test_no_manifest_shape_can_raise(monkeypatch, tmp_path, body):
    """The guard: this reports, it never refuses.

    An empty file, malformed JSON, and valid JSON of the wrong type all
    have to come back as a note.
    """
    install(monkeypatch, tmp_path, body=body)

    note, stale = status.source_state("caption index")

    assert isinstance(note, str) and note
    assert stale is True


# ------------------------------------------------- artifacts not covered


def test_an_artifact_with_no_recorded_fingerprint_says_nothing():
    """Some artifacts have no manifest, and that is not a failure.

    They keep exactly the coverage answer they had before this check
    existed.
    """
    for artifact in ["captions", "face scan", "thumbnails", "palette"]:
        assert status.source_state(artifact) == (None, False)


def test_only_the_builders_that_record_a_source_are_listed():
    """Stated, not invented.

    Three builders write a source sha into a manifest. An artifact
    cannot be checked this way until its builder does.
    """
    assert set(status.SOURCE_FINGERPRINTS) == {
        "siglip2 index", "dinov2 index", "caption index"
    }


def test_thumbnails_and_palette_are_deliberately_absent():
    """Not an oversight, and not the same trade.

    A drifted thumbnail is a wrong picture, which is visible to anyone
    looking at the grid, and thumbnails are read on every page load - so
    a check erring cautious costs a blank grid. The dinov2 index is the
    opposite on both counts, which is why it is listed and these are
    not.
    """
    assert "thumbnails" not in status.SOURCE_FINGERPRINTS
    assert "palette" not in status.SOURCE_FINGERPRINTS


def test_each_fingerprinted_artifact_is_a_real_coverage_artifact():
    """A typo in the table would silently disable the check."""
    names = {entry[0] for entry in status.COVERAGE}

    assert set(status.SOURCE_FINGERPRINTS) <= names


def test_each_manifest_names_its_own_source_field():
    """The field differs per manifest, so it is stated per artifact."""
    expected = {
        "siglip2 index": (status.CATALOG, "catalog_sha256"),
        "dinov2 index": (status.CATALOG, "catalog_sha256"),
        "caption index": (status.CAPTIONS, "captions_sha256"),
    }

    for artifact, (source, field) in expected.items():
        _, actual_source, actual_field = status.SOURCE_FINGERPRINTS[artifact]

        assert (actual_source, actual_field) == (source, field)


def test_the_two_catalog_indexes_share_a_source_and_a_field():
    """Both derive from the catalog, so both read the same sha.

    They keep separate manifests because they are rebuilt separately;
    the shared source is what makes one rebuild unable to vouch for the
    other.
    """
    siglip_manifest, siglip_source, siglip_field = (
        status.SOURCE_FINGERPRINTS["siglip2 index"]
    )
    dino_manifest, dino_source, dino_field = (
        status.SOURCE_FINGERPRINTS["dinov2 index"]
    )

    assert (siglip_source, siglip_field) == (dino_source, dino_field)
    assert siglip_manifest != dino_manifest


# ------------------------------------------------------------ the report


def report_for(monkeypatch, tmp_path, *, catalog, covered, recorded):
    """One-artifact coverage report, so the two signals can be separated."""
    lookup = tmp_path / "caption_lookup.csv"
    lookup.write_text(
        "source_path\n" + "".join(path + "\n" for path in covered),
        encoding="utf-8",
    )

    install(monkeypatch, tmp_path, recorded=recorded)

    monkeypatch.setattr(status, "COVERAGE", [
        ("caption index", lookup, "build_caption_embeddings.py",
         None, "source_path"),
    ])

    return status.coverage_report(set(catalog))[0]


def test_full_coverage_and_a_stale_source_still_reads_stale(
        monkeypatch, tmp_path):
    """The regression, end to end.

    Every catalog image is covered, so coverage alone says OK. The
    source has moved on, so the row must not.
    """
    entry = report_for(
        monkeypatch, tmp_path,
        catalog=["/photos/1.jpg", "/photos/2.jpg"],
        covered=["/photos/1.jpg", "/photos/2.jpg"],
        recorded="0" * 64,
    )

    assert entry["covers"] == 2
    assert entry["missing"] == 0
    assert entry["orphaned"] == 0
    assert entry["status"] == "STALE - built from superseded captions.csv"
    assert entry["source_stale"] is True
    assert entry["stale"] is True


def test_full_coverage_and_a_current_source_reads_ok(monkeypatch, tmp_path):
    """No false staleness on the case that is genuinely fine."""
    entry = report_for(
        monkeypatch, tmp_path,
        catalog=["/photos/1.jpg"],
        covered=["/photos/1.jpg"],
        recorded="match",
    )

    assert entry["status"] == "OK"
    assert entry["source_stale"] is False
    assert entry["source_note"] is None
    assert entry["stale"] is False


def test_both_reasons_appear_together(monkeypatch, tmp_path):
    """Coverage and fingerprint are separate questions, both answerable."""
    entry = report_for(
        monkeypatch, tmp_path,
        catalog=["/photos/1.jpg", "/photos/2.jpg"],
        covered=["/photos/1.jpg"],
        recorded="0" * 64,
    )

    assert entry["status"] == (
        "STALE - 1 not covered, built from superseded captions.csv"
    )
    assert entry["missing"] == 1
    assert entry["source_stale"] is True


def test_a_stale_source_names_the_rebuild_script(monkeypatch, tmp_path):
    """main() prints this script under REBUILD NEEDED, so it must be there."""
    entry = report_for(
        monkeypatch, tmp_path,
        catalog=["/photos/1.jpg"],
        covered=["/photos/1.jpg"],
        recorded="0" * 64,
    )

    assert entry["script"] == "build_caption_embeddings.py"


def test_every_report_row_carries_the_new_keys(monkeypatch, tmp_path):
    """The /status endpoint serialises these rows straight out."""
    entry = report_for(
        monkeypatch, tmp_path,
        catalog=["/photos/1.jpg"],
        covered=["/photos/1.jpg"],
        recorded="match",
    )

    assert "source_stale" in entry
    assert "source_note" in entry


def test_a_missing_artifact_still_reports_the_new_keys(
        monkeypatch, tmp_path):
    """The MISSING branch returns early and must keep the same shape."""
    install(monkeypatch, tmp_path)

    monkeypatch.setattr(status, "COVERAGE", [
        ("caption index", tmp_path / "absent.csv",
         "build_caption_embeddings.py", None, "source_path"),
    ])

    entry = status.coverage_report({"/photos/1.jpg"})[0]

    assert entry["status"] == "MISSING"
    assert entry["source_stale"] is False
    assert entry["source_note"] is None


# ------------------------------------------------------------------ hash


def test_the_file_hash_matches_hashlib(tmp_path):
    """Read in blocks, so a file larger than one block must still agree."""
    path = tmp_path / "big.bin"
    payload = b"x" * (3 * (1 << 20) + 17)
    path.write_bytes(payload)

    assert status.sha256_file(path) == hashlib.sha256(payload).hexdigest()
