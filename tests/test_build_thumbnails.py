"""Thumbnail sizing and resume logic, without generating images.

The decode path needs real files and a slow Pillow round-trip; the
arithmetic and the skip decision do not, and those are where an error
would be silent rather than loud.
"""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = (Path(__file__).resolve().parents[1]
          / "scripts" / "build_thumbnails.py")

_spec = importlib.util.spec_from_file_location("build_thumbnails", SCRIPT)
build = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build)


@pytest.fixture
def thumb_root(monkeypatch, tmp_path):
    monkeypatch.setattr(build, "THUMBNAILS", tmp_path)
    return tmp_path


# --- sizing ----------------------------------------------------------


def test_longest_edge_hits_the_target_for_a_landscape_image():
    assert build.thumbnail_size(4000, 3000, 400) == (400, 300)


def test_longest_edge_hits_the_target_for_a_portrait_image():
    """Orientation must not decide which edge is constrained."""
    assert build.thumbnail_size(3000, 4000, 400) == (300, 400)


def test_aspect_ratio_survives_the_scale():
    width, height = build.thumbnail_size(5472, 3648, 1600)

    assert width == 1600
    assert abs(width / height - 5472 / 3648) < 0.01


def test_a_small_source_is_not_enlarged():
    """A 200px scan blown up to 400 is softer, not better."""
    assert build.thumbnail_size(200, 150, 400) == (200, 150)


def test_a_source_exactly_at_the_target_is_left_alone():
    assert build.thumbnail_size(400, 260, 400) == (400, 260)


def test_an_extreme_panorama_keeps_at_least_one_pixel():
    """Rounding a 10000x3 strip must not produce a zero dimension."""
    width, height = build.thumbnail_size(10000, 3, 400)

    assert width == 400
    assert height == 1


def test_a_zero_dimension_is_rejected():
    with pytest.raises(ValueError):
        build.thumbnail_size(0, 100, 400)


# --- resume ----------------------------------------------------------


def write(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_nothing_on_disk_means_both_sizes_are_pending(thumb_root):
    assert sorted(build.pending_kinds("abc")) == ["grid", "lightbox"]


def test_both_present_means_nothing_is_pending(thumb_root):
    for kind in build.SIZES:
        write(thumb_root / kind / "abc.jpg")

    assert build.pending_kinds("abc") == []


def test_a_half_finished_image_regenerates_only_what_is_missing(thumb_root):
    """An interrupted run leaves exactly this state behind."""
    write(thumb_root / "grid" / "abc.jpg")

    assert build.pending_kinds("abc") == ["lightbox"]


def test_force_marks_everything_pending_even_when_present(thumb_root):
    for kind in build.SIZES:
        write(thumb_root / kind / "abc.jpg")

    assert sorted(build.pending_kinds("abc", force=True)) == [
        "grid", "lightbox"
    ]


def test_outputs_are_keyed_by_sha_not_filename(thumb_root):
    """Filenames repeat in this corpus; a filename key would overwrite."""
    path = build.output_path("grid", "0" * 64)

    assert path.name == "0" * 64 + ".jpg"
    assert path.parent.name == "grid"


# --- decode budget ---------------------------------------------------


def test_the_draft_keeps_enough_resolution_for_the_largest_output():
    assert build.draft_longest(["grid", "lightbox"]) == 1600
    assert build.draft_longest(["lightbox"]) == 1600


def test_a_grid_only_rebuild_drafts_down_to_the_grid_size():
    """The whole point of drafting: do not decode 45 MP for a 400px file."""
    assert build.draft_longest(["grid"]) == 400


# --- manifest --------------------------------------------------------


def test_an_unreadable_image_records_blank_dimensions_not_zeros():
    row = build.row_for("abc", "a.jpg", {}, "unreadable")

    assert row["status"] == "unreadable"
    assert row["grid_width"] == ""
    assert row["lightbox_height"] == ""


def test_a_manifest_row_carries_both_output_sizes():
    row = build.row_for(
        "abc", "a.jpg", {"grid": (400, 300), "lightbox": (1600, 1200)}, "ok"
    )

    assert (row["grid_width"], row["grid_height"]) == (400, 300)
    assert (row["lightbox_width"], row["lightbox_height"]) == (1600, 1200)
    assert list(row) == build.FIELDNAMES


def test_status_covers_thumbnails_against_the_catalog():
    """The manifest keys on source_path so status.py can diff it."""
    assert "source_path" in build.FIELDNAMES

    status_spec = importlib.util.spec_from_file_location(
        "status", SCRIPT.parent / "status.py"
    )
    status = importlib.util.module_from_spec(status_spec)
    status_spec.loader.exec_module(status)

    names = [name for name, _, _ in status.COVERAGE]
    entry = [row for row in status.COVERAGE if row[0] == "thumbnails"]

    assert "thumbnails" in names
    assert entry[0][1] == build.MANIFEST
    assert entry[0][2] == "build_thumbnails.py"
