"""Hue and lightness extraction, on images built in the test.

Synthetic rather than sampled: a flat colour has one right answer, and
the interesting cases - an image that is mostly grey, one with no hue at
all - are hard to find in a corpus and trivial to construct.
"""
import importlib.util
from pathlib import Path

import pytest
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_palette.py"

_spec = importlib.util.spec_from_file_location("build_palette", SCRIPT)
build = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build)

SIZE = (80, 80)


def flat(colour):
    return Image.new("RGB", SIZE, colour)


def circular_gap(first, second):
    return abs((first - second + 180.0) % 360.0 - 180.0)


# --- flat colour -----------------------------------------------------


@pytest.mark.parametrize("colour,expected", [
    ((255, 0, 0), 0.0),
    ((255, 255, 0), 60.0),
    ((0, 255, 0), 120.0),
    ((0, 255, 255), 180.0),
    ((0, 0, 255), 240.0),
    ((255, 0, 255), 300.0),
])
def test_a_flat_colour_reports_its_own_hue(colour, expected):
    hue, _ = build.extract(flat(colour))

    assert circular_gap(hue, expected) < 0.5


def test_hue_is_a_real_angle_not_a_bucket():
    """The builder must not quantise; the frontend picks the palette.

    Two colours inside one 10-degree histogram bucket have to come out
    as different numbers, or the bucket has reached the output.
    """
    first, _ = build.extract(flat((255, 40, 0)))
    second, _ = build.extract(flat((255, 60, 0)))

    assert first != second
    assert 0 < first < second < 30


# --- a saturated region inside a grey image --------------------------


def test_one_saturated_region_beats_a_grey_majority():
    """The case a mean colour gets wrong.

    Seven eighths of this image is mid grey. The mean is essentially
    grey with an arbitrary hue angle; the dominant hue is the blue that
    is actually in the picture.
    """
    image = flat((128, 128, 128))
    image.paste(Image.new("RGB", (80, 10), (0, 90, 220)), (0, 0))

    hue, _ = build.extract(image)

    assert circular_gap(hue, 205.0) < 15.0


def test_near_black_pixels_do_not_vote():
    """Their hue is noise: a channel or two apart swings it anywhere."""
    image = Image.new("RGB", SIZE, (4, 0, 9))
    image.paste(Image.new("RGB", (80, 16), (0, 200, 60)), (0, 0))

    hue, _ = build.extract(image)

    assert circular_gap(hue, 138.0) < 15.0


# --- no hue at all ---------------------------------------------------


@pytest.mark.parametrize("colour", [(0, 0, 0), (128, 128, 128),
                                    (255, 255, 255)])
def test_an_image_with_no_colour_reports_no_hue(colour):
    """None is an answer, not a failure.

    An all-black frame has no dominant hue to report. The row is still
    status ok, because the builder read it and measured it correctly -
    calling it a failure would put it in status.py's unreadable count.
    """
    hue, lightness = build.extract(flat(colour))

    assert hue is None
    assert lightness is not None


def test_black_and_white_anchor_the_lightness_scale():
    assert build.extract(flat((0, 0, 0)))[1] == 0.0
    assert build.extract(flat((255, 255, 255)))[1] == pytest.approx(1.0)


# --- lightness is independent of hue ---------------------------------


@pytest.mark.parametrize("bright,dark", [
    ((255, 0, 0), (96, 0, 0)),
    ((0, 255, 0), (0, 96, 0)),
    ((0, 0, 255), (0, 0, 96)),
])
def test_the_same_hue_at_two_brightnesses_keeps_one_hue(bright, dark):
    """Hue and lightness are separate axes.

    The strip draws hue as colour and lightness as mark height, so a
    dark flash photo and a bright beach shot have to differ in height
    even when they agree in colour.
    """
    bright_hue, bright_light = build.extract(flat(bright))
    dark_hue, dark_light = build.extract(flat(dark))

    assert circular_gap(bright_hue, dark_hue) < 0.5
    assert dark_light < bright_light - 0.2


def test_two_hues_at_one_lightness_are_told_apart_by_hue_alone():
    """The mirror case: colour differs, height should not have to.

    The pair is luminance-matched deliberately. Equal-looking colours
    are not equally light - green carries 0.72 of relative luminance
    and blue 0.07 - so a naive pair like teal against purple differs in
    L* by 0.18 and would prove nothing about independence.
    """
    first_hue, first_light = build.extract(flat((65, 0, 0)))
    second_hue, second_light = build.extract(flat((0, 0, 110)))

    assert circular_gap(first_hue, second_hue) > 90.0
    assert abs(first_light - second_light) < 0.01


# --- wraparound ------------------------------------------------------


def test_reds_either_side_of_zero_average_to_red_not_cyan():
    """A plain mean of 359 and 1 is 180, which is the wrong colour."""
    image = Image.new("RGB", SIZE, (255, 0, 6))
    image.paste(Image.new("RGB", (80, 40), (255, 6, 0)), (0, 0))

    hue, _ = build.extract(image)

    assert circular_gap(hue, 0.0) < 10.0


# --- row shape -------------------------------------------------------


def test_a_hueless_row_leaves_the_column_empty_not_zero():
    """Zero degrees is red. Empty is the absence of an answer."""
    row = build.row_for("abc", None, 0.5, "ok")

    assert row["hue"] == ""
    assert row["lightness"] == "0.50000"
    assert list(row) == build.FIELDNAMES


def test_an_unreadable_row_carries_neither_measurement():
    row = build.row_for("abc", None, None, "unreadable")

    assert row["hue"] == ""
    assert row["lightness"] == ""
    assert row["status"] == "unreadable"
