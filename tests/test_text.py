"""Agreement between a count and its noun.

An event holding one photograph rendered as "1 images" on the item page.
The count was composed separately in the item page, the gallery's filter
block and the face contact sheet, each with its own hardcoded "s", so
one fix per site was needed and one site was always going to be missed.
"""
import pytest

from visualmind.text import counted, plural


@pytest.mark.parametrize("count,expected", [
    (0, "images"),
    (1, "image"),
    (2, "images"),
    (33, "images"),
])
def test_only_one_takes_the_singular(count, expected):
    """Zero is plural in English: "0 images", not "0 image"."""
    assert plural(count, "image") == expected


def test_an_irregular_plural_can_be_given():
    assert plural(1, "person", "people") == "person"
    assert plural(3, "person", "people") == "people"


def test_counted_puts_the_number_with_the_noun():
    assert counted(1, "image") == "1 image"
    assert counted(13, "image") == "13 images"


def test_the_noun_is_returned_alone_for_a_split_phrase():
    """"3 of 4 photographs" agrees with the total, not the count.

    The two numbers are not adjacent, which is why plural() returns the
    noun rather than the number with it.
    """
    assert "3 of " + counted(4, "photograph") == "3 of 4 photographs"
    assert "1 of " + counted(1, "photograph") == "1 of 1 photograph"


def test_the_reported_case():
    """The event on the item page, which is where this was seen."""
    assert counted(1, "image") == "1 image"
    assert counted(13, "image") == "13 images"
