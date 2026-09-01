"""Numbers rendered for a reader.

An event holding one photograph rendered as "1 images" on the item page.
The count is composed in several places - the item page, the gallery's
filter block, the face contact sheet - and each had its own concatenation
with a hardcoded "s", so the same bug had to be fixed in each of them.
"""


def plural(count, singular, many=None):
    """The form of `singular` that agrees with `count`.

    Returns the noun alone rather than the number with it, because the
    two are not always adjacent: "3 of 4 photographs" agrees with the
    total, not with the count standing in front of it.
    """
    if many is None:
        many = singular + "s"

    return singular if count == 1 else many


def counted(count, singular, many=None):
    """`count` followed by the noun that agrees with it."""
    return str(count) + " " + plural(count, singular, many)
