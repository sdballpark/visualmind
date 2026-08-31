"""combine_filters composes the person and event filters.

None and the empty set carry different meanings here and the whole
filter contract rests on the difference, so each case asserts which one
came back.
"""
from visualmind import retrieval


def test_no_filter_requested_returns_none(label_files):
    """C-1: None for paths, so search() treats every image as permitted."""
    allowed, people_names, person_counts, event_names, event_counts = (
        retrieval.combine_filters([], [])
    )

    assert allowed is None
    assert people_names == []
    assert person_counts == {}
    assert event_names == []
    assert event_counts == {}


def test_person_and_event_intersect(label_files):
    """C-3: Ada is in img1-3, Fixture Picnic covers img1-2.

    A union would return three images here, so this distinguishes them.
    """
    allowed, people_names, _, event_names, _ = retrieval.combine_filters(
        ["Ada Fixture"], ["event-001"]
    )

    assert allowed == {"img1.jpg", "img2.jpg"}
    assert people_names == ["Ada Fixture"]
    assert event_names == ["Fixture Picnic"]


def test_disjoint_person_and_event_return_empty_not_none(label_files):
    """C-4: Cy is in img5, Fixture Picnic covers img1-2.

    Nothing satisfies both. Returning None here would read as 'no filter
    requested' and hand back the entire corpus - a wrong answer that
    looks like a working search.
    """
    allowed, _, _, _, _ = retrieval.combine_filters(
        ["Cy Sample"], ["event-001"]
    )

    assert allowed == set()
    assert allowed is not None
