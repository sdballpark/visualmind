"""Unit tests for the retrieval helpers that take no I/O."""
from visualmind import retrieval


def test_equal_scores_order_independently_of_input_order():
    """SO-5: the same paths and scores, handed over in two orders.

    This is the unit-level form of the tie-break defect. search() feeds
    semantic_order a list built from a set, so 'the order the caller
    passed' is not a stable thing to inherit; the sort tie-breaks on
    path instead.
    """
    scores = {"a": 1.0, "b": 1.0, "c": 1.0}

    forward, _ = retrieval.semantic_order(["a", "b", "c"], scores, 0.8, False)
    backward, _ = retrieval.semantic_order(["c", "b", "a"], scores, 0.8, False)

    assert forward == backward


def lookup(*captions):
    return [
        {
            "source_path": "p" + str(i + 1),
            "filename": "p" + str(i + 1) + ".jpg",
            "caption": caption,
        }
        for i, caption in enumerate(captions)
    ]


ROWS = lookup("a zebra here", "a zebra there", "a plain wall")


def test_allowed_none_scans_every_row():
    """T-3a: None means no filter was requested."""
    hits, terms = retrieval.term_hits("zebra", ROWS, None)

    assert terms == 1
    assert hits == {"p1": 1, "p2": 1}


def test_allowed_empty_set_scans_nothing():
    """T-3b: an empty set means nobody matched, which is not the same.

    Collapsing these two would make every filter silently no-op and
    return the whole corpus.
    """
    hits, terms = retrieval.term_hits("zebra", ROWS, set())

    assert terms == 1
    assert hits == {}


def test_allowed_subset_scans_only_that_subset():
    """T-3c: a populated filter restricts the scan to its own paths."""
    hits, terms = retrieval.term_hits("zebra", ROWS, {"p2"})

    assert terms == 1
    assert hits == {"p2": 1}
