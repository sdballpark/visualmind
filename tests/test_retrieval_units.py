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


# --- term matching: the three shapes from finding 6 -------------------


def one(caption):
    return [{"source_path": "p", "filename": "p.jpg", "caption": caption}]


def hits(query, caption):
    found, total = retrieval.term_hits(query, one(caption))
    return found.get("p", 0), total


def test_plural_query_matches_a_singular_caption():
    """The pattern appended an optional "s" to the query term only.

    "cat" found "cats"; "cats" never found "cat". Folding both sides
    makes the comparison symmetric.
    """
    assert hits("cats", "a cat sleeps") == (1, 1)
    assert hits("cat", "two cats sleep") == (1, 1)


def test_accented_query_matches_its_own_caption():
    """[a-z]+ cut "café" down to "caf".

    Matching then failed against the very caption the query was copied
    from, because no word boundary sits between "caf" and the accent.
    """
    assert retrieval.content_terms("café") == ["café"]
    assert hits("café", "a photo at a café") == (1, 1)


def test_digits_survive_tokenising():
    """"4th of july" lost its digit and carried the junk term "th".

    Every content term has to match for a full match, so a caption
    reading exactly "4th of july parade" could not produce one, and the
    query fell through to the gradient branch.
    """
    assert retrieval.content_terms("4th of july") == ["4th", "july"]
    assert hits("4th of july", "4th of july parade") == (2, 2)


def test_word_boundaries_still_hold():
    """Folding must not turn matching into substring search."""
    assert hits("cat", "a catalog on the table") == (0, 1)
    assert hits("car", "a carpet in the hall") == (0, 1)


def test_folding_is_symmetric_even_when_wrong():
    """"christmas" folds to "christma" on both sides, so it matches.

    A fold that is wrong in isolation is harmless while it is applied
    to query and caption alike.
    """
    assert retrieval.content_terms("christmas tree") == ["christma", "tree"]
    assert hits("christmas tree", "a christmas tree indoors") == (2, 2)


def test_double_s_words_are_not_stripped():
    """The "ss" guard keeps "dress" and "glass" intact."""
    assert retrieval.singular("dress") == "dress"
    assert retrieval.singular("glass") == "glass"
    assert retrieval.singular("glasses") == "glass"
