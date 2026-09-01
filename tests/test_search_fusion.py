"""Fusion behaviour in search(), where both known defects live.

gradient_cutoff and rrf are correct in isolation; what went wrong was
which array reached them and which score ordered the result. That wiring
is only observable from search(), so these tests run it whole against a
synthetic corpus.
"""
import pytest

from visualmind import retrieval

POOL = ["p1", "p2", "p3", "p4", "p5", "p6"]
OUTSIDE = ["p7", "p8"]
CORPUS = POOL + OUTSIDE

# Outside the filter, high enough to dominate if the pool is ever ignored.
BEYOND = {"p7": 0.99, "p8": 0.98}

# Steep first drop then a plateau: gradient_cutoff stops at 2.
PLATEAU = {"p1": 0.90, "p2": 0.50, "p3": 0.49,
           "p4": 0.48, "p5": 0.47, "p6": 0.46}

# Even decline, no plateau: gradient_cutoff runs to the end of the pool.
DECLINE = {"p1": 0.90, "p2": 0.80, "p3": 0.70,
           "p4": 0.60, "p5": 0.50, "p6": 0.10}


def test_image_cutoff_comes_from_image_scores_under_a_filter(fake_corpus):
    """S-1: caption scores plateau, image scores do not.

    Pre-fix both cutoffs were computed from the caption subset, so
    img_cut collapsed onto cap_cut and the pool was truncated to 2.
    """
    fake_corpus(
        CORPUS,
        image={**DECLINE, **BEYOND},
        caption={**PLATEAU, **BEYOND},
        allowed=POOL,
    )

    outcome = retrieval.search("zebra", persons=["Ada Fixture"])

    assert outcome["cap_cut"] == 2
    assert outcome["img_cut"] == 6
    assert len(outcome["results"]) == 6


def test_caption_cutoff_comes_from_caption_scores_under_a_filter(fake_corpus):
    """S-2: the same corpus with the modalities swapped.

    The returned count is 6 either way here, so only the reported
    img_cut separates the fix from a version that hardcodes the other
    modality. Assert the cutoffs, not just the count.
    """
    fake_corpus(
        CORPUS,
        image={**PLATEAU, **BEYOND},
        caption={**DECLINE, **BEYOND},
        allowed=POOL,
    )

    outcome = retrieval.search("zebra", persons=["Ada Fixture"])

    assert outcome["img_cut"] == 2
    assert outcome["cap_cut"] == 6
    assert len(outcome["results"]) == 6


MATCHED = ["m1", "m2", "m3", "m4"]
MATCHED_CORPUS = MATCHED + ["filler"]

# Caption and image rankings are exact inverses of each other, so the
# ordering a mode produces is unambiguous.
MATCHED_CAPTION = {"m1": 0.90, "m2": 0.80, "m3": 0.70,
                   "m4": 0.60, "filler": 0.10}
MATCHED_IMAGE = {"m1": 0.10, "m2": 0.20, "m3": 0.30,
                 "m4": 0.40, "filler": 0.05}
MATCHED_CAPTIONS = {path: "a zebra here" for path in MATCHED}
MATCHED_CAPTIONS["filler"] = "a plain wall"


def order(outcome):
    return [path for path, _ in outcome["results"]]


def test_image_mode_orders_a_matched_set_by_image_score(fake_corpus):
    """S-3: --mode image reached only the fused list before the fix.

    Every caption here contains the query term, so search takes the full
    branch. Pre-fix that branch ranked by caption score whatever the
    mode said, returning m1..m4.
    """
    fake_corpus(
        MATCHED_CORPUS,
        image=MATCHED_IMAGE,
        caption=MATCHED_CAPTION,
        captions=MATCHED_CAPTIONS,
    )

    outcome = retrieval.search("zebra", mode="image")

    assert "mentions the query term" in outcome["basis"]
    assert order(outcome) == ["m4", "m3", "m2", "m1"]


@pytest.mark.parametrize("mode", ["hybrid", "caption"])
def test_other_modes_keep_caption_ordering_in_a_matched_set(
    fake_corpus, mode
):
    """S-4: guards the evaluated default.

    Finding 7 in evals/retrieval-evaluation.md measured RRF ordering
    against caption ordering inside a matched set and kept caption
    order. This fails if someone later makes hybrid fuse here.
    """
    fake_corpus(
        MATCHED_CORPUS,
        image=MATCHED_IMAGE,
        caption=MATCHED_CAPTION,
        captions=MATCHED_CAPTIONS,
    )

    outcome = retrieval.search("zebra", mode=mode)

    assert order(outcome) == ["m1", "m2", "m3", "m4"]


TIED = [f"img{number:02d}" for number in range(1, 17)]


def test_tied_fused_scores_break_by_path(fake_corpus):
    """S-5: image and caption rankings are exact inverses.

    img01 ranks 1st by image and 16th by caption; img16 is the mirror,
    so their RRF sums are equal - and so are seven other pairs. Before
    the tie-break, order came from set iteration and varied with
    PYTHONHASHSEED. Eight independent pairs are kept so that a
    regression removing the tie-break fails at 255 runs in 256 rather
    than on a coin flip.
    """
    fake_corpus(
        TIED,
        image={path: 1.0 - 0.01 * i for i, path in enumerate(TIED)},
        caption={path: 0.01 * (i + 1) for i, path in enumerate(TIED)},
    )

    outcome = retrieval.search("zebra")
    score = dict(outcome["results"])
    position = {path: i for i, path in enumerate(order(outcome))}

    for i in range(8):
        low, high = sorted((TIED[i], TIED[15 - i]))

        assert score[low] == score[high], "fixture no longer ties"
        assert position[low] < position[high]


EQUAL = ["q1", "q2", "q3", "q4", "q5", "q6"]


def test_tied_matched_set_breaks_by_path(fake_corpus):
    """S-6: six full matches sharing one caption score.

    semantic_order receives list(full) from a set, so before the
    tie-break the output order was whatever that set happened to give.
    """
    fake_corpus(
        EQUAL,
        image={path: 0.5 for path in EQUAL},
        caption={path: 0.77 for path in EQUAL},
        captions={path: "a zebra here" for path in EQUAL},
    )

    outcome = retrieval.search("zebra")

    assert "mentions the query term" in outcome["basis"]
    assert order(outcome) == sorted(EQUAL)


# --- score_kind: which scale results[i][1] is on -----------------------
#
# The three scales are an order of magnitude apart and nothing in the
# value distinguishes them, so the branch that produced a result has to
# say. In a matched set the kind follows --mode, because match_score is
# mode-selected.


def test_full_match_in_hybrid_reports_a_caption_cosine(fake_corpus):
    fake_corpus(
        MATCHED_CORPUS,
        image=MATCHED_IMAGE,
        caption=MATCHED_CAPTION,
        captions=MATCHED_CAPTIONS,
    )

    outcome = retrieval.search("zebra")

    assert "mentions the query term" in outcome["basis"]
    assert outcome["score_kind"] == retrieval.SCORE_CAPTION
    assert dict(outcome["results"])["m1"] == MATCHED_CAPTION["m1"]


def test_full_match_under_image_mode_reports_an_image_cosine(fake_corpus):
    """The same branch and the same corpus, a different scale.

    Nothing but score_kind separates these two outcomes, which is the
    reason it exists.
    """
    fake_corpus(
        MATCHED_CORPUS,
        image=MATCHED_IMAGE,
        caption=MATCHED_CAPTION,
        captions=MATCHED_CAPTIONS,
    )

    outcome = retrieval.search("zebra", mode="image")

    assert "mentions the query term" in outcome["basis"]
    assert outcome["score_kind"] == retrieval.SCORE_IMAGE
    assert dict(outcome["results"])["m1"] == MATCHED_IMAGE["m1"]


def test_top_k_reports_a_fused_sum(fake_corpus):
    fake_corpus(
        MATCHED_CORPUS,
        image=MATCHED_IMAGE,
        caption=MATCHED_CAPTION,
        captions=MATCHED_CAPTIONS,
    )

    outcome = retrieval.search("zebra", top_k=3)

    assert "fixed number of results" in outcome["basis"]
    assert outcome["score_kind"] == retrieval.SCORE_FUSED

    # An RRF sum of two 1/(60+rank) terms cannot reach a cosine's range.
    assert all(score < 0.05 for _, score in outcome["results"])


def test_gradient_fallback_reports_a_fused_sum(fake_corpus):
    fake_corpus(CORPUS, image={**DECLINE, **BEYOND},
                caption={**PLATEAU, **BEYOND})

    outcome = retrieval.search("zebra")

    assert "similarity gradient" in outcome["basis"]
    assert outcome["score_kind"] == retrieval.SCORE_FUSED


def test_filter_only_query_reports_no_scale(fake_corpus):
    """Beyond the four branches above: empty_result's placeholder zeros.

    They are not a measurement, so naming them one would be worse than
    saying nothing.
    """
    fake_corpus(CORPUS, image={**DECLINE, **BEYOND},
                caption={**PLATEAU, **BEYOND}, allowed=POOL)

    outcome = retrieval.search("", persons=["Ada Fixture"])

    assert outcome["score_kind"] == retrieval.SCORE_NONE
    assert all(score == 0.0 for _, score in outcome["results"])


def test_the_basis_is_grammatical_for_a_single_term(fake_corpus):
    """The one-term case is the common one, and it used to read badly.

    "full caption match - 4 of 4 captions contain all 1 term" was written
    for a console, where the numbers read as precision. The frontend sets
    this string as its headline, where "all 1 term" reads as unfinished.
    """
    fake_corpus(
        MATCHED_CORPUS,
        image=MATCHED_IMAGE,
        caption=MATCHED_CAPTION,
        captions=MATCHED_CAPTIONS,
    )

    basis = retrieval.search("zebra")["basis"]

    assert basis == "Every caption here mentions the query term."
    assert "all 1" not in basis


def test_the_basis_counts_terms_when_there_is_more_than_one(fake_corpus):
    """Two terms take the plural branch, so the number is worth printing."""
    fake_corpus(
        MATCHED_CORPUS,
        image=MATCHED_IMAGE,
        caption=MATCHED_CAPTION,
        captions=MATCHED_CAPTIONS,
    )

    basis = retrieval.search("zebra here")["basis"]

    assert basis == "Every caption here mentions all 2 query terms."


def test_no_basis_carries_the_result_count(fake_corpus):
    """The count is printed by every consumer separately.

    search_hybrid.py has its own returning line, the gallery prints one,
    and the frontend sets one above the headline. A count inside the
    sentence is a second copy that can disagree with all three.
    """
    fake_corpus(
        MATCHED_CORPUS,
        image=MATCHED_IMAGE,
        caption=MATCHED_CAPTION,
        captions=MATCHED_CAPTIONS,
    )

    basis = retrieval.search("zebra")["basis"]

    assert str(len(MATCHED_CORPUS)) not in basis
    assert "4 of" not in basis


def test_every_basis_reads_as_a_sentence(fake_corpus):
    """One voice, rendered at 30px by one of the three consumers."""
    fake_corpus(
        MATCHED_CORPUS,
        image=MATCHED_IMAGE,
        caption=MATCHED_CAPTION,
        captions=MATCHED_CAPTIONS,
    )

    for outcome in (
        retrieval.search("zebra"),
        retrieval.search("zebra", top_k=3),
        retrieval.search("nothing here mentions this"),
    ):
        basis = outcome["basis"]

        assert basis[0].isupper(), basis
        assert basis.endswith("."), basis
        # The hyphen was doing a colon's work in every branch.
        assert " - " not in basis, basis


def test_each_branch_reports_its_own_identifier(fake_corpus):
    """basis_kind is assigned beside basis, so the two cannot disagree.

    A sentence saying one thing while the token says another would be
    worse than having no token: consumers would split, some reading the
    prose and some the identifier.
    """
    fake_corpus(
        MATCHED_CORPUS,
        image=MATCHED_IMAGE,
        caption=MATCHED_CAPTION,
        captions=MATCHED_CAPTIONS,
    )

    full = retrieval.search("zebra")
    assert full["basis_kind"] == retrieval.BASIS_FULL
    assert "mentions the query term" in full["basis"]

    cut = retrieval.search("zebra", top_k=3)
    assert cut["basis_kind"] == retrieval.BASIS_TOP_K
    assert "fixed number of results" in cut["basis"]

    fallback = retrieval.search("nothing here mentions this")
    assert fallback["basis_kind"] == retrieval.BASIS_GRADIENT
    assert "similarity gradient" in fallback["basis"]

    nothing = retrieval.search("")
    assert nothing["basis_kind"] == retrieval.BASIS_NO_QUERY
    assert "Nothing was asked for" in nothing["basis"]


def test_every_outcome_carries_a_known_kind(fake_corpus):
    """Including the empty-result path, which returns from elsewhere."""
    fake_corpus(
        MATCHED_CORPUS,
        image=MATCHED_IMAGE,
        caption=MATCHED_CAPTION,
        captions=MATCHED_CAPTIONS,
    )

    for outcome in (
        retrieval.search("zebra"),
        retrieval.search("zebra", top_k=3),
        retrieval.search("nothing here mentions this"),
        retrieval.search(""),
    ):
        assert outcome["basis_kind"] in retrieval.BASIS_KINDS
