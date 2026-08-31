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

    assert outcome["basis"].startswith("full caption match")
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

    assert outcome["basis"].startswith("full caption match")
    assert order(outcome) == sorted(EQUAL)
