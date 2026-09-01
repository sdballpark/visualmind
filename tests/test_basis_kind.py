"""The branch identifier that sits beside the explanation.

`basis` is a sentence for a reader. `basis_kind` is the branch that
produced it, as a token. The split exists because search_gallery.py once
chose its colour with `"gradient" in outcome["basis"]`: correct only
while the sentence happened to contain that word, and silently wrong the
moment the prose was rewritten - with no test to catch it.

So the rule this file enforces is not "basis_kind exists" but "nothing
decides anything by reading basis". A consumer may render the sentence;
it may not branch on it.
"""
import re
from pathlib import Path

import pytest

from visualmind import retrieval

ROOT = Path(__file__).resolve().parents[1]

# Everything that receives an outcome and does something with it.
CONSUMERS = [
    ROOT / "src" / "visualmind" / "api.py",
    ROOT / "scripts" / "search_gallery.py",
    ROOT / "scripts" / "search_hybrid.py",
    ROOT / "frontend" / "src" / "App.tsx",
    ROOT / "frontend" / "src" / "useCollection.ts",
    ROOT / "frontend" / "src" / "searchMemory.ts",
    ROOT / "frontend" / "src" / "PhotoPage.tsx",
]

# Ways a consumer could reach into the prose to make a decision. Reading
# basis to print it is fine; testing it is not.
INSPECTIONS = [
    # Python: "gradient" in outcome["basis"], basis.startswith(...)
    re.compile(r'in\s+outcome\[["\']basis["\']\]'),
    re.compile(r'\bbasis\b\s*(==|!=)'),
    re.compile(r'outcome\[["\']basis["\']\]\s*(==|!=)'),
    re.compile(r'\bbasis\b\.\s*(startswith|endswith|find|lower|upper|split)'),
    # TypeScript: basis.includes(...), basis.startsWith(...)
    re.compile(r'\bbasis\b\.\s*(includes|startsWith|endsWith|indexOf|match)'),
    re.compile(r'\.basis\s*(===|!==)'),
]


def code_lines(path):
    """Source lines with comment-only lines dropped.

    A comment may name the old sniff - this file's own docstring does -
    without any code depending on it.
    """
    for number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = line.strip()

        if stripped.startswith(("#", "//", "*", "/*")):
            continue

        yield number, line


@pytest.mark.parametrize("path", CONSUMERS, ids=lambda p: p.name)
def test_no_consumer_decides_by_reading_the_basis_text(path):
    """The rule the identifier exists to make enforceable."""
    offenders = [
        str(number) + ": " + line.strip()
        for number, line in code_lines(path)
        for pattern in INSPECTIONS
        if pattern.search(line) and "basis_kind" not in line
    ]

    assert not offenders, (
        path.name + " inspects the basis sentence to make a decision:\n  "
        + "\n  ".join(offenders)
        + "\nBranch on outcome['basis_kind'] instead - the prose is free "
        "to change and the token is not."
    )


def test_the_gallery_branches_on_the_identifier():
    """The specific case that motivated this, pinned by name."""
    source = (ROOT / "scripts" / "search_gallery.py").read_text(
        encoding="utf-8"
    )

    assert "retrieval.BASIS_GRADIENT" in source
    assert '"gradient" in outcome["basis"]' not in source


def test_every_kind_is_a_known_identifier():
    """A typo in a branch would otherwise ship as a new kind."""
    assert retrieval.BASIS_KINDS == {
        "full_match",
        "partial_match",
        "top_k",
        "gradient",
        "filter_only",
        "no_query",
        "empty_pool",
    }


def test_the_identifiers_are_distinct():
    """Two branches sharing a token would be indistinguishable."""
    named = [
        retrieval.BASIS_FULL,
        retrieval.BASIS_PARTIAL,
        retrieval.BASIS_TOP_K,
        retrieval.BASIS_GRADIENT,
        retrieval.BASIS_FILTER_ONLY,
        retrieval.BASIS_NO_QUERY,
        retrieval.BASIS_EMPTY_POOL,
    ]

    assert len(set(named)) == len(named)
    assert set(named) == retrieval.BASIS_KINDS


def test_the_identifiers_carry_no_wording():
    """Named for the mechanism, so a rewrite cannot reach them.

    A token containing words from its own sentence would be the same
    coupling in a different place.
    """
    for kind in retrieval.BASIS_KINDS:
        assert kind == kind.lower()
        assert " " not in kind
        assert "caption" not in kind
        assert "mention" not in kind
