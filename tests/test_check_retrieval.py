"""The comparison behind scripts/check_retrieval.py.

The script needs a GPU to gather results, but the logic deciding whether
anything moved does not - and that logic is what stands between a
regression and a silent "ok", so it is worth pinning here in the fast
suite.
"""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_retrieval.py"

_spec = importlib.util.spec_from_file_location("check_retrieval", SCRIPT)
check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check)


@pytest.fixture
def queries(monkeypatch):
    def use(names):
        monkeypatch.setattr(check, "QUERIES", names)

    return use


def entry(count, filenames):
    return {"count": count, "filenames": sorted(filenames)}


def test_duplicate_filenames_survive_the_delta():
    """Two source paths in this corpus share a filename.

    A set would collapse them and hide the loss of one.
    """
    gained, lost = check.multiset_delta(
        ["a.jpg", "a.jpg", "b.jpg"], ["a.jpg", "b.jpg"]
    )

    assert gained == []
    assert lost == ["a.jpg"]


def test_identical_results_report_no_drift(queries):
    queries(["dog"])
    measured = {"dog": entry(2, ["a.jpg", "b.jpg"])}

    assert check.compare(measured, dict(measured)) == []


def test_membership_change_at_a_steady_count_is_drift(queries):
    """The case a count comparison alone would miss."""
    queries(["dog"])
    measured = {"dog": entry(2, ["a.jpg", "c.jpg"])}
    baseline = {"dog": entry(2, ["a.jpg", "b.jpg"])}

    drift = check.compare(measured, baseline)

    assert len(drift) == 1
    assert drift[0]["count"] == (2, 2)
    assert drift[0]["gained"] == ["c.jpg"]
    assert drift[0]["lost"] == ["b.jpg"]


def test_count_change_reports_direction(queries):
    queries(["dog"])
    measured = {"dog": entry(3, ["a.jpg", "b.jpg", "c.jpg"])}
    baseline = {"dog": entry(1, ["a.jpg"])}

    drift = check.compare(measured, baseline)

    assert drift[0]["count"] == (1, 3)
    assert check.arrow(1, 3) == "1 -> 3  (+2)"
    assert check.arrow(3, 1) == "3 -> 1  (-2)"


def test_a_query_missing_from_the_baseline_is_drift(queries):
    """Adding a query to the suite must not pass silently."""
    queries(["dog", "cat"])
    measured = {"dog": entry(1, ["a.jpg"]), "cat": entry(1, ["b.jpg"])}
    baseline = {"dog": entry(1, ["a.jpg"])}

    drift = check.compare(measured, baseline)

    assert [d["kind"] for d in drift] == ["new"]
    assert drift[0]["query"] == "cat"


def test_a_query_dropped_from_the_suite_is_drift(queries):
    """So does removing one, which would otherwise erase its record."""
    queries(["dog"])
    measured = {"dog": entry(1, ["a.jpg"])}
    baseline = {"dog": entry(1, ["a.jpg"]), "cat": entry(1, ["b.jpg"])}

    drift = check.compare(measured, baseline)

    assert [d["kind"] for d in drift] == ["dropped"]
    assert drift[0]["query"] == "cat"


def test_the_shipped_suite_matches_the_shipped_baseline():
    """The ten queries in the script are the ten in the baseline file.

    Catches a query added to one and not the other.
    """
    import json

    baseline = json.loads(
        (Path(check.__file__).resolve().parents[1]
         / "evals" / "retrieval-baseline.json").read_text(encoding="utf-8")
    )

    assert sorted(baseline["queries"]) == sorted(check.QUERIES)
