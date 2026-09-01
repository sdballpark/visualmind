#!/usr/bin/env python3
"""Re-run the published evaluation queries against a recorded baseline.

The nine queries in Finding 6 of evals/retrieval-evaluation.md, plus the
cat control from Finding 3. The stopword list, the tokeniser and the
term-match rule are shared by every query, so a change made for one of
them moves the rest. That is how the published "people wearing
sunglasses" count drifted from 10 to 30 with nobody noticing.

Compares count and membership, not ordering. Ordering has its own tests
in the fast suite, and a deliberate tie-break change should not surface
here as a content regression.

The baseline records filenames only. Captions and source paths describe
private photographs and the pre-commit hook refuses them.

A moved count is a signal, not a verdict. Read the report, decide
whether the move is an improvement, then record it with --update in a
commit of its own.

Exit code is 1 if any query drifted, 2 if the run could not be made.
"""
import argparse
import json
import sys
from pathlib import Path

from visualmind import retrieval

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "evals" / "retrieval-baseline.json"

# Finding 6 lists these nine; "cat" is the absent-concept control from
# Finding 3. Keep them in the document's order so the two read together.
QUERIES = [
    "dog",
    "christmas tree",
    "beach",
    "birthday cake",
    "wedding",
    "swimming pool",
    "people wearing sunglasses",
    "a red car",
    "someone holding a baby",
    "cat",
]

RULE = "=" * 76
THIN = "-" * 76


def measure(query):
    """Count and filenames for one query, at default settings.

    Filenames are not unique - two source paths in this corpus share a
    name - so membership is a sorted list rather than a set, and a swap
    between two identically named files would not be visible here.
    """
    outcome = retrieval.search(query)

    return {
        "count": len(outcome["results"]),
        "filenames": sorted(Path(path).name for path, _ in outcome["results"]),
    }


def collect():
    measured = {}

    for position, query in enumerate(QUERIES, start=1):
        print("  [" + str(position) + "/" + str(len(QUERIES)) + "] " + query,
              flush=True)
        measured[query] = measure(query)

    return measured


def multiset_delta(before, after):
    """What `after` gained and lost relative to `before`, duplicates kept."""
    gained = list(after)
    lost = []

    for name in before:
        if name in gained:
            gained.remove(name)
        else:
            lost.append(name)

    return gained, lost


def compare(measured, baseline):
    """One record per query whose count or membership moved."""
    drift = []

    for query in QUERIES:
        now = measured[query]
        was = baseline.get(query)

        if was is None:
            drift.append({
                "query": query,
                "kind": "new",
                "count": (None, now["count"]),
                "gained": now["filenames"],
                "lost": [],
            })
            continue

        gained, lost = multiset_delta(was["filenames"], now["filenames"])

        if was["count"] == now["count"] and not gained and not lost:
            continue

        drift.append({
            "query": query,
            "kind": "moved",
            "count": (was["count"], now["count"]),
            "gained": gained,
            "lost": lost,
        })

    for query in baseline:
        if query not in QUERIES:
            drift.append({
                "query": query,
                "kind": "dropped",
                "count": (baseline[query]["count"], None),
                "gained": [],
                "lost": baseline[query]["filenames"],
            })

    return drift


def arrow(before, after):
    if before is None:
        return "absent -> " + str(after)

    if after is None:
        return str(before) + " -> no longer checked"

    move = after - before
    sign = "+" if move > 0 else ""

    return str(before) + " -> " + str(after) + "  (" + sign + str(move) + ")"


def name_list(names):
    if not names:
        return "none"

    return ", ".join(names)


def report(measured, baseline, drift):
    print()
    print(RULE)
    print("RETRIEVAL REGRESSION - " + str(len(QUERIES)) + " queries")
    print(RULE)
    print()

    moved = {entry["query"] for entry in drift}

    for query in QUERIES:
        mark = "DRIFTED" if query in moved else "ok"
        print("  " + query.ljust(30)
              + str(measured[query]["count"]).rjust(4) + "  " + mark)

    if not drift:
        print()
        print("No change against " + str(BASELINE.relative_to(ROOT)) + ".")
        print()
        return

    print()
    print(THIN)
    print("DRIFT")
    print(THIN)

    for entry in drift:
        print()
        print(entry["query"])
        print("  count    " + arrow(*entry["count"]))
        print("  gained   " + str(len(entry["gained"])).rjust(3) + "  "
              + name_list(entry["gained"]))
        print("  lost     " + str(len(entry["lost"])).rjust(3) + "  "
              + name_list(entry["lost"]))

    net = sum(
        (entry["count"][1] or 0) - (entry["count"][0] or 0)
        for entry in drift
    )
    direction = "+" if net > 0 else ""

    print()
    print(THIN)
    print(str(len(drift)) + " of " + str(len(QUERIES))
          + " queries moved, " + direction + str(net) + " images overall.")
    print(THIN)
    print()
    print("A moved count is a signal, not a verdict. Relabel the affected")
    print("queries by hand, update the tables in")
    print("evals/retrieval-evaluation.md, then record the new baseline in a")
    print("commit of its own:")
    print()
    print("  python scripts/check_retrieval.py --update")
    print()


def write_baseline(measured):
    BASELINE.write_text(
        json.dumps(
            {
                "note": (
                    "Counts and filenames for the queries published in "
                    "evals/retrieval-evaluation.md. Regenerate with "
                    "scripts/check_retrieval.py --update, deliberately."
                ),
                "queries": measured,
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--update",
        action="store_true",
        help="Record the current results as the new baseline.",
    )
    args = parser.parse_args()

    if not BASELINE.exists() and not args.update:
        print()
        print("No baseline at " + str(BASELINE.relative_to(ROOT)) + ".")
        print("Create one with: python scripts/check_retrieval.py --update")
        print()
        return 2

    print()
    print("Running " + str(len(QUERIES)) + " queries "
          + "(two model loads each, so this is slow)...")
    print()

    try:
        measured = collect()
    except retrieval.IndexMismatch as error:
        print()
        print(str(error))
        print()
        return 2

    if args.update:
        existing = {}

        if BASELINE.exists():
            existing = json.loads(
                BASELINE.read_text(encoding="utf-8")
            )["queries"]
            report(measured, existing, compare(measured, existing))

        write_baseline(measured)
        print("Baseline written to " + str(BASELINE.relative_to(ROOT)) + ".")
        print()
        return 0

    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))["queries"]
    drift = compare(measured, baseline)
    report(measured, baseline, drift)

    return 1 if drift else 0


if __name__ == "__main__":
    sys.exit(main())
