"""Resolve event names to the images they contain.

Events come from data/metadata/events.csv, built by build_events.py from
EXIF capture time. Like person labels, they are a local lookup no model
has seen, so callers pass them explicitly rather than embedding them in
a query string.

An event can be named three ways, all matched leniently:

    event-042                  the generated id
    "Happy Birthday Lisa"      the borrowed email subject
    2024-06                    a date prefix, matching every event
                               starting in that month

Ambiguous matches raise rather than guessing.
"""
import csv
from collections import defaultdict
from pathlib import Path

EVENTS = Path("data/metadata/events.csv")


class AmbiguousEvent(Exception):
    def __init__(self, query, candidates):
        self.query = query
        self.candidates = candidates
        shown = candidates[:6]
        more = "" if len(candidates) <= 6 else (
            " and " + str(len(candidates) - 6) + " more")
        super().__init__(
            "'" + query + "' matches " + str(len(candidates))
            + " events: " + ", ".join(shown) + more
        )


class UnknownEvent(Exception):
    def __init__(self, query):
        self.query = query
        super().__init__("no event matching '" + query + "'")


# Where images go when they have no capture time and no unambiguous
# thread to place them by. It is a bucket, not an occasion, and callers
# that ask "which event is this in" have to say so rather than matching
# the string themselves.
UNASSIGNED = "unassigned"


def available():
    return EVENTS.exists()


def index():
    """Map each event id to its metadata and member image paths."""
    if not available():
        return {}

    events = {}
    members = defaultdict(set)

    with EVENTS.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            event_id = row["event_id"]

            members[event_id].add(row["source_path"])

            if event_id not in events:
                events[event_id] = {
                    "id": event_id,
                    "name": row["event_name"],
                    "start": row["event_start"],
                    "end": row["event_end"],
                }

    for event_id, entry in events.items():
        entry["paths"] = members[event_id]
        entry["images"] = len(members[event_id])

    return events


def roster():
    """Events sorted by start date, newest first, unassigned last."""
    entries = list(index().values())

    return sorted(
        entries,
        key=lambda e: (e["id"] == UNASSIGNED, e["start"] or ""),
        reverse=True,
    )


def resolve(query, events):
    """Match a typed event reference against known events."""
    lowered = query.strip().lower()

    if lowered in events:
        return lowered

    exact = [
        eid for eid, e in events.items()
        if e["name"].lower() == lowered
    ]

    if len(exact) == 1:
        return exact[0]

    candidates = [
        eid for eid, e in events.items()
        if lowered in e["name"].lower()
        or (e["start"] or "").startswith(lowered)
        or lowered in eid
    ]

    if not candidates:
        raise UnknownEvent(query)

    if len(candidates) > 1:
        raise AmbiguousEvent(
            query,
            sorted(
                events[eid]["id"] + " (" + events[eid]["name"] + ")"
                for eid in candidates
            ),
        )

    return candidates[0]


def filter_paths(names):
    """Images belonging to any of the named events.

    Events are a union, not an intersection: an image belongs to exactly
    one event, so requiring membership of two would always return
    nothing. Returns (paths, resolved, counts); None paths means no
    event filter was requested.
    """
    if not names:
        return None, [], {}

    events = index()

    if not events:
        raise UnknownEvent(names[0])

    resolved = [resolve(name, events) for name in names]

    counts = {
        events[eid]["name"]: events[eid]["images"] for eid in resolved
    }

    paths = set()

    for eid in resolved:
        paths |= events[eid]["paths"]

    return paths, [events[eid]["name"] for eid in resolved], counts
