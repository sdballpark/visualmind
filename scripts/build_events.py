"""Group photographs into events.

Events are built from capture time, not from when a photo was emailed.
Those differ by decades in this corpus: a 2006 baptism photo arrived in a
2026 message, and 28 of 66 dated email threads span multiple years. Email
threads are retrospective collections - "Re: Happy Birthday Daniel" holds
photos from 2000 to 2023 - so they cannot define an event.

Three passes:

  1. Photos with usable EXIF cluster by time gap. A break longer than
     --gap-hours starts a new event.
  2. Photos without EXIF fall back to their email thread. If every dated
     photo in that thread landed in one event, the undated siblings join
     it. If the thread spans several events, they stay unassigned - a
     wrong date is worse than no date.
  3. Events are named from the email subject when the thread agrees,
     otherwise left with a date label.

Writes data/metadata/events.csv.
"""
import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

CATALOG = Path("data/metadata/image_catalog.csv")
FACE_CLUSTERS = Path("data/metadata/face_clusters.csv")
LABELS = Path("data/metadata/person_labels.json")
OUTPUT = Path("data/metadata/events.csv")

GAP_HOURS = 72

EXIF_FORMATS = ["%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"]

FIELDNAMES = [
    "event_id", "event_name", "event_start", "event_end",
    "image_count", "date_source",
    "source_path", "filename", "capture_time",
    "gmail_subject", "gmail_thread_id",
]


def read_csv(path):
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_exif(value):
    """Parse an EXIF timestamp, rejecting the zero date cameras emit."""
    text = (value or "").strip()

    if not text or text.startswith("0000"):
        return None

    for fmt in EXIF_FORMATS:
        try:
            parsed = datetime.strptime(text, fmt)
        except ValueError:
            continue

        # Digital photographs predate neither 1990 nor the present.
        if 1990 <= parsed.year <= datetime.now().year:
            return parsed

    return None


def clean_subject(subject):
    text = (subject or "").strip()

    if not text:
        return ""

    text = re.sub(r"^((re|fwd|fw)\s*:\s*)+", "", text, flags=re.I)

    return text.strip()


def label_for(rows, start, end):
    """Name an event from its email subjects when they agree."""
    subjects = Counter(
        clean_subject(r.get("gmail_subject"))
        for r in rows
        if clean_subject(r.get("gmail_subject"))
    )

    span = start.strftime("%b %Y")

    if start.date() != end.date():
        if start.strftime("%Y-%m") == end.strftime("%Y-%m"):
            span = start.strftime("%b %Y")
        else:
            span = start.strftime("%b %Y") + " - " + end.strftime("%b %Y")

    if not subjects:
        return span

    top, count = subjects.most_common(1)[0]

    # Only borrow a subject when most of the event shares it.
    if count >= max(2, len(rows) * 0.6):
        return top + " (" + span + ")"

    return span


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gap-hours", type=float, default=GAP_HOURS)
    parser.add_argument("--show", type=int, default=25)
    args = parser.parse_args()

    rows = read_csv(CATALOG)

    dated = []
    undated = []

    for row in rows:
        when = parse_exif(row.get("best_exif_date"))

        if when:
            dated.append((when, row))
        else:
            undated.append(row)

    dated.sort(key=lambda pair: pair[0])

    print()
    print("=" * 76)
    print("VISUALMIND - EVENT GROUPING")
    print("=" * 76)
    print("Catalog images:   " + str(len(rows)))
    print("With capture time:" + str(len(dated)).rjust(5))
    print("Without:          " + str(len(undated)).rjust(5))
    print("Gap threshold:    " + str(args.gap_hours) + " hours")

    # Pass 1 - time gap clustering over dated photos.
    events = []
    current = []
    previous = None

    for when, row in dated:
        if previous is None or (when - previous).total_seconds() <= (
            args.gap_hours * 3600
        ):
            current.append((when, row))
        else:
            events.append(current)
            current = [(when, row)]

        previous = when

    if current:
        events.append(current)

    print("Events from EXIF: " + str(len(events)).rjust(5))

    # Pass 2 - place undated photos via their email thread.
    event_of_path = {}

    for index, group in enumerate(events):
        for _when, row in group:
            event_of_path[row["source_path"]] = index

    thread_events = defaultdict(set)

    for row in rows:
        thread = (row.get("gmail_thread_id") or "").strip()
        index = event_of_path.get(row["source_path"])

        if thread and index is not None:
            thread_events[thread].add(index)

    placed = 0
    unplaced = []

    extra = defaultdict(list)

    for row in undated:
        thread = (row.get("gmail_thread_id") or "").strip()
        candidates = thread_events.get(thread, set())

        if len(candidates) == 1:
            extra[next(iter(candidates))].append(row)
            placed += 1
        else:
            unplaced.append(row)

    print("Placed by thread: " + str(placed).rjust(5))
    print("Unassigned:       " + str(len(unplaced)).rjust(5))

    # Assemble output.
    out_rows = []
    summaries = []

    for index, group in enumerate(events, start=1):
        times = [when for when, _row in group]
        start, end = min(times), max(times)

        members = [row for _when, row in group]
        borrowed = extra.get(index - 1, [])

        name = label_for(members, start, end)
        event_id = "event-" + str(index).zfill(3)
        total = len(members) + len(borrowed)

        summaries.append((event_id, name, start, end, total, len(borrowed)))

        for when, row in group:
            out_rows.append({
                "event_id": event_id,
                "event_name": name,
                "event_start": start.isoformat(sep=" "),
                "event_end": end.isoformat(sep=" "),
                "image_count": str(total),
                "date_source": "exif",
                "source_path": row["source_path"],
                "filename": row["filename"],
                "capture_time": when.isoformat(sep=" "),
                "gmail_subject": row.get("gmail_subject", ""),
                "gmail_thread_id": row.get("gmail_thread_id", ""),
            })

        for row in borrowed:
            out_rows.append({
                "event_id": event_id,
                "event_name": name,
                "event_start": start.isoformat(sep=" "),
                "event_end": end.isoformat(sep=" "),
                "image_count": str(total),
                "date_source": "thread",
                "source_path": row["source_path"],
                "filename": row["filename"],
                "capture_time": "",
                "gmail_subject": row.get("gmail_subject", ""),
                "gmail_thread_id": row.get("gmail_thread_id", ""),
            })

    for row in unplaced:
        out_rows.append({
            "event_id": "unassigned",
            "event_name": "unassigned",
            "event_start": "", "event_end": "",
            "image_count": str(len(unplaced)),
            "date_source": "none",
            "source_path": row["source_path"],
            "filename": row["filename"],
            "capture_time": "",
            "gmail_subject": row.get("gmail_subject", ""),
            "gmail_thread_id": row.get("gmail_thread_id", ""),
        })

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(out_rows)

    named = sum(1 for s in summaries if not s[1][0].isdigit()
                and "(" in s[1])

    print()
    print("-" * 76)
    print("SUMMARY")
    print("-" * 76)
    print("Events:           " + str(len(events)))
    print("Named by subject: " + str(named))
    print("Images placed:    " + str(len(out_rows) - len(unplaced))
          + " of " + str(len(rows)))

    print()
    print("-" * 76)
    print("LARGEST EVENTS")
    print("-" * 76)

    for event_id, name, start, end, total, borrowed in sorted(
        summaries, key=lambda s: s[4], reverse=True
    )[:args.show]:
        span = start.strftime("%Y-%m-%d")

        if start.date() != end.date():
            span += " to " + end.strftime("%Y-%m-%d")

        note = ""

        if borrowed:
            note = "  (+" + str(borrowed) + " by thread)"

        print(str(total).rjust(4) + "  " + span.ljust(26)
              + name[:40] + note)

    print()
    print("Events: " + str(OUTPUT.resolve()))
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
