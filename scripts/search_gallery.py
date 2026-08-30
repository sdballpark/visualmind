"""Hybrid search with an HTML thumbnail gallery.

Retrieval logic lives in visualmind.retrieval so this and
scripts/search_hybrid.py cannot drift apart. This module only renders.

Output pages embed base64 copies of the source photos and, when a filter
is used, the names of real people. outputs/ is gitignored and blocked by
the pre-commit hook.
"""
import argparse
import base64
import html
import io
import re
import sys
from pathlib import Path

from PIL import Image

from visualmind import events, people, retrieval

OUTPUT_ROOT = Path("outputs/search")
THUMBNAIL = (420, 420)

PAGE_CSS = """
body { font-family: system-ui, sans-serif; margin: 30px; background: #f4f4f4;
       color: #222; }
h1 { margin-bottom: 4px; }
.meta { margin-bottom: 8px; color: #444; font-size: 14px; }
.filter { background: #eef4ff; border-left: 4px solid #2b5fa8;
          padding: 12px 16px; margin: 10px 0 18px; font-size: 13px;
          line-height: 1.6; }
.filter strong { display: block; margin-bottom: 4px; }
.filter .kind { display: inline-block; width: 62px; color: #555; }
.filter .caveat { color: #555; margin-top: 8px; }
.basis { display: inline-block; padding: 5px 11px; border-radius: 4px;
         background: #1f6f3f; color: #fff; font-size: 13px;
         margin: 4px 0 24px; }
.basis.gradient { background: #8a6d1f; }
.basis.warn { background: #8a1f1f; }
.note { background: #fff3f3; border-left: 4px solid #8a1f1f;
        padding: 12px 16px; margin-bottom: 24px; font-size: 13px;
        line-height: 1.5; }
.trimmed { background: #fdf6e3; border-left: 4px solid #8a6d1f;
           padding: 12px 16px; margin-bottom: 24px; font-size: 13px;
           line-height: 1.6; }
.trimmed ul { margin: 6px 0 0; padding-left: 20px; }
.grid { display: grid;
        grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
        gap: 22px; }
.card { background: #fff; border-radius: 10px; overflow: hidden;
        position: relative; box-shadow: 0 2px 8px rgba(0,0,0,.12); }
.card img { display: block; width: 100%; height: 280px; object-fit: contain;
            background: #111; }
.content { padding: 14px 16px 18px; }
.rank { position: absolute; top: 10px; left: 10px; background: #000;
        color: #fff; padding: 5px 9px; border-radius: 6px; font-weight: 600;
        font-size: 13px; }
.terms { position: absolute; top: 10px; right: 10px; background: #1f6f3f;
         color: #fff; padding: 5px 9px; border-radius: 6px; font-size: 12px; }
.name { font-weight: 600; margin-bottom: 6px; word-break: break-all; }
.ranks { font-size: 12px; color: #666; margin-bottom: 10px; }
.caption { font-size: 13px; line-height: 1.45; color: #333; }
"""


def safe_name(text):
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "query"


def thumbnail_data_uri(path):
    try:
        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail(THUMBNAIL)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=82)
    except Exception:
        return ""

    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return "data:image/jpeg;base64," + encoded


def filter_block(outcome):
    if not outcome["people"] and not outcome["events"]:
        return ""

    rows = ""

    for name in outcome["people"]:
        rows += ('<div><span class="kind">person</span>'
                 + html.escape(name) + " - "
                 + str(outcome["person_counts"][name]) + " images</div>")

    for name in outcome["events"]:
        rows += ('<div><span class="kind">event</span>'
                 + html.escape(name) + " - "
                 + str(outcome["event_counts"][name]) + " images</div>")

    caveat = ""

    if outcome["people"]:
        caveat = ('<div class="caveat">A face that was not detected will '
                  + "not match, so the person filter under-reports rather "
                  + "than over-reports.</div>")

    return (
        '<div class="filter"><strong>Filter - searching '
        + str(outcome["pool_size"]) + " of "
        + str(outcome["corpus_size"]) + " images</strong>"
        + rows + caveat + "</div>"
    )


def render(query, outcome, mode, rrf_k):
    by_path = {r["source_path"]: r for r in outcome["caption_lookup"]}
    img_rank = outcome["image_rank"]
    cap_rank = outcome["caption_rank"]
    hits = outcome["hits"]
    matched = outcome["matched"]
    scored = bool(outcome["total_terms"])

    cards = []

    for rank, (path, score) in enumerate(outcome["results"], start=1):
        row = by_path.get(path, {})
        name = row.get("filename", Path(path).name)
        caption = row.get("caption", "")
        thumb = thumbnail_data_uri(Path(path))

        badge = ""

        if path in matched:
            badge = ('<div class="terms">' + str(hits.get(path, 0))
                     + " / " + str(outcome["total_terms"]) + " terms</div>")

        if scored:
            ranks = ('<div class="ranks">score ' + format(score, ".5f")
                     + " &nbsp;|&nbsp; image #" + str(img_rank.get(path, "-"))
                     + " &nbsp;|&nbsp; caption #"
                     + str(cap_rank.get(path, "-")) + "</div>")
        else:
            ranks = '<div class="ranks">catalog order</div>'

        cards.append(
            '<article class="card">'
            + '<div class="rank">#' + str(rank) + "</div>"
            + badge
            + '<img src="' + thumb + '" alt="' + html.escape(name) + '">'
            + '<div class="content">'
            + '<div class="name">' + html.escape(name) + "</div>"
            + ranks
            + '<div class="caption">' + html.escape(caption) + "</div>"
            + "</div></article>"
        )

    if not cards:
        cards.append("<p>No results.</p>")

    basis_class = "basis"

    if outcome["low_confidence"]:
        basis_class += " warn"
    elif "gradient" in outcome["basis"]:
        basis_class += " gradient"

    note = ""

    if outcome["low_confidence"]:
        note = (
            '<div class="note">No caption mentions these terms and neither '
            "score curve flattened. These are nearest neighbours, not "
            "matches. Treat the count as a ceiling, not an answer.</div>"
        )

    trimmed_block = ""

    if outcome["trimmed"]:
        items = "".join(
            "<li>" + html.escape(
                by_path.get(p, {}).get("filename", Path(p).name)
            ) + "</li>"
            for p in outcome["trimmed"]
        )
        trimmed_block = (
            '<div class="trimmed"><strong>'
            + str(len(outcome["trimmed"]))
            + " result(s) trimmed</strong> for trailing the set on caption "
            + "score. These matched every query term:<ul>" + items
            + "</ul></div>"
        )

    if scored:
        img_note = "" if outcome["img_plateau"] else " (ceiling)"
        cap_note = "" if outcome["cap_plateau"] else " (ceiling)"
        meta = ('<div class="meta">mode ' + mode
                + " &nbsp;|&nbsp; RRF k=" + str(rrf_k)
                + " &nbsp;|&nbsp; content terms "
                + str(outcome["total_terms"])
                + " &nbsp;|&nbsp; image cut " + str(outcome["img_cut"])
                + img_note
                + " &nbsp;|&nbsp; caption cut " + str(outcome["cap_cut"])
                + cap_note + "</div>")
    else:
        meta = '<div class="meta">no text query - filter only</div>'

    heading = query

    if not heading:
        parts = outcome["people"] + outcome["events"]
        heading = " / ".join(parts) if parts else "Results"

    return (
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        + "<title>VisualMind - " + html.escape(heading) + "</title>"
        + "<style>" + PAGE_CSS + "</style></head><body>"
        + "<h1>" + html.escape(heading) + "</h1>"
        + meta
        + filter_block(outcome)
        + '<div class="' + basis_class + '">returning '
        + str(len(outcome["results"])) + " - "
        + html.escape(outcome["basis"]) + "</div>"
        + note
        + trimmed_block
        + '<div class="grid">' + "".join(cards) + "</div>"
        + "</body></html>"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", default="")
    parser.add_argument(
        "--person",
        action="append",
        default=[],
        metavar="NAME",
        help="Only images containing this person. Repeat for several.",
    )
    parser.add_argument(
        "--event",
        action="append",
        default=[],
        metavar="REF",
        help=(
            "Only images from this event. Accepts an id, a date prefix, "
            "or a name. Repeat for several."
        ),
    )
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--rrf-k", type=int, default=retrieval.RRF_K)
    parser.add_argument(
        "--gradient-floor",
        type=float,
        default=retrieval.GRADIENT_FLOOR,
    )
    parser.add_argument(
        "--min-partial-terms",
        type=int,
        default=retrieval.MIN_PARTIAL_TERMS,
    )
    parser.add_argument("--trim", action="store_true")
    parser.add_argument(
        "--semantic-drop",
        type=float,
        default=retrieval.SEMANTIC_DROP,
    )
    parser.add_argument(
        "--mode",
        choices=["hybrid", "image", "caption"],
        default="hybrid",
    )
    args = parser.parse_args()

    if not args.query and not args.person and not args.event:
        parser.error("give a query, --person, --event, or a combination")

    try:
        outcome = retrieval.search(
            args.query,
            mode=args.mode,
            top_k=args.top_k,
            rrf_k=args.rrf_k,
            gradient_floor=args.gradient_floor,
            min_partial_terms=args.min_partial_terms,
            semantic_drop=args.semantic_drop,
            trim=args.trim,
            persons=args.person,
            event_names=args.event,
        )
    except (people.AmbiguousName, people.UnknownName,
            events.AmbiguousEvent, events.UnknownEvent) as error:
        print()
        print(str(error))
        print()
        return 1

    page = render(args.query, outcome, args.mode, args.rrf_k)

    stem = safe_name(args.query) if args.query else "filter"

    for value in args.person + args.event:
        stem += "--" + safe_name(value)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_ROOT / (stem[:120] + ".html")
    output_path.write_text(page, encoding="utf-8")

    print()
    print("=" * 72)
    print("VISUALMIND SEARCH GALLERY")
    print("=" * 72)
    print("Query:    " + (args.query or "(none)"))

    if outcome["people"]:
        print("People:   " + ", ".join(outcome["people"]))

    if outcome["events"]:
        print("Events:   " + ", ".join(outcome["events"]))

    if outcome["people"] or outcome["events"]:
        print("Pool:     " + str(outcome["pool_size"]) + " of "
              + str(outcome["corpus_size"]) + " images")

    print("Returned: " + str(len(outcome["results"]))
          + "  (" + outcome["basis"] + ")")
    print("Gallery:  " + str(output_path.resolve()))
    print("=" * 72)
    print()
    print('Open with: explorer.exe "$(wslpath -w '
          + str(output_path) + ')"')

    return 0


if __name__ == "__main__":
    sys.exit(main())
