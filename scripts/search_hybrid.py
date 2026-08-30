"""Hybrid search, console output.

Retrieval logic lives in visualmind.retrieval so this and
scripts/search_gallery.py cannot drift apart. This module only prints.
"""
import argparse
import sys
from pathlib import Path

from visualmind import people, retrieval


def list_people():
    roster = people.roster()

    if not roster:
        print("No labelled people. Run cluster_faces.py, then "
              "label_faces.py.")
        return 1

    print()
    print("=" * 76)
    print("KNOWN PEOPLE - " + str(len(roster)))
    print("=" * 76)
    print()
    print("NAME".ljust(30) + "IMAGES".rjust(8) + "FACES".rjust(8))
    print("-" * 76)

    for entry in roster:
        print(entry["name"].ljust(30)
              + str(entry["images"]).rjust(8)
              + str(entry["faces"]).rjust(8))

    print()
    print("Filter a search with: --person \"Lisa Bogan\"")
    print("Partial names work when unambiguous: --person lisa")
    print()

    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", default="")
    parser.add_argument(
        "--person",
        action="append",
        default=[],
        metavar="NAME",
        help=(
            "Only images containing this person. Repeat for several - "
            "every named person must be present."
        ),
    )
    parser.add_argument(
        "--list-people",
        action="store_true",
        help="Show known people and exit.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Force a fixed result count. Default 0 derives the count.",
    )
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
    parser.add_argument(
        "--trim",
        action="store_true",
        help=(
            "Discard term-matched results whose caption score trails the "
            "set. Off by default: it cost a true match on the one "
            "relational query tested."
        ),
    )
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
    parser.add_argument(
        "--captions",
        action="store_true",
        help="Print the caption under each result.",
    )
    args = parser.parse_args()

    if args.list_people:
        return list_people()

    if not args.query and not args.person:
        parser.error("give a query, --person, or --list-people")

    print()
    print("=" * 76)
    print("HYBRID SEARCH - " + (args.query or "(no text query)"))
    print("=" * 76)

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
        )
    except people.AmbiguousName as error:
        print()
        print(str(error))
        print()
        print("Use a fuller name.")
        print()
        return 1
    except people.UnknownName as error:
        print()
        print(str(error))
        print()
        print("See --list-people.")
        print()
        return 1

    if outcome["people"]:
        print()
        print("-" * 76)
        print("PERSON FILTER")
        print("-" * 76)

        for name in outcome["people"]:
            print("  " + name.ljust(30)
                  + str(outcome["person_counts"][name]).rjust(4) + " images")

        if len(outcome["people"]) > 1:
            print()
            print("  Together in " + str(outcome["pool_size"]) + " images")

        print()
        print("  Searching " + str(outcome["pool_size"]) + " of "
              + str(outcome["corpus_size"]) + " images")
        print()
        print("  Note: someone whose face was not detected in a photo "
              "will not")
        print("  match, so this filter under-reports rather than over-"
              "reports.")

    img_note = "" if outcome["img_plateau"] else "  (ceiling, no plateau)"
    cap_note = "" if outcome["cap_plateau"] else "  (ceiling, no plateau)"

    print()
    print("-" * 76)
    print("RESULT COUNT")
    print("-" * 76)
    print("Query content terms:  " + str(outcome["total_terms"]))
    print("Full term matches:    " + str(outcome["full_count"]))
    print("Partial matches:      " + str(outcome["partial_count"]))
    print("Image gradient cut:   " + str(outcome["img_cut"]) + img_note)
    print("Caption gradient cut: " + str(outcome["cap_cut"]) + cap_note)
    print("Returning:            " + str(len(outcome["results"]))
          + "  (" + outcome["basis"] + ")")

    if outcome["low_confidence"]:
        print()
        print("LOW CONFIDENCE: no caption mentions these terms and neither")
        print("score curve flattened. Results are nearest neighbours, not")
        print("matches - treat the count as a ceiling, not an answer.")

    by_path = {r["source_path"]: r for r in outcome["caption_lookup"]}
    hits = outcome["hits"]
    matched = outcome["matched"]

    print()

    for rank, (path, score) in enumerate(outcome["results"], start=1):
        row = by_path.get(path, {})
        name = row.get("filename", Path(path).name)

        if path in matched:
            mark = "*" + str(hits.get(path, 0))
        else:
            mark = "  "

        print("#" + str(rank) + mark + "  " + name)
        print("     score=" + format(score, ".5f")
              + "  image_rank=" + str(outcome["image_rank"].get(path, "-"))
              + "  caption_rank="
              + str(outcome["caption_rank"].get(path, "-")))

        if args.captions:
            print("     " + row.get("caption", "")[:150])

        print()

    if outcome["trimmed"]:
        print("Trimmed " + str(len(outcome["trimmed"])) + " result(s):")

        for path in outcome["trimmed"]:
            print("  - " + by_path.get(path, {}).get(
                "filename", Path(path).name))

        print()

    if matched:
        print("*N = N of " + str(outcome["total_terms"])
              + " query terms appear literally in the caption")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
