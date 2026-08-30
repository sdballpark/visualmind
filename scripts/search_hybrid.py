"""Hybrid search, console output.

Retrieval logic lives in visualmind.retrieval so this and
scripts/search_gallery.py cannot drift apart. This module only prints.
"""
import argparse
import sys
from pathlib import Path

from visualmind import retrieval


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
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
        help="Trim threshold as a fraction of the set's score range.",
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

    print()
    print("=" * 76)
    print("HYBRID SEARCH - " + args.query)
    print("=" * 76)

    outcome = retrieval.search(
        args.query,
        mode=args.mode,
        top_k=args.top_k,
        rrf_k=args.rrf_k,
        gradient_floor=args.gradient_floor,
        min_partial_terms=args.min_partial_terms,
        semantic_drop=args.semantic_drop,
        trim=args.trim,
    )

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
        print("Trimmed " + str(len(outcome["trimmed"]))
              + " result(s) below the caption-score threshold:")

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
