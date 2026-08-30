"""Assign real names to face clusters, and keep them across re-clustering.

Labels anchor to face IDs, not cluster IDs. Cluster IDs are unstable -
change eps and Person_007 becomes someone else entirely - so storing a
name against a cluster would silently mislabel people on the next run.
Storing the face IDs that were identified means any future cluster
containing those faces inherits the name.

Assigning the same name to two clusters merges them, which is how
over-split identities (the same person at 5 and at 30) get reunited.

Writes data/metadata/person_labels.json. Real names of real people:
gitignored and blocked by the pre-commit hook.
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

CLUSTERS = Path("data/metadata/face_clusters.csv")
LABELS = Path("data/metadata/person_labels.json")


def read_clusters():
    with CLUSTERS.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    grouped = defaultdict(list)

    for row in rows:
        grouped[row["person"]].append(row)

    return grouped


def load_labels():
    if not LABELS.exists():
        return {}

    return json.loads(LABELS.read_text(encoding="utf-8"))


def save_labels(labels):
    LABELS.parent.mkdir(parents=True, exist_ok=True)
    LABELS.write_text(
        json.dumps(labels, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def resolve(grouped, labels):
    """Map each cluster to a name, by face-ID overlap with stored labels."""
    face_to_name = {}

    for name, face_ids in labels.items():
        for face_id in face_ids:
            face_to_name[face_id] = name

    resolved = {}

    for cluster, rows in grouped.items():
        votes = Counter(
            face_to_name[r["face_id"]]
            for r in rows
            if r["face_id"] in face_to_name
        )

        if votes:
            resolved[cluster] = votes.most_common(1)[0]

    return resolved


def cmd_status(grouped, labels):
    resolved = resolve(grouped, labels)

    named = 0
    total_faces = 0
    named_faces = 0

    print()
    print("=" * 76)
    print("FACE CLUSTER LABELS")
    print("=" * 76)
    print()
    print("CLUSTER".ljust(14) + "FACES".rjust(6) + "  " + "NAME")
    print("-" * 76)

    ordered = sorted(
        grouped.items(),
        key=lambda kv: (kv[0] == "unassigned", -len(kv[1])),
    )

    for cluster, rows in ordered:
        total_faces += len(rows)

        if cluster in resolved:
            name, votes = resolved[cluster]
            named += 1
            named_faces += len(rows)
            note = name

            if votes < len(rows):
                note += "  (" + str(votes) + "/" + str(len(rows)) + " known)"
        else:
            note = "-"

        print(cluster.ljust(14) + str(len(rows)).rjust(6) + "  " + note)

    print("-" * 76)
    print("Named clusters: " + str(named) + " of " + str(len(grouped)))
    print("Faces covered:  " + str(named_faces) + " of " + str(total_faces))

    people = sorted(set(n for n, _ in resolved.values()))

    if people:
        print()
        print("People: " + ", ".join(people))

    print()

    return 0


def cmd_set(grouped, labels, cluster, name):
    if cluster not in grouped:
        print("No such cluster: " + cluster)
        return 1

    face_ids = [r["face_id"] for r in grouped[cluster]]

    # Remove these faces from any other name first, so reassigning a
    # cluster does not leave the old name claiming its faces.
    for other, ids in list(labels.items()):
        remaining = [i for i in ids if i not in set(face_ids)]

        if remaining:
            labels[other] = remaining
        else:
            del labels[other]

    existing = set(labels.get(name, []))
    labels[name] = sorted(existing | set(face_ids), key=int)

    save_labels(labels)

    merged = [
        c for c, rows in grouped.items()
        if c != cluster
        and any(r["face_id"] in set(labels[name]) for r in rows)
    ]

    print()
    print(cluster + " -> " + name + "  (" + str(len(face_ids)) + " faces)")

    if merged:
        print("Merged with: " + ", ".join(sorted(merged)))

    print()

    return 0


def cmd_forget(labels, name):
    if name not in labels:
        print("No such name: " + name)
        return 1

    count = len(labels[name])
    del labels[name]
    save_labels(labels)

    print()
    print("Removed " + name + " (" + str(count) + " faces unlabelled)")
    print()

    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("status", help="Show clusters and their names.")

    p_set = sub.add_parser("set", help="Name a cluster.")
    p_set.add_argument("cluster")
    p_set.add_argument("name")

    p_forget = sub.add_parser("forget", help="Remove a name entirely.")
    p_forget.add_argument("name")

    args = parser.parse_args()

    if not CLUSTERS.exists():
        print("No clusters. Run cluster_faces.py first.")
        return 1

    grouped = read_clusters()
    labels = load_labels()

    if args.command == "set":
        return cmd_set(grouped, labels, args.cluster, args.name)

    if args.command == "forget":
        return cmd_forget(labels, args.name)

    return cmd_status(grouped, labels)


if __name__ == "__main__":
    sys.exit(main())
