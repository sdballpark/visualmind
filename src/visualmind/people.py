"""Resolve person names to the images they appear in.

Person filtering is a lookup, not a model inference. Names live in
data/metadata/person_labels.json, which no model has seen, so a name in a
free-text query cannot be interpreted by SigLIP2 or BGE. Callers pass
names explicitly rather than embedding them in the query string.

Names are matched leniently - "lisa" finds "Lisa Bogan" - but an
ambiguous match returns every candidate rather than guessing. "robert"
matches both Robert Bogan Jr and Robert L Bogan Sr, and the caller is
told so.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

CLUSTERS = Path("data/metadata/face_clusters.csv")
LABELS = Path("data/metadata/person_labels.json")


class AmbiguousName(Exception):
    def __init__(self, query, candidates):
        self.query = query
        self.candidates = candidates
        super().__init__(
            "'" + query + "' matches " + str(len(candidates)) + " people: "
            + ", ".join(candidates)
        )


class UnknownName(Exception):
    def __init__(self, query, known):
        self.query = query
        self.known = known
        super().__init__("no person matching '" + query + "'")


def available():
    """Return True when face labelling has been done."""
    return CLUSTERS.exists() and LABELS.exists()


def _face_to_name():
    labels = json.loads(LABELS.read_text(encoding="utf-8"))

    mapping = {}

    for name, face_ids in labels.items():
        for face_id in face_ids:
            mapping[face_id] = name

    return mapping


def index():
    """Map each known person to the set of images they appear in.

    Also returns per-person face counts, which differ from image counts
    when someone appears more than once in a frame.
    """
    if not available():
        return {}, {}

    face_to_name = _face_to_name()

    images = defaultdict(set)
    faces = defaultdict(int)

    with CLUSTERS.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            name = face_to_name.get(row["face_id"])

            if not name:
                continue

            images[name].add(row["source_path"])
            faces[name] += 1

    return dict(images), dict(faces)


UNPLACED = "unassigned"


def unmatched():
    """Detected faces per image that clustering did not place with anyone.

    DBSCAN marks a face as noise when too few of its neighbours sit
    inside eps, and a noise face carries no person - so index() drops it,
    and a photograph with three detected faces and two named people
    reads as a photograph with two people. That is incomplete rather
    than wrong, and it is the one place this interface says nothing
    about its own uncertainty.

    The count is returned and nothing else, deliberately. Every unplaced
    face has a nearest labelled neighbour and a distance to it, and both
    are computable from the same files - one frame in this corpus holds a
    face 0.455 from its nearest match against an eps of 0.45, which is a
    named person missed by five thousandths. Offering that name here
    would invite exactly the trust the clusterer withheld when it
    declined to place the face, which is the error a search makes when
    it presents a gradient guess as a match.
    """
    if not available():
        return {}

    counts = defaultdict(int)

    with CLUSTERS.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["person"] == UNPLACED:
                counts[row["source_path"]] += 1

    return dict(counts)


def roster():
    """People sorted by how many images they appear in."""
    images, faces = index()

    return sorted(
        (
            {
                "name": name,
                "images": len(paths),
                "faces": faces.get(name, 0),
            }
            for name, paths in images.items()
        ),
        key=lambda entry: entry["images"],
        reverse=True,
    )


def resolve(query, known):
    """Match a typed name against known people.

    Exact match wins outright. Otherwise every name containing the query
    as a word-prefix is a candidate; one candidate resolves, several
    raise rather than guess.
    """
    lowered = query.strip().lower()

    for name in known:
        if name.lower() == lowered:
            return name

    candidates = [
        name for name in known
        if any(part.lower().startswith(lowered)
               for part in name.split())
    ]

    if not candidates:
        candidates = [
            name for name in known if lowered in name.lower()
        ]

    if not candidates:
        raise UnknownName(query, sorted(known))

    if len(candidates) > 1:
        raise AmbiguousName(query, sorted(candidates))

    return candidates[0]


def filter_paths(names):
    """Images containing every named person.

    Returns (paths, resolved_names, per_person_counts). An empty name
    list returns None for paths, meaning no filtering was requested -
    distinct from an empty set, which means nobody matched.
    """
    if not names:
        return None, [], {}

    images, _faces = index()

    if not images:
        raise UnknownName(
            names[0],
            [],
        )

    resolved = [resolve(name, images.keys()) for name in names]

    counts = {name: len(images[name]) for name in resolved}

    paths = set(images[resolved[0]])

    for name in resolved[1:]:
        paths &= images[name]

    return paths, resolved, counts
