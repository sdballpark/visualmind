# Face clustering evaluation

Face detection, clustering, and person-filtered search over the same
441-image corpus. All identity judgements are manual, made against a
contact sheet of cropped faces.

## Setup

- Detector and recogniser: InsightFace buffalo_l (SCRFD + ArcFace)
- Clustering: DBSCAN over cosine distance
- Quality filters: detection score >= 0.60, face box >= 40px on both sides

Detection ran on CPU. `onnxruntime-gpu` installs into the same directory
as the CPU-only `onnxruntime` that InsightFace pulls in transitively, and
the CPU package wins. At 1.6s per image the full pass took 11.4 minutes,
which was not worth fighting the dependency over.

## Detection results

| Measure                        | Value       |
|--------------------------------|-------------|
| Images scanned                 | 441         |
| Images with at least one face  | 365         |
| Images with none               | 76          |
| Faces detected                 | 1,325       |
| Rejected by quality filters    | 174         |

The 76 images with no faces are text cards, illustrations, and a monarch
butterfly migration map - correct rejections, not misses.

## Finding 1 - scanned and detected are different facts

An image with no faces is absent from the face index but was still
scanned. Without a separate record of what the scan looked at, those two
states are indistinguishable, and a staleness check reads 365 of 441 as a
gap rather than a complete pass.

`build_faces.py` therefore writes `face_scanned.csv` alongside the index.
This also fixes resumption: without it, every re-run would re-scan the 76
face-less images forever.

## Finding 2 - the clustering threshold decides between two failure modes

DBSCAN eps controls whether the same person splits across clusters or
different people merge into one.

| eps  | Clusters | Unassigned | Largest cluster | Age span of largest |
|------|----------|------------|-----------------|---------------------|
| 0.60 | 36       | 156 (11.8%)| 257 faces       | 3-76                |
| 0.45 | 52       | 251 (18.9%)| 73 faces        | 34-78               |

At 0.60 the largest cluster held 257 faces with an apparent age range of
73 years. DBSCAN chains: face A is close to B, B to C, and a cluster grows
transitively even where its ends are not similar at all. Visual inspection
confirmed several distinct people in one cluster.

At 0.45 the same cluster resolved into a single person photographed across
decades. The cost is 7 percentage points more unassigned faces.

That trade is worth taking. An unassigned face can be labelled later; a
wrongly merged cluster silently corrupts every search for both people in
it, and nothing in the output signals the error.

Note that the estimated ages come from InsightFace's own age model and are
noisy. They were used as a smell test for merged clusters, not as data.

## Finding 3 - over-splitting is the correct failure, because labelling can undo it

52 clusters resolved to 44 distinct people. Eight clusters were duplicates
of someone already labelled:

| Person        | Clusters merged |
|---------------|-----------------|
| Daniel Bogan  | 3               |
| Colin Welch   | 3               |
| Andrew Bogan  | 2               |
| Christine Coleman | 2           |
| Liam Bogan    | 2               |
| (cartoon faces) | 2             |

Assigning the same name to two clusters merges them. This is the intended
path for age-split identities - the same person at 8 and at 30 does not
cluster together, and no threshold fixes that without merging different
people instead.

Merging is a labelling decision, not a clustering one. That is the right
place for it: a human looking at two contact sheets can answer "same
person?" reliably, and the algorithm cannot.

## Finding 4 - labels must anchor to faces, not clusters

Cluster IDs are unstable. Re-running at a different eps renumbers
everything, so a name stored against `Person_007` would silently attach to
a different person on the next run.

Labels therefore store the face IDs that were identified.
`person_labels.json` maps a name to a list of face IDs, and any future
cluster containing those faces inherits the name.

Verified by re-clustering at eps 0.50 after labelling one cluster:

    eps 0.45:  Person_001, 73 faces  -> Lisa Bogan
    eps 0.50:  Person_001, 74 faces  -> Lisa Bogan (73/74 known)

The name followed the faces, and the output flagged that the looser
threshold had pulled in one face that was not part of the original label.

## Finding 5 - person filtering is exact, and under-reports by design

44 people cover 1,074 of 1,325 faces (81%). Person filters intersect:
naming two people returns images containing both.

| Query                          | Pool | Returned | Correct |
|--------------------------------|------|----------|---------|
| "dog" --person lisa            | 73   | 3        | 3       |
| --person casey --person manvi  | 16   | 16       | 16      |

The intersection case was checked image by image: all 16 contain both
people.

The known limitation is recall, not precision. Someone present in a photo
whose face was turned away, too small, or below the detection threshold
does not match. The filter under-reports rather than over-reports, and the
search output states this rather than leaving it implicit.

## Finding 6 - names cannot go through the models

Every other query in this system is interpreted by SigLIP2 or BGE. Person
names cannot be: they live in a local JSON file no model has seen.

Parsing names out of free text would mean deciding whether "beach" is a
place or a surname, whether "Robert" means the Sr or the Jr in this
corpus, and where "Casey and Manvi at Christmas" splits. Each failure
produces wrong results with no signal that parsing was the cause.

Names are therefore passed explicitly as `--person`, and partial names
resolve only when unambiguous:

    lisa        -> Lisa Bogan
    robert      -> AmbiguousName: Robert Bogan Jr, Robert L Bogan Sr
    zzz         -> UnknownName

The ambiguous case raises rather than guessing. In a future UI this
becomes a dropdown, which is the right shape anyway.

## Privacy handling

Face embeddings are biometric identifiers for identifiable people, most of
whom are family members who did not opt into being indexed. Person labels
are their real names.

Three layers keep them out of the repository: gitignore patterns, a
pre-commit hook that refuses to stage them by filename or extension, and
verification with `git ls-files` after each commit touching this area.

The hook is local to the clone and does not travel with the repository,
which is noted in the README.

## Not yet tested

- Whether a tighter eps with explicit merge review beats 0.45
- Whether the 251 unassigned faces are mostly poor detections or mostly
  people appearing once or twice
- Whether age-split clusters could be merged automatically using capture
  date as a bridge
- Union semantics for person filters ("either person") alongside the
  current intersection
