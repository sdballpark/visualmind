# API design

`src/visualmind/api.py`, served by `scripts/serve.py`.

## Read-only by decision

Every endpoint is a `GET` that reads an artifact some builder already
produced. Nothing mutates the archive, the catalog, or any index. That is
a decision, not a stage the project has not reached yet.

The reason is the asymmetry between the two halves of the system. The read
half is cheap to be wrong about: a bad ranking shows the wrong photo and
the person looking at it says so immediately. The write half is not. This
is a family archive of scans and originals going back to 2002, much of it
with no other copy. A move applied to the wrong file is not self-announcing
the way a bad search result is, and there is no second archive to restore
from.

So the API does not move, rename, delete, or tag anything, and adding an
endpoint that does is not a small change to this file. It is the thing
described under Plan B.

## Transport only

The API calls `retrieval.search`, `people.roster`, `events.roster`, and the
existing scripts. It reshapes what they return and nothing else.

No endpoint may contain retrieval logic. If a view needs search behaviour
that does not exist, that behaviour goes into `src/visualmind/` with tests,
where both `tests/` and `scripts/check_retrieval.py` can see it. An
endpoint that filters or re-ranks its own results is invisible to the
evaluation suite, and the published numbers in
`evals/retrieval-evaluation.md` would quietly stop describing what a user
actually gets.

The one thing the API adds is a join: catalog, captions and thumbnail
dimensions, keyed by source path, built once and held in app state.

`/image/{sha256}` is the same principle applied to one photograph. It
reshapes what `people.index()`, `events.index()` and the duplicate
report already hold - who is in it, which event it belongs to, which
near-duplicates it sits with - and decides none of it. It also returns
the image's position in catalog order, so a deep link can pull the
photographs around it through the existing paged `/images` rather than
this endpoint growing a second way to list them. Self is kept in the
duplicate group so a group reads the same from any member.

## What every result carries

`sha256`, `filename`, `caption`, and the `grid` and `lightbox` dimensions
from `thumbnails/manifest.csv`.

The dimensions are not a convenience. A justified grid computes row heights
from aspect ratios before any image loads. Without true ratios in the
payload the layout reflows as thumbnails arrive, which is the visible jank
that makes a photo grid feel broken. The keys are always present. The
values are `null` only for an image `build_thumbnails.py` recorded as
unreadable, and `/status` reports those separately so a frontend can tell
"no thumbnail for this image" from "thumbnails were never built".

Source paths are never returned. Images are addressed by sha through
`/thumbnails/grid/<sha>.jpg` and `/thumbnails/lightbox/<sha>.jpg`, and every
path-keyed structure in a search outcome - `image_rank`, `caption_rank`,
`hits`, `matched`, `trimmed` - is re-keyed to sha on the way out. Paths in
this corpus are absolute and describe private photographs; the frontend has
no use for them.

### One field is not passed through

`/search` returns the whole search outcome except `caption_lookup`, which
is the entire caption corpus rather than a property of the search.
Returning it would put a few hundred kilobytes of unrelated captions in
every response. Captions for the images actually returned are on the result
objects, and the full set is available from `/images`.

## Binding

`127.0.0.1` only, and `scripts/serve.py` has no `--host` flag. There is no
authentication and the payloads describe private photographs. Making this
reachable from the LAN should require standing up a reverse proxy that
terminates auth - a decision someone makes on purpose, rather than
something that happens because a flag was easy to pass.

## Model lifetime

Encoders are not loaded at import or at startup, so an idle server holds no
VRAM and starts immediately. The first search loads them, and
`retrieval.hold_models(True)` keeps them resident from then on. A cold
search is 5.5s, almost all of it weight loading; a warm one is under 0.05s.
Loading per request would make search-as-you-type impossible.

Holding is opt-in and off by default, so the one-shot scripts still hand
their VRAM back the moment they exit.

## Errors

`AmbiguousName`, `UnknownName`, `AmbiguousEvent` and `UnknownEvent` are
client mistakes and map to `400` with the candidate list in the body, so a
frontend can offer them as suggestions instead of restating a rejected
string.

`IndexMismatch` maps to `503`, not `400`. It means an embedding matrix and
its lookup no longer correspond, which is server state: the request was
well formed and will succeed unchanged once the operator rebuilds. The body
carries the rebuild command.

## Plan B: the action layer, deliberately out of scope

The obvious next step is letting VisualMind propose changes to the archive
- "these 14 are the same photo, keep the largest", "these 300 are unsorted,
file them under Christmas 2009" - and letting a human approve them. That is
Plan B. It is not built, and none of it is half-built behind a flag.

Its shape, recorded here so this API does not accidentally foreclose it:

- A proposal is a durable object: what it would do, to which shas, why, and
  which evidence produced it.
- Nothing touches the filesystem until a person approves that specific
  proposal.
- Approval is recorded, and every applied action is reversible using
  information kept at approval time.

### Persistent approval state is what will force a redesign

Everything this API serves today is derived and disposable. Delete
`indexes/` and `thumbnails/` and a few scripts rebuild them exactly. That
property is why the API can be a thin read layer over CSV files with no
database, no migrations and no write path: there is no state whose loss
would matter.

Approval state breaks that property, and it is the only part of Plan B that
does. An approval is a human judgement. It cannot be recomputed from the
corpus, it is not derivable from any index, and losing it means asking
someone to make the same decisions over again. The moment the first
approval is recorded, this project has data that must be backed up,
migrated, and kept consistent with a corpus that keeps moving underneath it
- shas outlive paths, but a proposal referencing an image deleted since
still has to mean something.

Concretely, the parts to expect to redesign:

- **Storage.** CSV files rebuilt wholesale are right for derived data and
  wrong for an append-only record of human decisions. This is where SQLite
  arrives, and with it the first schema in this project that cannot simply
  be regenerated when it changes.
- **The read/write split.** The loopback binding and absent auth are
  defensible precisely because nothing a request can reach is damageable. A
  write path removes that argument, and "who may approve" stops being
  theoretical.
- **Staleness.** `status.py` compares derived artifacts against the catalog
  and tells you to rebuild. A pending proposal cannot be rebuilt, so it
  needs different handling: reconciled, expired, or surfaced as needing
  review.

None of that is an argument against Plan B. It is an argument for building
it as its own thing, with its storage decision made deliberately, rather
than growing it out of an endpoint here.
