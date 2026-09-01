"""Read-only HTTP surface over the retrieval layer.

Transport only. Every endpoint calls retrieval, people, events or the
existing scripts and reshapes what comes back; none of them decide
anything about ranking. Anything that needs new search behaviour belongs
in those modules, with tests, where the eval suite can see it.

Result objects carry the grid and lightbox dimensions recorded by
build_thumbnails.py. A justified grid computes row heights before any
image loads, so without true ratios in the payload the layout reflows as
thumbnails arrive. The keys are always present; the values are null only
for an image build_thumbnails.py could not read.

Source paths are never returned. The frontend addresses images by
sha256 through the static mounts, and every path-keyed structure in a
search outcome is re-keyed to sha on the way out.

See docs/api-design.md for what this deliberately does not do.
"""
import csv
import importlib.util
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from visualmind import events, people, retrieval

CATALOG = Path("data/metadata/image_catalog.csv")
DUPLICATES = Path("data/metadata/duplicate_groups.csv")
THUMBNAILS = Path("thumbnails")
THUMBNAIL_MANIFEST = THUMBNAILS / "manifest.csv"
STATUS_SCRIPT = Path("scripts/status.py")

KINDS = ("grid", "lightbox")


def read_csv(path, encoding="utf-8"):
    if not path.exists():
        return []

    with path.open("r", encoding=encoding, newline="") as handle:
        return list(csv.DictReader(handle))


def dimensions(row, kind):
    """Thumbnail dimensions, or None when the image has no thumbnail."""
    width = row.get(kind + "_width", "")
    height = row.get(kind + "_height", "")

    if not width or not height:
        return None

    return {"width": int(width), "height": int(height)}


def load_library():
    """Join catalog, captions and thumbnail dimensions by source path.

    Held in app state because it is the same few hundred rows for every
    request and re-reading three CSVs per call would dominate the cost
    of an endpoint that otherwise does nothing.
    """
    captions = {
        row["source_path"]: row["caption"]
        for row in read_csv(retrieval.CAPTION_LOOKUP)
    }
    thumbnails = {
        row["source_path"]: row
        for row in read_csv(THUMBNAIL_MANIFEST)
    }

    order = []
    by_path = {}

    for row in read_csv(CATALOG, encoding="utf-8-sig"):
        path = row["source_path"]
        thumbnail = thumbnails.get(path, {})

        record = {
            "sha256": row["sha256"],
            "filename": row["filename"],
            "caption": captions.get(path, ""),
            "grid": dimensions(thumbnail, "grid"),
            "lightbox": dimensions(thumbnail, "lightbox"),
        }

        order.append(record)
        by_path[path] = record

    return {"order": order, "by_path": by_path}


def load_status_module():
    spec = importlib.util.spec_from_file_location("status", STATUS_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def record_for(library, path):
    """The library record for a source path, or a sha-less placeholder.

    A search can return a path the catalog no longer lists, which
    status.py reports as staleness. Dropping it here would make the
    result count disagree with the count the outcome reports.
    """
    found = library["by_path"].get(path)

    if found is not None:
        return found

    return {
        "sha256": None,
        "filename": Path(path).name,
        "caption": "",
        "grid": None,
        "lightbox": None,
    }


def sha_map(library, by_path):
    """Re-key a path-keyed mapping to sha256."""
    out = {}

    for path, value in by_path.items():
        record = library["by_path"].get(path)

        if record is not None:
            out[record["sha256"]] = value

    return out


def sha_list(library, paths):
    return [
        library["by_path"][path]["sha256"]
        for path in paths
        if path in library["by_path"]
    ]


def search_payload(library, outcome):
    """The whole outcome, with every path-keyed structure re-keyed."""
    image_rank = outcome["image_rank"]
    caption_rank = outcome["caption_rank"]
    hits = outcome["hits"]
    matched = outcome["matched"]

    results = []

    for rank, (path, score) in enumerate(outcome["results"], start=1):
        record = record_for(library, path)

        results.append({
            **record,
            "rank": rank,
            "score": score,
            "image_rank": image_rank.get(path),
            "caption_rank": caption_rank.get(path),
            "term_hits": hits.get(path, 0),
            "matched": path in matched,
        })

    return {
        "results": results,
        "score_kind": outcome["score_kind"],
        "basis": outcome["basis"],
        "low_confidence": outcome["low_confidence"],
        "total_terms": outcome["total_terms"],
        "full_count": outcome["full_count"],
        "partial_count": outcome["partial_count"],
        "img_cut": outcome["img_cut"],
        "img_plateau": outcome["img_plateau"],
        "cap_cut": outcome["cap_cut"],
        "cap_plateau": outcome["cap_plateau"],
        "corpus_size": outcome["corpus_size"],
        "pool_size": outcome["pool_size"],
        "people": outcome["people"],
        "person_counts": outcome["person_counts"],
        "events": outcome["events"],
        "event_counts": outcome["event_counts"],
        "matched": sha_list(library, sorted(matched)),
        "trimmed": sha_list(library, outcome["trimmed"]),
        "hits": sha_map(library, hits),
        "image_rank": sha_map(library, image_rank),
        "caption_rank": sha_map(library, caption_rank),
    }


def duplicate_groups(library):
    groups = {}

    for row in read_csv(DUPLICATES):
        group = groups.setdefault(row["group"], {
            "group": row["group"],
            "tier": row["tier"],
            "members": [],
        })

        group["members"].append({
            **record_for(library, row["source_path"]),
            "keep": row["keep"] == "1",
        })

    return list(groups.values())


def create_app():
    app = FastAPI(
        title="VisualMind",
        description=__doc__,
        version="0.1.0",
    )

    # Loading weights at import would reserve VRAM on an idle server and
    # make startup wait on two encoders. retrieval loads them on the
    # first search instead, and holds them from then on.
    retrieval.hold_models(True)

    app.state.library = None
    app.state.status_module = None

    def library():
        if app.state.library is None:
            app.state.library = load_library()

        return app.state.library

    def status_module():
        if app.state.status_module is None:
            app.state.status_module = load_status_module()

        return app.state.status_module

    app.state.get_library = library

    @app.exception_handler(people.AmbiguousName)
    async def ambiguous_name(request, error):
        return JSONResponse(status_code=400, content={
            "error": "ambiguous_person",
            "detail": str(error),
            "query": error.query,
            "candidates": error.candidates,
        })

    @app.exception_handler(people.UnknownName)
    async def unknown_name(request, error):
        return JSONResponse(status_code=400, content={
            "error": "unknown_person",
            "detail": str(error),
            "query": error.query,
            "candidates": error.known,
        })

    @app.exception_handler(events.AmbiguousEvent)
    async def ambiguous_event(request, error):
        return JSONResponse(status_code=400, content={
            "error": "ambiguous_event",
            "detail": str(error),
            "query": error.query,
            "candidates": error.candidates,
        })

    @app.exception_handler(events.UnknownEvent)
    async def unknown_event(request, error):
        return JSONResponse(status_code=400, content={
            "error": "unknown_event",
            "detail": str(error),
            "query": error.query,
            "candidates": [],
        })

    @app.exception_handler(retrieval.IndexMismatch)
    async def index_mismatch(request, error):
        # Server state, not a client mistake: the same request will
        # succeed once the operator rebuilds.
        return JSONResponse(status_code=503, content={
            "error": "index_mismatch",
            "detail": str(error),
            "remedy": str(error).split("Rebuild with ")[-1].rstrip("."),
        })

    @app.get("/search")
    def search(
        q: str = Query("", description="Free-text query."),
        person: list[str] = Query(default=[]),
        event: list[str] = Query(default=[]),
        mode: str = Query("hybrid", pattern="^(hybrid|image|caption)$"),
        top_k: int = Query(0, ge=0),
        trim: bool = Query(False),
    ):
        if not q and not person and not event:
            raise HTTPException(
                status_code=400,
                detail="give q, person or event",
            )

        outcome = retrieval.search(
            q,
            mode=mode,
            top_k=top_k,
            trim=trim,
            persons=person,
            event_names=event,
        )

        return search_payload(library(), outcome)

    @app.get("/images")
    def images(offset: int = Query(0, ge=0), limit: int = Query(100, ge=1)):
        order = library()["order"]

        return {
            "total": len(order),
            "offset": offset,
            "limit": limit,
            "images": order[offset:offset + limit],
        }

    @app.get("/people")
    def roster_people():
        return {"people": people.roster()}

    @app.get("/events")
    def roster_events():
        return {"events": events.roster()}

    @app.get("/duplicates")
    def duplicates():
        return {"groups": duplicate_groups(library())}

    @app.get("/status")
    def status():
        module = status_module()
        catalog_paths = module.read_paths(module.CATALOG)

        if catalog_paths is None:
            return {"catalog": None, "artifacts": [], "ok": False}

        report = module.coverage_report(catalog_paths)

        return {
            "catalog": len(catalog_paths),
            "artifacts": report,
            "stale": [
                entry["artifact"] for entry in report if entry["stale"]
            ],
            "unreadable": {
                entry["artifact"]: entry["failures"]
                for entry in report if entry["failures"]
            },
            "ok": not any(entry["stale"] for entry in report),
        }

    for kind in KINDS:
        app.mount(
            "/thumbnails/" + kind,
            StaticFiles(directory=THUMBNAILS / kind, check_dir=False),
            name=kind,
        )

    return app
