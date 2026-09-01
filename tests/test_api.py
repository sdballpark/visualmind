"""The HTTP surface, with retrieval patched the way the rest of the suite does.

The API is transport, so these assert shape and translation rather than
ranking: that the outcome survives the trip whole, that source paths do
not, and that an error reaches the client as the right status code.
"""
import csv
import json

import pytest
import torch
from fastapi.testclient import TestClient

from visualmind import api, events, people, retrieval

PATHS = ["/photos/a.jpg", "/photos/b.jpg", "/photos/c.jpg"]
SHAS = {path: format(i, "064x") for i, path in enumerate(PATHS, start=1)}


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def library_files(monkeypatch, tmp_path):
    """A catalog, caption lookup and thumbnail manifest on disk."""
    catalog = tmp_path / "image_catalog.csv"
    captions = tmp_path / "caption_lookup.csv"
    manifest = tmp_path / "manifest.csv"

    write_csv(catalog, ["source_path", "filename", "sha256"], [
        {"source_path": path, "filename": path.rsplit("/", 1)[-1],
         "sha256": SHAS[path]}
        for path in PATHS
    ])
    write_csv(captions, ["source_path", "filename", "caption"], [
        {"source_path": path, "filename": path.rsplit("/", 1)[-1],
         "caption": "a zebra, number " + path[-5]}
        for path in PATHS
    ])
    write_csv(manifest, [
        "sha256", "source_path", "grid_width", "grid_height",
        "lightbox_width", "lightbox_height", "status",
    ], [
        {"sha256": SHAS[PATHS[0]], "source_path": PATHS[0],
         "grid_width": 400, "grid_height": 300,
         "lightbox_width": 1600, "lightbox_height": 1200, "status": "ok"},
        {"sha256": SHAS[PATHS[1]], "source_path": PATHS[1],
         "grid_width": 300, "grid_height": 400,
         "lightbox_width": 1200, "lightbox_height": 1600, "status": "ok"},
        # No thumbnail: build_thumbnails could not read it.
        {"sha256": SHAS[PATHS[2]], "source_path": PATHS[2],
         "grid_width": "", "grid_height": "",
         "lightbox_width": "", "lightbox_height": "",
         "status": "unreadable"},
    ])

    monkeypatch.setattr(api, "CATALOG", catalog)
    monkeypatch.setattr(api, "THUMBNAIL_MANIFEST", manifest)
    monkeypatch.setattr(retrieval, "CAPTION_LOOKUP", captions)

    # Absent unless a test writes it, so no test can reach the real
    # duplicate report by forgetting to patch this one path.
    monkeypatch.setattr(api, "DUPLICATES", tmp_path / "duplicate_groups.csv")

    return tmp_path


@pytest.fixture
def client(library_files, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(retrieval, "hold_models", lambda enabled=True: None)

    return TestClient(api.create_app())


def fake_search(**outcome):
    base = {
        "results": [], "score_kind": "caption_cosine", "basis": "full",
        "matched": set(), "hits": {}, "trimmed": [], "total_terms": 1,
        "full_count": 0, "partial_count": 0, "img_cut": 2,
        "img_plateau": True, "cap_cut": 3, "cap_plateau": False,
        "low_confidence": False, "caption_lookup": [{"huge": "corpus"}],
        "caption_score": {}, "people": [], "person_counts": {},
        "events": [], "event_counts": {}, "corpus_size": 3, "pool_size": 3,
        "image_rank": {}, "caption_rank": {},
    }
    base.update(outcome)
    return base


# --- search ----------------------------------------------------------


def test_search_returns_the_outcome_fields_the_frontend_needs(
    client, monkeypatch
):
    """Named in the brief because the UI reasons about all of them."""
    monkeypatch.setattr(retrieval, "search", lambda *a, **k: fake_search(
        results=[(PATHS[0], 0.53)],
        image_rank={PATHS[0]: 15}, caption_rank={PATHS[0]: 1},
    ))

    body = client.get("/search", params={"q": "zebra"}).json()

    for field in ("basis", "score_kind", "low_confidence", "img_cut",
                  "cap_cut", "img_plateau", "cap_plateau", "image_rank",
                  "caption_rank", "corpus_size", "pool_size"):
        assert field in body, field


def test_every_result_carries_sha_caption_and_both_dimensions(
    client, monkeypatch
):
    """A justified grid needs the ratios before the images load."""
    monkeypatch.setattr(retrieval, "search", lambda *a, **k: fake_search(
        results=[(PATHS[0], 0.53), (PATHS[1], 0.41)],
    ))

    results = client.get("/search", params={"q": "zebra"}).json()["results"]

    assert [r["sha256"] for r in results] == [SHAS[PATHS[0]], SHAS[PATHS[1]]]
    assert results[0]["grid"] == {"width": 400, "height": 300}
    assert results[0]["lightbox"] == {"width": 1600, "height": 1200}
    assert results[1]["grid"] == {"width": 300, "height": 400}
    assert results[0]["caption"].startswith("a zebra")


def test_an_image_without_a_thumbnail_keeps_the_keys_with_null_values(
    client, monkeypatch
):
    """Absent is reported, not omitted, so the frontend can branch."""
    monkeypatch.setattr(retrieval, "search", lambda *a, **k: fake_search(
        results=[(PATHS[2], 0.30)],
    ))

    result = client.get("/search", params={"q": "zebra"}).json()["results"][0]

    assert result["grid"] is None
    assert result["lightbox"] is None
    assert "grid" in result and "lightbox" in result


def test_no_source_path_reaches_the_client(client, monkeypatch):
    """Paths here are absolute and describe private photographs."""
    monkeypatch.setattr(retrieval, "search", lambda *a, **k: fake_search(
        results=[(PATHS[0], 0.53)],
        image_rank={p: i for i, p in enumerate(PATHS, 1)},
        caption_rank={p: i for i, p in enumerate(PATHS, 1)},
        hits={PATHS[0]: 1}, matched={PATHS[0]}, trimmed=[PATHS[1]],
    ))

    raw = client.get("/search", params={"q": "zebra"}).text

    assert "/photos/" not in raw
    assert "source_path" not in raw


def test_path_keyed_maps_are_rekeyed_to_sha(client, monkeypatch):
    monkeypatch.setattr(retrieval, "search", lambda *a, **k: fake_search(
        results=[(PATHS[0], 0.53)],
        image_rank={PATHS[0]: 15}, caption_rank={PATHS[0]: 1},
        hits={PATHS[0]: 2}, matched={PATHS[0]}, trimmed=[PATHS[1]],
    ))

    body = client.get("/search", params={"q": "zebra"}).json()

    assert body["image_rank"] == {SHAS[PATHS[0]]: 15}
    assert body["caption_rank"] == {SHAS[PATHS[0]]: 1}
    assert body["hits"] == {SHAS[PATHS[0]]: 2}
    assert body["matched"] == [SHAS[PATHS[0]]]
    assert body["trimmed"] == [SHAS[PATHS[1]]]


def test_the_caption_corpus_is_not_echoed_into_every_response(
    client, monkeypatch
):
    """The one outcome field deliberately not passed through."""
    monkeypatch.setattr(retrieval, "search",
                        lambda *a, **k: fake_search(results=[]))

    body = client.get("/search", params={"q": "zebra"}).json()

    assert "caption_lookup" not in body


def test_search_forwards_its_parameters_unchanged(client, monkeypatch):
    """The API decides nothing about retrieval; it passes the request on."""
    seen = {}

    def capture(query, **kwargs):
        seen["query"] = query
        seen.update(kwargs)
        return fake_search()

    monkeypatch.setattr(retrieval, "search", capture)

    client.get("/search", params={
        "q": "zebra", "person": ["Ada Fixture", "Bo Fixture"],
        "event": ["event-001"], "mode": "image", "top_k": 5, "trim": True,
    })

    assert seen["query"] == "zebra"
    assert seen["persons"] == ["Ada Fixture", "Bo Fixture"]
    assert seen["event_names"] == ["event-001"]
    assert seen["mode"] == "image"
    assert seen["top_k"] == 5
    assert seen["trim"] is True


def test_an_empty_request_is_rejected(client):
    assert client.get("/search").status_code == 400


def test_an_unknown_mode_is_rejected(client, monkeypatch):
    monkeypatch.setattr(retrieval, "search",
                        lambda *a, **k: fake_search())

    assert client.get(
        "/search", params={"q": "x", "mode": "telepathy"}
    ).status_code == 422


# --- errors ----------------------------------------------------------


@pytest.mark.parametrize("error,code,expected", [
    (people.AmbiguousName("bo", ["Bo Fixture", "Bob Fixture"]),
     400, "ambiguous_person"),
    (people.UnknownName("zz", ["Ada Fixture"]), 400, "unknown_person"),
    (events.AmbiguousEvent("jun", ["event-001", "event-002"]),
     400, "ambiguous_event"),
    (events.UnknownEvent("nope"), 400, "unknown_event"),
])
def test_resolution_failures_are_client_errors_with_candidates(
    client, monkeypatch, error, code, expected
):
    def raise_it(*args, **kwargs):
        raise error

    monkeypatch.setattr(retrieval, "search", raise_it)

    response = client.get("/search", params={"q": "zebra", "person": "x"})
    body = response.json()

    assert response.status_code == code
    assert body["error"] == expected
    assert "candidates" in body


def test_a_drifted_index_is_server_state_not_a_client_mistake(
    client, monkeypatch
):
    """503, because the same request succeeds once the operator rebuilds."""
    def raise_it(*args, **kwargs):
        raise retrieval.IndexMismatch(
            "caption index: the lookup has been rebuilt since the "
            "embeddings were. Rebuild with "
            "scripts/build_caption_embeddings.py."
        )

    monkeypatch.setattr(retrieval, "search", raise_it)

    response = client.get("/search", params={"q": "zebra"})

    assert response.status_code == 503
    assert response.json()["error"] == "index_mismatch"
    assert response.json()["remedy"] == "scripts/build_caption_embeddings.py"


# --- listings --------------------------------------------------------


def test_images_pages_and_reports_a_total(client):
    """The total lets the frontend pre-allocate scroll height."""
    body = client.get("/images", params={"offset": 1, "limit": 1}).json()

    assert body["total"] == 3
    assert body["offset"] == 1
    assert len(body["images"]) == 1
    assert body["images"][0]["sha256"] == SHAS[PATHS[1]]


def test_images_past_the_end_is_empty_not_an_error(client):
    body = client.get("/images", params={"offset": 500}).json()

    assert body["total"] == 3
    assert body["images"] == []


def test_people_and_events_pass_the_rosters_through(client, monkeypatch):
    monkeypatch.setattr(people, "roster", lambda: [
        {"name": "Ada Fixture", "images": 3, "faces": 4},
    ])
    monkeypatch.setattr(events, "roster", lambda: [
        {"id": "event-001", "name": "Picnic", "start": "2020-06-01",
         "end": "2020-06-01", "images": 2},
    ])

    assert client.get("/people").json()["people"][0]["faces"] == 4
    assert client.get("/events").json()["events"][0]["start"] == "2020-06-01"


def test_duplicates_are_grouped_by_tier_with_enriched_members(
    client, monkeypatch, tmp_path
):
    write_csv(api.DUPLICATES, ["tier", "group", "keep", "source_path"], [
        {"tier": "NEAR", "group": "near-1", "keep": "1",
         "source_path": PATHS[0]},
        {"tier": "NEAR", "group": "near-1", "keep": "0",
         "source_path": PATHS[1]},
    ])

    groups = client.get("/duplicates").json()["groups"]

    assert len(groups) == 1
    assert groups[0]["tier"] == "NEAR"
    assert [m["keep"] for m in groups[0]["members"]] == [True, False]
    assert groups[0]["members"][0]["grid"] == {"width": 400, "height": 300}


def test_status_reports_coverage_as_json(client, monkeypatch):
    """Same computation status.py prints, so the two cannot disagree."""
    class FakeStatus:
        CATALOG = "catalog"

        @staticmethod
        def read_paths(path):
            return set(PATHS)

        @staticmethod
        def coverage_report(catalog_paths):
            return [
                {"artifact": "captions", "stale": False, "failures": {}},
                {"artifact": "thumbnails", "stale": True,
                 "failures": {"unreadable": 1}},
            ]

    monkeypatch.setattr(api, "load_status_module", lambda: FakeStatus)

    body = client.get("/status").json()

    assert body["catalog"] == 3
    assert body["stale"] == ["thumbnails"]
    assert body["unreadable"] == {"thumbnails": {"unreadable": 1}}
    assert body["ok"] is False


def test_the_library_is_built_once_and_held(client, monkeypatch):
    """Three CSV reads per request would dominate an endpoint this thin."""
    calls = []
    real = api.load_library
    monkeypatch.setattr(api, "load_library",
                        lambda: (calls.append(1), real())[1])

    fresh = TestClient(api.create_app())
    fresh.get("/images")
    fresh.get("/images")
    fresh.get("/duplicates")

    assert len(calls) == 1
