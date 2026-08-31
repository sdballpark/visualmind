"""Shared fixtures.

search() reaches CUDA, two sets of model weights, and four files on disk.
All of that sits behind four seams - image_scores, caption_scores,
torch.cuda.is_available, and the two filter_paths - so the fusion logic
can be exercised with synthetic scores and no GPU.
"""
import csv
from pathlib import Path

import numpy as np
import pytest
import torch

from visualmind import events, people, retrieval

FIXTURES = Path(__file__).parent / "fixtures"

FLAT_CAPTION = "a plain wall"


@pytest.fixture
def fake_corpus(monkeypatch, tmp_path):
    """Install a synthetic corpus, returning the score maps search() sees.

    `paths` fixes the corpus row order; `image` and `caption` give each
    path its score in that modality. A test states only the score shape
    it cares about and lets the rest default to a flat caption that
    matches no query term.
    """
    def install(paths, image, caption, captions=None, allowed=None):
        captions = captions or {path: FLAT_CAPTION for path in paths}

        lookup = [
            {
                "source_path": path,
                "filename": path + ".jpg",
                "caption": captions[path],
            }
            for path in paths
        ]

        image_vector = np.array([image[path] for path in paths], dtype=float)
        caption_vector = np.array(
            [caption[path] for path in paths], dtype=float
        )

        monkeypatch.setattr(
            retrieval, "image_scores", lambda query: (image_vector, lookup)
        )
        monkeypatch.setattr(
            retrieval, "caption_scores", lambda query: (caption_vector, lookup)
        )
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        # An empty query bypasses both score functions and reads the
        # caption lookup straight off disk, so that path needs a file
        # of its own or the suite stops being hermetic.
        lookup_csv = tmp_path / "caption_lookup.csv"

        with lookup_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["source_path", "filename", "caption"]
            )
            writer.writeheader()
            writer.writerows(lookup)

        monkeypatch.setattr(retrieval, "CAPTION_LOOKUP", lookup_csv)

        if allowed is not None:
            pool = set(allowed)

            def filter_paths(names):
                if not names:
                    return None, [], {}

                return pool, ["Ada Fixture"], {"Ada Fixture": len(pool)}

            monkeypatch.setattr(retrieval.people, "filter_paths", filter_paths)

        return lookup

    return install


@pytest.fixture
def label_files(monkeypatch):
    """Point people and events at synthetic labels instead of the archive.

    Ada appears in img1-3, Bo in img2-3, Cy in img5 alone. Fixture Picnic
    covers img1-2, Sample Trip covers img4-5.
    """
    monkeypatch.setattr(people, "CLUSTERS", FIXTURES / "face_clusters.csv")
    monkeypatch.setattr(
        people, "LABELS", FIXTURES / "person_labels.example.json"
    )
    monkeypatch.setattr(events, "EVENTS", FIXTURES / "events.csv")
