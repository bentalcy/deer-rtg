from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.gallery_utils import (
    compute_prototype,
    cosine_similarity,
    load_gallery,
    rank_matches,
    save_gallery,
)


def test_save_and_load_gallery_roundtrip(tmp_path: Path) -> None:
    gallery = {
        "5A": {
            "left": {
                "embeddings": [[1.0, 0.0], [0.8, 0.2]],
                "image_paths": ["images/a.jpg", "images/b.jpg"],
            },
            "right": {
                "embeddings": [[0.0, 1.0]],
                "image_paths": ["images/c.jpg"],
            },
        }
    }
    path = tmp_path / "gallery.json"

    save_gallery(gallery, path)
    loaded = load_gallery(path)

    assert loaded == gallery


def test_compute_prototype_returns_l2_normalized_mean() -> None:
    emb = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]

    proto = compute_prototype(emb)

    expected = np.array([2.0 / 3.0, 1.0 / 3.0], dtype=np.float32)
    expected = expected / np.linalg.norm(expected)
    assert np.allclose(proto, expected, atol=1e-6)
    assert np.isclose(np.linalg.norm(proto), 1.0, atol=1e-6)


def test_rank_matches_enforces_side_gating() -> None:
    gallery = {
        "left_deer": {
            "left": {"embeddings": [[1.0, 0.0]], "image_paths": ["a.jpg"]},
            "right": {"embeddings": [], "image_paths": []},
        },
        "right_deer": {
            "left": {"embeddings": [], "image_paths": []},
            "right": {"embeddings": [[0.0, 1.0]], "image_paths": ["b.jpg"]},
        },
    }

    query = np.array([1.0, 0.0], dtype=np.float32)
    matches = rank_matches(query, gallery, query_side="left", top_k=3)

    assert len(matches) == 1
    assert matches[0]["deer_id"] == "left_deer"
    assert matches[0]["side"] == "left"


def test_rank_matches_unknown_queries_both_sides() -> None:
    gallery = {
        "left_deer": {
            "left": {"embeddings": [[1.0, 0.0]], "image_paths": ["a.jpg"]},
            "right": {"embeddings": [], "image_paths": []},
        },
        "right_deer": {
            "left": {"embeddings": [], "image_paths": []},
            "right": {"embeddings": [[0.0, 1.0]], "image_paths": ["b.jpg"]},
        },
    }

    query = np.array([0.0, 1.0], dtype=np.float32)
    matches = rank_matches(query, gallery, query_side="unknown", top_k=2)

    assert len(matches) == 1
    assert matches[0]["deer_id"] == "right_deer"
    assert matches[0]["side"] == "right"


def test_rank_matches_empty_gallery_returns_unknown() -> None:
    query = np.array([1.0, 0.0], dtype=np.float32)
    matches = rank_matches(query, {}, query_side="left", top_k=3)

    assert matches == [{"deer_id": "UNKNOWN", "side": "", "confidence": 0.0}]


def test_rank_matches_below_threshold_returns_unknown() -> None:
    gallery = {
        "left_deer": {
            "left": {"embeddings": [[1.0, 0.0]], "image_paths": ["a.jpg"]},
            "right": {"embeddings": [], "image_paths": []},
        }
    }

    query = np.array([0.0, 1.0], dtype=np.float32)
    matches = rank_matches(query, gallery, query_side="left", top_k=1)

    assert matches == [{"deer_id": "UNKNOWN", "side": "", "confidence": 0.0}]


def test_rank_matches_confidence_formula_known_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query = np.array([1.0, 0.0], dtype=np.float32)
    cos_057_y = float(np.sqrt(1.0 - (0.57**2)))
    gallery = {
        "cos_100": {
            "left": {"embeddings": [[1.0, 0.0]], "image_paths": ["a.jpg"]},
            "right": {"embeddings": [], "image_paths": []},
        },
        "cos_057": {
            "left": {
                "embeddings": [[0.57, cos_057_y]],
                "image_paths": ["b.jpg"],
            },
            "right": {"embeddings": [], "image_paths": []},
        },
        "cos_neg100": {
            "left": {"embeddings": [[-1.0, 0.0]], "image_paths": ["c.jpg"]},
            "right": {"embeddings": [], "image_paths": []},
        },
    }

    monkeypatch.setattr("scripts.gallery_utils.DEFAULT_SIMILARITY_THRESHOLD", -1.0)
    matches = rank_matches(query, gallery, query_side="left", top_k=3)
    by_id = {row["deer_id"]: row["confidence"] for row in matches}

    assert np.isclose(by_id["cos_100"], 100.0, atol=1e-6)
    assert np.isclose(by_id["cos_057"], 78.5, atol=1e-6)
    assert np.isclose(by_id["cos_neg100"], 0.0, atol=1e-6)


def test_cosine_similarity_matches_dot_product() -> None:
    a = np.array([0.6, 0.8], dtype=np.float32)
    b = np.array([1.0, 0.0], dtype=np.float32)

    sim = cosine_similarity(a, b)

    assert np.isclose(sim, 0.6, atol=1e-6)
