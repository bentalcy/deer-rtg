from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.gallery_utils import save_gallery
from scripts.identify_deer import format_ranked_results, identify_image


def test_identify_image_passes_side_and_top_k(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    image = tmp_path / "query.jpg"
    image.write_bytes(b"x")
    seen: dict[str, object] = {}

    monkeypatch.setattr(
        "scripts.identify_deer.load_megadescriptor", lambda: (object(), object())
    )
    monkeypatch.setattr(
        "scripts.identify_deer.embed_single",
        lambda model, transform, path: np.array([1.0, 0.0], dtype=np.float32),
    )
    monkeypatch.setattr(
        "scripts.identify_deer.load_gallery",
        lambda path: {
            "5A": {
                "left": {"embeddings": [[1.0, 0.0]], "image_paths": ["a.jpg"]},
                "right": {"embeddings": [], "image_paths": []},
            }
        },
    )

    def _fake_rank(query_emb, gallery, query_side, top_k):
        seen["query_side"] = query_side
        seen["top_k"] = top_k
        return [{"deer_id": "5A", "side": "left", "confidence": 99.2}]

    monkeypatch.setattr("scripts.identify_deer.rank_matches", _fake_rank)

    matches = identify_image(
        image=image, gallery_path=tmp_path / "gallery.json", side="left", top_k=2
    )

    assert seen["query_side"] == "left"
    assert seen["top_k"] == 2
    assert matches[0]["deer_id"] == "5A"


def test_identify_image_missing_image_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        identify_image(
            image=tmp_path / "missing.jpg",
            gallery_path=tmp_path / "gallery.json",
            side="right",
            top_k=3,
        )


def test_format_ranked_results_outputs_expected_lines() -> None:
    lines = format_ranked_results(
        [
            {"deer_id": "5A", "side": "left", "confidence": 87.2},
            {"deer_id": "3B", "side": "left", "confidence": 61.0},
            {"deer_id": "UNKNOWN", "side": "", "confidence": 0.0},
        ]
    )

    assert lines[0] == "Rank 1: 5A (left) confidence: 87%"
    assert lines[1] == "Rank 2: 3B (left) confidence: 61%"
    assert lines[2] == "Rank 3: UNKNOWN"


def test_identify_image_end_to_end_returns_matching_deer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    image = tmp_path / "query.jpg"
    image.write_bytes(b"x")
    gallery_path = tmp_path / "gallery.json"
    save_gallery(
        {
            "5A": {
                "left": {
                    "embeddings": [[1.0, 0.0]],
                    "image_paths": ["images/a.jpg"],
                },
                "right": {"embeddings": [], "image_paths": []},
            }
        },
        gallery_path,
    )

    monkeypatch.setattr(
        "scripts.identify_deer.load_megadescriptor", lambda: (object(), object())
    )
    monkeypatch.setattr(
        "scripts.identify_deer.embed_single",
        lambda model, transform, path: np.array([1.0, 0.0], dtype=np.float32),
    )

    matches = identify_image(
        image=image,
        gallery_path=gallery_path,
        side="left",
        top_k=1,
    )

    assert matches[0]["deer_id"] == "5A"
    assert matches[0]["side"] == "left"
    assert matches[0]["confidence"] == 100.0


def test_identify_image_end_to_end_below_threshold_returns_unknown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    image = tmp_path / "query.jpg"
    image.write_bytes(b"x")
    gallery_path = tmp_path / "gallery.json"
    save_gallery(
        {
            "5A": {
                "left": {
                    "embeddings": [[1.0, 0.0]],
                    "image_paths": ["images/a.jpg"],
                },
                "right": {"embeddings": [], "image_paths": []},
            }
        },
        gallery_path,
    )

    monkeypatch.setattr(
        "scripts.identify_deer.load_megadescriptor", lambda: (object(), object())
    )
    monkeypatch.setattr(
        "scripts.identify_deer.embed_single",
        lambda model, transform, path: np.array([0.0, 1.0], dtype=np.float32),
    )

    matches = identify_image(
        image=image,
        gallery_path=gallery_path,
        side="left",
        top_k=1,
    )

    assert matches == [{"deer_id": "UNKNOWN", "side": "", "confidence": 0.0}]
