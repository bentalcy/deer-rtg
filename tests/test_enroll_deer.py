from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.enroll_deer import enroll_image, normalize_side


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_enroll_image_creates_new_deer_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = tmp_path / "img.jpg"
    image.write_bytes(b"x")
    gallery = tmp_path / "gallery.json"

    monkeypatch.setattr(
        "scripts.enroll_deer.load_megadescriptor", lambda: (object(), object())
    )
    monkeypatch.setattr(
        "scripts.enroll_deer.embed_single",
        lambda model, transform, path: np.array([0.8, 0.6], dtype=np.float32),
    )

    result = enroll_image(image=image, deer_id="5A", side="left", gallery_path=gallery)

    assert result["deer_id"] == "5A"
    assert result["side"] == "left"
    assert result["count_for_side"] == 1

    saved = json.loads(gallery.read_text(encoding="utf-8"))
    assert "5A" in saved
    assert saved["5A"]["left"]["image_paths"] == [str(image)]
    assert saved["5A"]["left"]["embeddings"][0] == [
        0.800000011920929,
        0.6000000238418579,
    ]


def test_enroll_image_appends_to_existing_side(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = tmp_path / "img2.jpg"
    image.write_bytes(b"x")
    gallery = tmp_path / "gallery.json"
    _write_json(
        gallery,
        {
            "5A": {
                "left": {"embeddings": [[1.0, 0.0]], "image_paths": ["old.jpg"]},
                "right": {"embeddings": [], "image_paths": []},
            }
        },
    )

    monkeypatch.setattr(
        "scripts.enroll_deer.load_megadescriptor", lambda: (object(), object())
    )
    monkeypatch.setattr(
        "scripts.enroll_deer.embed_single",
        lambda model, transform, path: np.array([0.0, 1.0], dtype=np.float32),
    )

    result = enroll_image(image=image, deer_id="5A", side="left", gallery_path=gallery)

    assert result["count_for_side"] == 2
    saved = json.loads(gallery.read_text(encoding="utf-8"))
    assert saved["5A"]["left"]["image_paths"] == ["old.jpg", str(image)]
    assert saved["5A"]["left"]["embeddings"][1] == [0.0, 1.0]


def test_enroll_image_validates_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        enroll_image(
            image=tmp_path / "missing.jpg",
            deer_id="5A",
            side="left",
            gallery_path=tmp_path / "gallery.json",
        )


def test_normalize_side_rejects_invalid_value() -> None:
    with pytest.raises(ValueError):
        normalize_side("unknown")


def test_enroll_image_duplicate_path_is_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = tmp_path / "img.jpg"
    image.write_bytes(b"x")
    gallery = tmp_path / "gallery.json"
    _write_json(
        gallery,
        {
            "5A": {
                "left": {
                    "embeddings": [[1.0, 0.0]],
                    "image_paths": [str(image)],
                },
                "right": {"embeddings": [], "image_paths": []},
            }
        },
    )

    def _should_not_run(*args, **kwargs):
        raise AssertionError("embedding should not run for duplicate path")

    monkeypatch.setattr("scripts.enroll_deer.load_megadescriptor", _should_not_run)
    monkeypatch.setattr("scripts.enroll_deer.embed_single", _should_not_run)

    result = enroll_image(image=image, deer_id="5A", side="left", gallery_path=gallery)

    assert result["count_for_side"] == 1
    saved = json.loads(gallery.read_text(encoding="utf-8"))
    assert saved["5A"]["left"]["image_paths"] == [str(image)]
    assert saved["5A"]["left"]["embeddings"] == [[1.0, 0.0]]
