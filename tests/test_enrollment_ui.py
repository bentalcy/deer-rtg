from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.enrollment_ui import (
    build_items,
    handle_bulk_enroll_payload,
    handle_enroll_payload,
)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_build_items_marks_prefill_and_enrolled(tmp_path: Path) -> None:
    embeddings_csv = tmp_path / "embeddings.csv"
    gallery_json = tmp_path / "gallery.json"

    _write_csv(
        embeddings_csv,
        [
            {"image_path": str(tmp_path / "a.jpg"), "side_pred": "left"},
            {"image_path": str(tmp_path / "b.jpg"), "side_pred": "RIGHT"},
            {"image_path": str(tmp_path / "c.jpg"), "side_pred": "weird"},
        ],
    )
    _write_json(
        gallery_json,
        {
            "5A": {
                "left": {
                    "embeddings": [[1.0, 0.0]],
                    "image_paths": [str(tmp_path / "a.jpg")],
                },
                "right": {"embeddings": [], "image_paths": []},
            }
        },
    )

    items = build_items(gallery_path=gallery_json, embeddings_csv=embeddings_csv)

    assert len(items) == 3
    assert items[0]["side_prefill"] == "left"
    assert items[0]["already_enrolled"] is True
    assert items[1]["side_prefill"] == "right"
    assert items[2]["side_prefill"] == "unknown"


def test_build_items_from_images_dir(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "deer1.jpg").write_bytes(b"x")
    (images_dir / "deer2.JPG").write_bytes(b"x")
    (images_dir / "notes.txt").write_bytes(b"x")  # should be ignored

    items = build_items(gallery_path=tmp_path / "gallery.json", images_dir=images_dir)

    assert len(items) == 2
    assert all(item["side_prefill"] == "unknown" for item in items)
    assert all(item["already_enrolled"] is False for item in items)
    assert all("name_suggestion" in item for item in items)


def test_build_items_cluster_suggestion_prefill(tmp_path: Path) -> None:
    embeddings_csv = tmp_path / "embeddings.csv"
    clusters_csv = tmp_path / "clusters.csv"

    _write_csv(
        embeddings_csv,
        [{"image_path": str(tmp_path / "a.jpg"), "side_pred": "left"}],
    )
    _write_csv(
        clusters_csv,
        [{"image_path": str(tmp_path / "a.jpg"), "cluster_id": "3"}],
    )

    items = build_items(
        gallery_path=tmp_path / "gallery.json",
        embeddings_csv=embeddings_csv,
        clusters_csv=clusters_csv,
    )

    assert items[0]["name_suggestion"] == "cluster-3"


def test_build_items_cluster_noise_excluded(tmp_path: Path) -> None:
    embeddings_csv = tmp_path / "embeddings.csv"
    clusters_csv = tmp_path / "clusters.csv"

    _write_csv(
        embeddings_csv,
        [{"image_path": str(tmp_path / "a.jpg"), "side_pred": "left"}],
    )
    _write_csv(
        clusters_csv,
        [{"image_path": str(tmp_path / "a.jpg"), "cluster_id": "-1"}],
    )

    items = build_items(
        gallery_path=tmp_path / "gallery.json",
        embeddings_csv=embeddings_csv,
        clusters_csv=clusters_csv,
    )

    assert items[0]["name_suggestion"] == ""


def test_handle_enroll_payload_calls_enroll_fn(tmp_path: Path) -> None:
    seen: dict[str, str] = {}

    def _fake_enroll(
        image: Path, deer_id: str, side: str, gallery_path: Path
    ) -> dict[str, object]:
        seen["image"] = str(image)
        seen["deer_id"] = deer_id
        seen["side"] = side
        seen["gallery_path"] = str(gallery_path)
        return {"ok": True}

    payload = {
        "image_path": str(tmp_path / "x.jpg"),
        "deer_id": "5A",
        "side": "left",
    }
    out = handle_enroll_payload(
        payload, tmp_path / "gallery.json", enroll_fn=_fake_enroll
    )

    assert out == {"ok": True}
    assert seen["deer_id"] == "5A"
    assert seen["side"] == "left"


def test_handle_bulk_enroll_payload_processes_only_complete_rows(
    tmp_path: Path,
) -> None:
    called: list[tuple[str, str, str]] = []

    def _fake_enroll(
        image: Path, deer_id: str, side: str, gallery_path: Path
    ) -> dict[str, object]:
        called.append((str(image), deer_id, side))
        return {"ok": True}

    payload = {
        "items": [
            {"image_path": str(tmp_path / "a.jpg"), "deer_id": "5A", "side": "left"},
            {"image_path": str(tmp_path / "b.jpg"), "deer_id": "", "side": "left"},
            {"image_path": str(tmp_path / "c.jpg"), "deer_id": "6B", "side": "right"},
        ]
    }

    out = handle_bulk_enroll_payload(
        payload, tmp_path / "gallery.json", enroll_fn=_fake_enroll
    )

    assert out["requested"] == 3
    assert out["enrolled"] == 2
    assert len(called) == 2


def test_build_items_handles_missing_gallery_file(tmp_path: Path) -> None:
    embeddings_csv = tmp_path / "embeddings.csv"
    missing_gallery_json = tmp_path / "missing_gallery.json"

    _write_csv(
        embeddings_csv,
        [
            {"image_path": str(tmp_path / "a.jpg"), "side_pred": "left"},
            {"image_path": str(tmp_path / "b.jpg"), "side_pred": "right"},
        ],
    )

    items = build_items(
        gallery_path=missing_gallery_json,
        embeddings_csv=embeddings_csv,
    )

    assert len(items) == 2
    assert all(item["already_enrolled"] is False for item in items)
    assert items[0]["side_prefill"] == "left"
    assert items[1]["side_prefill"] == "right"
