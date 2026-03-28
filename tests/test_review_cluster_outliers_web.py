import csv
from pathlib import Path

import torch

from scripts.reid.review_cluster_outliers_web import ClusterReviewEngine, ReviewConfig


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _make_cfg(tmp_path: Path) -> ReviewConfig:
    clusters_csv = tmp_path / "clusters.csv"
    deer_map_csv = tmp_path / "deer_map.csv"
    metadata_csv = tmp_path / "metadata.csv"
    embeddings_pt = tmp_path / "emb.pt"

    _write_csv(
        clusters_csv,
        [
            {"image_path": "data/a.jpg", "cluster_id": "0"},
            {"image_path": "data/b.jpg", "cluster_id": "0"},
            {"image_path": "data/c.jpg", "cluster_id": "0"},
            {"image_path": "data/d.jpg", "cluster_id": "1"},
        ],
    )
    _write_csv(
        deer_map_csv,
        [
            {"cluster_id": "0", "cluster_size": "3", "deer_id": "DEER_001"},
            {"cluster_id": "1", "cluster_size": "1", "deer_id": "DEER_002"},
        ],
    )
    _write_csv(
        metadata_csv,
        [
            {"image_path": "data/a.jpg"},
            {"image_path": "data/b.jpg"},
            {"image_path": "data/c.jpg"},
            {"image_path": "data/d.jpg"},
        ],
    )
    emb = torch.tensor(
        [
            [1.0, 0.0],
            [0.98, 0.05],
            [0.95, -0.01],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    torch.save(emb, embeddings_pt)

    return ReviewConfig(
        clusters_csv=clusters_csv,
        deer_map_csv=deer_map_csv,
        out_clusters_csv=tmp_path / "out_clusters.csv",
        outlier_queue_csv=tmp_path / "outlier_queue.csv",
        decisions_csv=tmp_path / "decisions.csv",
        out_deer_map_csv=tmp_path / "out_deer_map.csv",
        side_labels_csv=tmp_path / "side_labels.csv",
        removed_images_csv=tmp_path / "removed_images.csv",
        embeddings=embeddings_pt,
        metadata=metadata_csv,
        sample_size=3,
        seed=42,
        assign_threshold=0.7,
        join_candidates=3,
        batches_per_cluster=1,
        min_review_cluster_size=2,
        auto_assign_singletons=False,
    )


def test_engine_keep_all_advances_and_writes_outputs(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    engine = ClusterReviewEngine(cfg)

    assert engine.current_cluster_id == 0
    assert len(engine.current_sample) == 3

    engine.action_keep_all()

    assert engine.finished is True
    assert cfg.out_clusters_csv.exists()
    assert cfg.out_deer_map_csv.exists()
    with cfg.decisions_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["action"] == "keep_all"


def test_engine_remove_selected_updates_assignment_and_queue(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    engine = ClusterReviewEngine(cfg)

    removed_path = engine.current_sample[0]
    engine.action_remove_selected([0], [removed_path])

    assert engine.current_assign[removed_path] != 0
    assert cfg.outlier_queue_csv.exists()
    with cfg.outlier_queue_csv.open(newline="") as f:
        queue_rows = list(csv.DictReader(f))
    assert len(queue_rows) == 1
    assert queue_rows[0]["image_path"] == removed_path
    with cfg.decisions_csv.open(newline="") as f:
        decision_rows = list(csv.DictReader(f))
    assert decision_rows[-1]["action"] == "remove"
