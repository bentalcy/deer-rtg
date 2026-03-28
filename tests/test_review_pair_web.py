import csv
from pathlib import Path
from typing import cast

from scripts.reid.review_pair_web import (
    build_state,
    merge_pairs_with_existing,
    repo_relative_path,
    save_decisions,
)


def test_merge_pairs_with_existing_keeps_prior_decision() -> None:
    pairs = [
        {
            "cluster_id": "11",
            "deer_id_likely": "DEER_011",
            "prototype_path": "a.jpg",
            "prototype_path_2": "b.jpg",
            "question_path": "c.jpg",
        },
        {
            "cluster_id": "12",
            "deer_id_likely": "DEER_012",
            "prototype_path": "d.jpg",
            "prototype_path_2": "",
            "question_path": "e.jpg",
        },
    ]
    existing = {"12": {"cluster_id": "12", "decision": "different"}}

    merged = merge_pairs_with_existing(pairs, existing)

    assert merged[0]["decision"] == ""
    assert merged[1]["decision"] == "different"


def test_build_state_picks_first_undecided_by_default() -> None:
    rows = [
        {
            "cluster_id": "1",
            "deer_id_likely": "DEER_001",
            "prototype_path": "a.jpg",
            "prototype_path_2": "",
            "question_path": "b.jpg",
            "decision": "same",
        },
        {
            "cluster_id": "2",
            "deer_id_likely": "DEER_002",
            "prototype_path": "c.jpg",
            "prototype_path_2": "",
            "question_path": "d.jpg",
            "decision": "",
        },
    ]

    state = build_state(rows)

    assert state["summary"] == {"total": 2, "decided": 1, "remaining": 1, "progress": "1/2"}
    assert state["current_index"] == 1
    current = cast(dict[str, str], state["current"])
    assert current["cluster_id"] == "2"


def test_save_decisions_writes_csv(tmp_path: Path) -> None:
    out_csv = tmp_path / "decisions.csv"
    rows = [
        {
            "cluster_id": "1",
            "deer_id_likely": "DEER_001",
            "prototype_path": "a.jpg",
            "prototype_path_2": "",
            "question_path": "b.jpg",
            "decision": "same",
        }
    ]

    save_decisions(out_csv, rows)

    with out_csv.open(newline="") as f:
        written = list(csv.DictReader(f))
    assert len(written) == 1
    assert written[0]["cluster_id"] == "1"
    assert written[0]["decision"] == "same"


def test_repo_relative_path_blocks_traversal(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    safe_path = repo_root / "data" / "img.jpg"
    safe_path.parent.mkdir(parents=True)
    _ = safe_path.write_bytes(b"x")

    assert repo_relative_path(repo_root, "data/img.jpg") == safe_path.resolve()
    assert repo_relative_path(repo_root, "../outside.jpg") is None
