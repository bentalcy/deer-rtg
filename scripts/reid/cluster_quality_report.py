import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import cast

import torch

repo_path = Path(__file__).resolve().parents[2]
if str(repo_path) not in sys.path:
    sys.path.insert(0, str(repo_path))

from scripts.reid.review_cluster_outliers_cv import ClusterRow, load_embeddings_by_path, load_clusters


def load_deer_map_sizes(path: Path) -> dict[int, int]:
    out: dict[int, int] = {}
    if not path.exists():
        return out
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            cid_raw = str(row.get("cluster_id") or "").strip()
            size_raw = str(row.get("cluster_size") or "").strip()
            if not cid_raw:
                continue
            try:
                cid = int(cid_raw)
            except ValueError:
                continue
            try:
                size = int(size_raw)
            except ValueError:
                continue
            out[cid] = size
    return out


def load_pair_decisions(path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not path.exists():
        return counts
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            decision = str(row.get("decision") or "").strip() or "undecided"
            counts[decision] += 1
    return counts


def load_outlier_status_counts(path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not path.exists():
        return counts
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            status = str(row.get("status") or "").strip() or "unknown"
            counts[status] += 1
    return counts


def build_members(rows: list[ClusterRow]) -> dict[int, list[str]]:
    members: dict[int, list[str]] = defaultdict(list)
    for r in rows:
        members[r.cluster_id].append(r.image_path)
    return dict(members)


def normalize(vec: torch.Tensor) -> torch.Tensor:
    return cast(torch.Tensor, vec / vec.norm())


def compute_cluster_compactness(
    members: dict[int, list[str]],
    emb_by_path: dict[str, torch.Tensor],
    min_cluster_size: int,
) -> dict[str, object]:
    cluster_means: list[float] = []
    clusters_with_embeddings = 0
    images_with_embeddings = 0
    clusters_missing_embeddings = 0

    for paths in members.values():
        vecs = [emb_by_path[p] for p in paths if p in emb_by_path]
        if len(vecs) < min_cluster_size:
            if len(paths) >= min_cluster_size:
                clusters_missing_embeddings += 1
            continue
        clusters_with_embeddings += 1
        images_with_embeddings += len(vecs)
        stacked = torch.stack([normalize(v) for v in vecs], dim=0)
        centroid = normalize(stacked.mean(dim=0))
        sims = cast(list[float], cast(list[float], (stacked @ centroid).tolist()))
        cluster_means.append(mean(sims))

    out: dict[str, object] = {
        "clusters_with_embeddings": clusters_with_embeddings,
        "clusters_missing_embeddings": clusters_missing_embeddings,
        "images_with_embeddings": images_with_embeddings,
        "cluster_mean_similarity_count": len(cluster_means),
        "cluster_mean_similarity_avg": (mean(cluster_means) if cluster_means else None),
        "cluster_mean_similarity_median": (median(cluster_means) if cluster_means else None),
    }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate cluster quality report")
    _ = parser.add_argument(
        "--clusters",
        default="data/reid/unified_round5/clusters_recognizable_plus_manual_round2_k40_corrected_web.csv",
    )
    _ = parser.add_argument(
        "--deer-map-csv",
        default="data/reid/unified_round5/deer_id_proposal_recognizable_plus_manual_round2_k40_corrected_web.csv",
    )
    _ = parser.add_argument(
        "--pair-decisions",
        default="data/reid/unified_round5/pair_review_current_decisions.csv",
    )
    _ = parser.add_argument(
        "--outlier-queue",
        default="data/reid/unified_round5/cluster_outlier_queue_k40_web.csv",
    )
    _ = parser.add_argument(
        "--embeddings",
        default="data/reid/unified_round5/embeddings_recognizable_plus_manual_round2.pt",
    )
    _ = parser.add_argument(
        "--metadata",
        default="data/reid/unified_round5/embeddings_recognizable_plus_manual_round2.csv",
    )
    _ = parser.add_argument("--min-cluster-size", type=int, default=2)
    _ = parser.add_argument(
        "--out",
        default="data/reid/unified_round5/cluster_quality_report.json",
    )
    args = parser.parse_args()

    a = cast(dict[str, object], vars(args))
    clusters_path = Path(cast(str, a["clusters"]))
    deer_map_path = Path(cast(str, a["deer_map_csv"]))
    pair_decisions_path = Path(cast(str, a["pair_decisions"]))
    outlier_queue_path = Path(cast(str, a["outlier_queue"]))
    embeddings_path = Path(cast(str, a["embeddings"]))
    metadata_path = Path(cast(str, a["metadata"]))
    min_cluster_size = int(cast(int, a["min_cluster_size"]))
    out_path = Path(cast(str, a["out"]))

    if not clusters_path.exists():
        raise FileNotFoundError(f"clusters not found: {clusters_path}")

    rows = load_clusters(clusters_path)
    members = build_members(rows)
    cluster_sizes = [len(paths) for paths in members.values()]
    singleton_count = sum(1 for size in cluster_sizes if size == 1)

    deer_sizes = load_deer_map_sizes(deer_map_path)
    zero_size_clusters = sum(1 for size in deer_sizes.values() if size == 0)

    emb_by_path = load_embeddings_by_path(embeddings_path, metadata_path)
    compactness = compute_cluster_compactness(members, emb_by_path, min_cluster_size)

    pair_counts = load_pair_decisions(pair_decisions_path)
    outlier_status = load_outlier_status_counts(outlier_queue_path)

    report = {
        "timestamp": int(time.time()),
        "clusters_with_images": len(members),
        "singleton_clusters": singleton_count,
        "singleton_rate": (singleton_count / len(members)) if members else None,
        "zero_size_clusters": zero_size_clusters,
        "pair_decisions": dict(pair_counts),
        "outlier_status_counts": dict(outlier_status),
        "compactness": compactness,
        "inputs": {
            "clusters": str(clusters_path),
            "deer_map": str(deer_map_path),
            "pair_decisions": str(pair_decisions_path),
            "outlier_queue": str(outlier_queue_path),
            "embeddings": str(embeddings_path),
            "metadata": str(metadata_path),
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(report, f, indent=2)

    print("Cluster quality report")
    print("clusters_with_images", report["clusters_with_images"])
    print("singleton_clusters", report["singleton_clusters"])
    print("singleton_rate", report["singleton_rate"])
    print("zero_size_clusters", report["zero_size_clusters"])
    print("pair_decisions", report["pair_decisions"])
    print("outlier_status_counts", report["outlier_status_counts"])
    print("compactness", report["compactness"])
    print("wrote", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
