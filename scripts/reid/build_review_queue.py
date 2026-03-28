import csv
import json
from pathlib import Path

import torch


def load_clusters(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows: list[dict[str, str]] = []
        for row in reader:
            rows.append(row)
        return rows


def load_embeddings(embeddings_path: Path) -> torch.Tensor:
    emb = torch.load(embeddings_path, map_location="cpu").float()
    return emb / emb.norm(dim=-1, keepdim=True)


def build_index(rows: list[dict[str, str]]) -> dict[str, dict]:
    by_instance: dict[str, dict] = {}
    for row in rows:
        instance_id = (row.get("instance_id") or "").strip()
        if not instance_id:
            continue
        by_instance[instance_id] = row
    return by_instance


def group_by_cluster(rows: list[dict[str, str]], label_key: str = "cluster_id") -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        cid = str(row.get(label_key, ""))
        grouped.setdefault(cid, []).append(row)
    return grouped


def medoid_index(indices: list[int], embeddings: torch.Tensor) -> int:
    if len(indices) == 1:
        return indices[0]
    subset = embeddings[indices]
    sim = subset @ subset.T
    # maximize average similarity
    scores = sim.mean(dim=1)
    best = int(torch.argmax(scores).item())
    return indices[best]


def find_disagreements(
    rows_a: list[dict[str, str]],
    rows_b: list[dict[str, str]],
    embeddings: torch.Tensor,
    max_pairs_per_cluster: int,
) -> tuple[list[dict], dict]:
    # rows_a and rows_b should be aligned by instance_id in the same order as embeddings
    idx_by_id: dict[str, int] = {}
    for i, row in enumerate(rows_a):
        iid = row.get("instance_id", "")
        if iid:
            idx_by_id[iid] = i

    grouped_a = group_by_cluster(rows_a, "cluster_id")
    grouped_b = group_by_cluster(rows_b, "cluster_id")

    queue: list[dict] = []
    stats = {
        "a_splits": 0,
        "b_splits": 0,
    }

    # For each cluster in A, see how many B clusters it spans.
    for cid_a, members in grouped_a.items():
        if cid_a == "-1":
            continue
        b_clusters: dict[str, list[str]] = {}
        for row in members:
            iid = row.get("instance_id", "")
            b_row = next((r for r in rows_b if r.get("instance_id") == iid), None)
            if b_row is None:
                continue
            cid_b = str(b_row.get("cluster_id", ""))
            b_clusters.setdefault(cid_b, []).append(iid)

        if len(b_clusters) <= 1:
            continue

        stats["a_splits"] += 1

        # Build medoid pairs across the largest conflicting B clusters.
        items = sorted(b_clusters.items(), key=lambda kv: len(kv[1]), reverse=True)
        pairs = 0
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if pairs >= max_pairs_per_cluster:
                    break
                cid_b1, ids1 = items[i]
                cid_b2, ids2 = items[j]
                idxs1 = [idx_by_id[iid] for iid in ids1 if iid in idx_by_id]
                idxs2 = [idx_by_id[iid] for iid in ids2 if iid in idx_by_id]
                if not idxs1 or not idxs2:
                    continue
                m1 = medoid_index(idxs1, embeddings)
                m2 = medoid_index(idxs2, embeddings)
                sim = float((embeddings[m1] @ embeddings[m2]).item())
                row1 = rows_a[m1]
                row2 = rows_a[m2]
                queue.append(
                    {
                        "reason": "A_cluster_split_by_B",
                        "cluster_a": cid_a,
                        "cluster_b1": cid_b1,
                        "cluster_b2": cid_b2,
                        "instance_id_a": row1["instance_id"],
                        "instance_id_b": row2["instance_id"],
                        "image_path_a": row1["image_path"],
                        "image_path_b": row2["image_path"],
                        "similarity": f"{sim:.6f}",
                    }
                )
                pairs += 1
            if pairs >= max_pairs_per_cluster:
                break

    # For each cluster in B, see how many A clusters it spans.
    for cid_b, members in grouped_b.items():
        if cid_b == "-1":
            continue
        a_clusters: dict[str, list[str]] = {}
        for row in members:
            iid = row.get("instance_id", "")
            a_row = next((r for r in rows_a if r.get("instance_id") == iid), None)
            if a_row is None:
                continue
            cid_a = str(a_row.get("cluster_id", ""))
            a_clusters.setdefault(cid_a, []).append(iid)

        if len(a_clusters) <= 1:
            continue

        stats["b_splits"] += 1

        items = sorted(a_clusters.items(), key=lambda kv: len(kv[1]), reverse=True)
        pairs = 0
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if pairs >= max_pairs_per_cluster:
                    break
                cid_a1, ids1 = items[i]
                cid_a2, ids2 = items[j]
                idxs1 = [idx_by_id[iid] for iid in ids1 if iid in idx_by_id]
                idxs2 = [idx_by_id[iid] for iid in ids2 if iid in idx_by_id]
                if not idxs1 or not idxs2:
                    continue
                m1 = medoid_index(idxs1, embeddings)
                m2 = medoid_index(idxs2, embeddings)
                sim = float((embeddings[m1] @ embeddings[m2]).item())
                row1 = rows_a[m1]
                row2 = rows_a[m2]
                queue.append(
                    {
                        "reason": "B_cluster_split_by_A",
                        "cluster_b": cid_b,
                        "cluster_a1": cid_a1,
                        "cluster_a2": cid_a2,
                        "instance_id_a": row1["instance_id"],
                        "instance_id_b": row2["instance_id"],
                        "image_path_a": row1["image_path"],
                        "image_path_b": row2["image_path"],
                        "similarity": f"{sim:.6f}",
                    }
                )
                pairs += 1
            if pairs >= max_pairs_per_cluster:
                break

    return queue, stats


def write_queue(out_csv: Path, rows: list[dict]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["reason"])
            writer.writeheader()
        return
    fieldnames_set: set[str] = set()
    for row in rows:
        fieldnames_set.update(row.keys())
    fieldnames = sorted(fieldnames_set)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--clusters-a", default="data/reid/clusters_hdbscan.csv")
    parser.add_argument("--clusters-b", default="data/reid/clusters_nn.csv")
    parser.add_argument("--embeddings", default="data/reid/embeddings.pt")
    parser.add_argument("--out-queue", default="data/reid/review_queue.csv")
    parser.add_argument("--out-summary", default="data/reid/review_queue.json")
    parser.add_argument("--max-pairs-per-cluster", type=int, default=3)
    args = parser.parse_args()

    rows_a = load_clusters(Path(args.clusters_a))
    rows_b = load_clusters(Path(args.clusters_b))
    if not rows_a or not rows_b:
        print("Missing cluster inputs")
        return

    by_id_b = build_index(rows_b)
    rows_b_aligned: list[dict[str, str]] = []
    for row in rows_a:
        iid = row.get("instance_id", "")
        rows_b_aligned.append(by_id_b.get(iid, {"instance_id": iid, "cluster_id": ""}))

    embeddings = load_embeddings(Path(args.embeddings))
    if embeddings.size(0) != len(rows_a):
        raise ValueError("embeddings and cluster length mismatch")

    queue, stats = find_disagreements(rows_a, rows_b_aligned, embeddings, args.max_pairs_per_cluster)
    write_queue(Path(args.out_queue), queue)

    Path(args.out_summary).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_summary, "w") as f:
        json.dump({"queue": len(queue), **stats}, f, indent=2)

    print("Wrote review queue to", args.out_queue)
    print("Summary:", {"queue": len(queue), **stats})


if __name__ == "__main__":
    main()
