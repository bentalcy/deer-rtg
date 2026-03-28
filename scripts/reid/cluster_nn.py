import csv
import json
from pathlib import Path

import torch


def load_metadata(metadata_csv: Path) -> list[dict[str, str]]:
    with metadata_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        rows: list[dict[str, str]] = []
        for row in reader:
            instance_id = (row.get("instance_id") or "").strip()
            image_path = (row.get("image_path") or "").strip()
            if not instance_id or not image_path:
                continue
            rows.append({"instance_id": instance_id, "image_path": image_path})
        return rows


def write_clusters(out_csv: Path, rows: list[dict], labels: list[int]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["instance_id", "image_path", "cluster_id", "is_noise"],
        )
        writer.writeheader()
        for row, label in zip(rows, labels):
            writer.writerow(
                {
                    "instance_id": row["instance_id"],
                    "image_path": row["image_path"],
                    "cluster_id": label,
                    "is_noise": "1" if label == -1 else "0",
                }
            )


def summarize(labels: list[int]) -> dict:
    counts: dict[str, int] = {}
    for label in labels:
        key = str(label)
        counts[key] = counts.get(key, 0) + 1
    return {
        "total": len(labels),
        "clusters": len([k for k in counts if k != "-1"]),
        "noise": counts.get("-1", 0),
        "counts": counts,
    }


def union_find(n: int):
    parent = list(range(n))
    size = [1] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]

    return find, union


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", default="data/reid/embeddings.pt")
    parser.add_argument("--metadata", default="data/reid/embeddings.csv")
    parser.add_argument("--out-clusters", default="data/reid/clusters_nn.csv")
    parser.add_argument("--out-summary", default="data/reid/clusters_nn.json")
    parser.add_argument("--cos-threshold", type=float, default=0.9)
    parser.add_argument("--min-cluster-size", type=int, default=2)
    parser.add_argument("--max-items", type=int, default=5000)
    args = parser.parse_args()

    rows = load_metadata(Path(args.metadata))
    if not rows:
        print("No metadata rows found")
        return

    embeddings = torch.load(args.embeddings, map_location="cpu").float()
    if embeddings.size(0) != len(rows):
        raise ValueError("embeddings and metadata length mismatch")

    n = embeddings.size(0)
    if n > args.max_items:
        raise ValueError(f"Too many items for dense similarity ({n} > {args.max_items})")

    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
    sim = embeddings @ embeddings.T

    find, union = union_find(n)
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j].item() >= args.cos_threshold:
                union(i, j)

    comp_map: dict[int, list[int]] = {}
    for i in range(n):
        comp_map.setdefault(find(i), []).append(i)

    labels = [-1] * n
    cluster_id = 0
    for comp in comp_map.values():
        if len(comp) < args.min_cluster_size:
            continue
        for idx in comp:
            labels[idx] = cluster_id
        cluster_id += 1

    write_clusters(Path(args.out_clusters), rows, labels)
    summary = summarize(labels)
    Path(args.out_summary).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_summary, "w") as f:
        json.dump(summary, f, indent=2)

    print("Wrote clusters to", args.out_clusters)
    print("Summary:", summary)


if __name__ == "__main__":
    main()
