import csv
from pathlib import Path

import torch


def kmeans(embeddings: torch.Tensor, k: int, iters: int, seed: int) -> torch.Tensor:
    if k <= 0:
        raise ValueError("k must be positive")
    if embeddings.numel() == 0:
        raise ValueError("no embeddings")
    k = min(k, embeddings.size(0))

    g = torch.Generator()
    g.manual_seed(seed)
    indices = torch.randperm(embeddings.size(0), generator=g)[:k]
    centroids = embeddings[indices].clone()

    labels = torch.zeros(embeddings.size(0), dtype=torch.long)
    for _ in range(iters):
        distances = torch.cdist(embeddings, centroids)
        labels = torch.argmin(distances, dim=1)
        new_centroids = []
        for i in range(k):
            mask = labels == i
            if mask.any():
                new_centroids.append(embeddings[mask].mean(dim=0))
            else:
                new_centroids.append(centroids[i])
        centroids = torch.stack(new_centroids, dim=0)

    return labels


def read_metadata(metadata_csv: Path) -> list[str]:
    with metadata_csv.open() as f:
        reader = csv.DictReader(f)
        return [row["image_path"] for row in reader]


def write_clusters(out_csv: Path, paths: list[str], labels: torch.Tensor) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "cluster_id"])
        writer.writeheader()
        for path, label in zip(paths, labels.tolist()):
            writer.writerow({"image_path": path, "cluster_id": label})


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", default="data/reid/embeddings.pt")
    parser.add_argument("--metadata", default="data/reid/embeddings.csv")
    parser.add_argument("--out-clusters", default="data/reid/clusters.csv")
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    embeddings = torch.load(args.embeddings)
    paths = read_metadata(Path(args.metadata))
    if embeddings.size(0) != len(paths):
        raise ValueError("embeddings and metadata length mismatch")

    labels = kmeans(embeddings, k=args.k, iters=args.iters, seed=args.seed)
    write_clusters(Path(args.out_clusters), paths, labels)
    print("Wrote clusters to", args.out_clusters)


if __name__ == "__main__":
    main()
