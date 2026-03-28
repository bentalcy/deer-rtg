import csv
import json
from pathlib import Path

import torch
import hdbscan


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


def write_clusters(out_csv: Path, rows: list[dict], labels: list[int], probs: list[float]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "instance_id",
                "image_path",
                "cluster_id",
                "membership_prob",
                "is_noise",
            ],
        )
        writer.writeheader()
        for row, label, prob in zip(rows, labels, probs):
            writer.writerow(
                {
                    "instance_id": row["instance_id"],
                    "image_path": row["image_path"],
                    "cluster_id": label,
                    "membership_prob": f"{prob:.6f}",
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


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", default="data/reid/embeddings.pt")
    parser.add_argument("--metadata", default="data/reid/embeddings.csv")
    parser.add_argument("--out-clusters", default="data/reid/clusters_hdbscan.csv")
    parser.add_argument("--out-summary", default="data/reid/clusters_hdbscan.json")
    parser.add_argument("--min-cluster-size", type=int, default=3)
    parser.add_argument("--min-samples", type=int, default=None)
    parser.add_argument("--metric", default="euclidean", choices=["euclidean", "cosine"])
    parser.add_argument("--normalize", action="store_true")
    args = parser.parse_args()

    rows = load_metadata(Path(args.metadata))
    if not rows:
        print("No metadata rows found")
        return

    embeddings = torch.load(args.embeddings, map_location="cpu")
    if embeddings.size(0) != len(rows):
        raise ValueError("embeddings and metadata length mismatch")

    feats = embeddings.float()
    if args.normalize:
        feats = feats / feats.norm(dim=-1, keepdim=True)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric=args.metric,
    )
    labels = clusterer.fit_predict(feats.numpy())
    probs = clusterer.probabilities_.tolist()

    write_clusters(Path(args.out_clusters), rows, labels.tolist(), probs)
    summary = summarize(labels.tolist())
    Path(args.out_summary).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_summary, "w") as f:
        json.dump(summary, f, indent=2)

    print("Wrote clusters to", args.out_clusters)
    print("Summary:", summary)


if __name__ == "__main__":
    main()
