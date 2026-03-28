import csv
import json
import random
from pathlib import Path

import torch


def load_index(index_csv: Path) -> list[dict[str, str]]:
    with index_csv.open(newline="") as f:
        return list(csv.DictReader(f))


def load_embeddings(embeddings_path: Path) -> torch.Tensor:
    emb = torch.load(embeddings_path, map_location="cpu").float()
    return emb / emb.norm(dim=-1, keepdim=True)


def build_encounter_map(rows: list[dict[str, str]]) -> dict[str, list[int]]:
    encounter_map: dict[str, list[int]] = {}
    for idx, row in enumerate(rows):
        encounter_id = (row.get("encounter_id") or "").strip()
        if not encounter_id:
            encounter_id = (row.get("video_id") or "").strip() or "unknown"
        encounter_map.setdefault(encounter_id, []).append(idx)
    return encounter_map


def assign_splits(encounter_ids: list[str], seed: int, train: float, val: float) -> dict[str, str]:
    rng = random.Random(seed)
    ids = encounter_ids[:]
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * train)
    n_val = int(n * val)
    split_map: dict[str, str] = {}
    for i, eid in enumerate(ids):
        if i < n_train:
            split_map[eid] = "train"
        elif i < n_train + n_val:
            split_map[eid] = "val"
        else:
            split_map[eid] = "test"
    return split_map


def write_splits(out_csv: Path, rows: list[dict[str, str]], split_map: dict[str, str]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["instance_id", "split", "encounter_id", "video_id"]
        )
        writer.writeheader()
        for row in rows:
            encounter_id = (row.get("encounter_id") or "").strip()
            if not encounter_id:
                encounter_id = (row.get("video_id") or "").strip() or "unknown"
            writer.writerow(
                {
                    "instance_id": row.get("instance_id", ""),
                    "split": split_map.get(encounter_id, "train"),
                    "encounter_id": encounter_id,
                    "video_id": row.get("video_id", ""),
                }
            )


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


def leakage_report(
    embeddings: torch.Tensor,
    rows: list[dict[str, str]],
    split_map: dict[str, str],
    cos_threshold: float,
    max_items: int,
) -> dict:
    n = embeddings.size(0)
    if n > max_items:
        return {"skipped": True, "reason": f"n>{max_items}"}

    sim = embeddings @ embeddings.T
    find, union = union_find(n)
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j].item() >= cos_threshold:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    leakage = []
    for indices in groups.values():
        if len(indices) < 2:
            continue
        splits = set()
        for idx in indices:
            encounter_id = (rows[idx].get("encounter_id") or "").strip()
            if not encounter_id:
                encounter_id = (rows[idx].get("video_id") or "").strip() or "unknown"
            splits.add(split_map.get(encounter_id, "train"))
        if len(splits) > 1:
            leakage.append({"size": len(indices), "splits": sorted(splits)})

    return {
        "checked": True,
        "groups": len(groups),
        "leakage_groups": len(leakage),
        "leakage_samples": leakage[:20],
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--index-csv", default="data/reid/index.csv")
    parser.add_argument("--embeddings", default="data/reid/embeddings.pt")
    parser.add_argument("--out-splits", default="data/reid/splits.csv")
    parser.add_argument("--out-report", default="data/reid/splits_report.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", type=float, default=0.7)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--cos-threshold", type=float, default=0.95)
    parser.add_argument("--max-items", type=int, default=5000)
    args = parser.parse_args()

    rows = load_index(Path(args.index_csv))
    if not rows:
        print("No index rows found")
        return

    encounter_map = build_encounter_map(rows)
    split_map = assign_splits(list(encounter_map.keys()), args.seed, args.train, args.val)
    write_splits(Path(args.out_splits), rows, split_map)

    embeddings = load_embeddings(Path(args.embeddings))
    if embeddings.size(0) != len(rows):
        raise ValueError("embeddings and index length mismatch")

    report = {
        "total_instances": len(rows),
        "encounters": len(encounter_map),
        "split_counts": {
            "train": list(split_map.values()).count("train"),
            "val": list(split_map.values()).count("val"),
            "test": list(split_map.values()).count("test"),
        },
        "leakage": leakage_report(
            embeddings,
            rows,
            split_map,
            cos_threshold=args.cos_threshold,
            max_items=args.max_items,
        ),
    }

    Path(args.out_report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_report, "w") as f:
        json.dump(report, f, indent=2)

    print("Wrote splits to", args.out_splits)
    print("Report:", report)


if __name__ == "__main__":
    main()
