import argparse
import csv
import json
import shutil
from pathlib import Path

import torch


def load_embeddings(path: Path) -> torch.Tensor:
    emb = torch.load(path, map_location="cpu").float()
    return emb / emb.norm(dim=-1, keepdim=True)


def load_metadata(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_clusters(path: Path) -> dict[str, int]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    out: dict[str, int] = {}
    for row in rows:
        image_path = (row.get("image_path") or "").strip()
        cluster_id = (row.get("cluster_id") or "").strip()
        if not image_path or not cluster_id:
            continue
        out[image_path] = int(cluster_id)
    return out


def load_deer_map(path: Path | None) -> dict[int, str]:
    if path is None or not path.exists():
        return {}
    rows = load_metadata(path)
    mapping: dict[int, str] = {}
    for row in rows:
        cid_raw = (row.get("cluster_id") or "").strip()
        if not cid_raw:
            continue
        deer_id = (row.get("deer_id") or row.get("proposed_deer_id") or "").strip()
        if not deer_id:
            continue
        mapping[int(cid_raw)] = deer_id
    return mapping


def build_pairs(
    embeddings: torch.Tensor,
    metadata: list[dict[str, str]],
    image_to_cluster: dict[str, int],
    same_side_only: bool,
    deer_map: dict[int, str],
) -> list[dict[str, str]]:
    paths = [row.get("image_path", "") for row in metadata]
    sides = [row.get("side_pred", "") for row in metadata]
    labels: list[int] = []
    for path in paths:
        if path not in image_to_cluster:
            raise ValueError(f"missing cluster for image_path: {path}")
        labels.append(image_to_cluster[path])

    clusters = sorted(set(labels))
    centroids: list[torch.Tensor] = []
    for cid in clusters:
        idx = [i for i, label in enumerate(labels) if label == cid]
        c = embeddings[idx].mean(dim=0)
        c = c / c.norm()
        centroids.append(c)
    centroid_tensor = torch.stack(centroids, dim=0)

    rows: list[dict[str, str]] = []
    for ci, cid in enumerate(clusters):
        idx = [i for i, label in enumerate(labels) if label == cid]
        if len(idx) < 2:
            continue

        scored: list[tuple[int, float, float, float]] = []
        for i in idx:
            v = embeddings[i]
            own_sim = float((v @ centroid_tensor[ci]).item())
            other_sims = [
                float((v @ centroid_tensor[oj]).item())
                for oj in range(len(clusters))
                if oj != ci
            ]
            best_other = max(other_sims) if other_sims else -1.0
            margin = own_sim - best_other
            scored.append((i, own_sim, best_other, margin))

        prototype = max(scored, key=lambda x: x[3])
        candidates = scored
        prototype_candidates = scored
        if same_side_only:
            p_side = sides[prototype[0]]
            if p_side and p_side != "unknown":
                same_side_candidates = [s for s in scored if sides[s[0]] == p_side]
                if len(same_side_candidates) >= 2:
                    candidates = same_side_candidates
                if same_side_candidates:
                    prototype_candidates = same_side_candidates

        prot_sorted = sorted(prototype_candidates, key=lambda x: x[3], reverse=True)
        prototype_1 = prot_sorted[0]
        prototype_2 = prot_sorted[1] if len(prot_sorted) > 1 else None
        question = min(candidates, key=lambda x: x[3])

        deer_id = deer_map.get(cid, f"CLUSTER_{cid:03d}")
        rows.append(
            {
                "cluster_id": str(cid),
                "deer_id_likely": deer_id,
                "cluster_size": str(len(idx)),
                "prototype_path": paths[prototype_1[0]],
                "prototype_side": sides[prototype_1[0]],
                "prototype_margin": f"{prototype_1[3]:.6f}",
                "prototype_own_sim": f"{prototype_1[1]:.6f}",
                "prototype_best_other_sim": f"{prototype_1[2]:.6f}",
                "prototype_path_2": paths[prototype_2[0]] if prototype_2 is not None else "",
                "prototype_side_2": sides[prototype_2[0]] if prototype_2 is not None else "",
                "prototype_margin_2": f"{prototype_2[3]:.6f}" if prototype_2 is not None else "",
                "question_path": paths[question[0]],
                "question_side": sides[question[0]],
                "question_margin": f"{question[3]:.6f}",
                "question_own_sim": f"{question[1]:.6f}",
                "question_best_other_sim": f"{question[2]:.6f}",
            }
        )

    rows.sort(key=lambda r: int(r["cluster_id"]))
    return rows


def write_pairs(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["cluster_id"])
            writer.writeheader()
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_preview(repo_root: Path, preview_dir: Path, rows: list[dict[str, str]]) -> None:
    if preview_dir.exists():
        shutil.rmtree(preview_dir)
    preview_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        cid = int(row["cluster_id"])
        cluster_dir = preview_dir / f"cluster_{cid:03d}"
        cluster_dir.mkdir(parents=True, exist_ok=True)
        prototype = repo_root / row["prototype_path"]
        prototype_2 = repo_root / row["prototype_path_2"] if row.get("prototype_path_2") else None
        question = repo_root / row["question_path"]
        if prototype.exists():
            (cluster_dir / "prototype_1.jpg").symlink_to(prototype)
        if prototype_2 is not None and prototype_2.exists():
            (cluster_dir / "prototype_2.jpg").symlink_to(prototype_2)
        if question.exists():
            (cluster_dir / "question.jpg").symlink_to(question)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", default="data/reid/unified_round5/embeddings.pt")
    parser.add_argument("--metadata", default="data/reid/unified_round5/embeddings.csv")
    parser.add_argument("--clusters", default="data/reid/unified_round5/clusters_refined.csv")
    parser.add_argument("--out-pairs", default="data/reid/unified_round5/pair_review_refined.csv")
    parser.add_argument("--preview-dir", default="data/reid/unified_round5/pairs_preview_refined")
    parser.add_argument("--out-summary", default="data/reid/unified_round5/pair_review_refined.json")
    parser.add_argument("--deer-map-csv", default=None)
    parser.add_argument("--same-side-only", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    embeddings = load_embeddings(Path(args.embeddings))
    metadata = load_metadata(Path(args.metadata))
    image_to_cluster = load_clusters(Path(args.clusters))
    deer_map = load_deer_map(Path(args.deer_map_csv) if args.deer_map_csv else None)

    if embeddings.size(0) != len(metadata):
        raise ValueError("embeddings and metadata length mismatch")

    rows = build_pairs(
        embeddings,
        metadata,
        image_to_cluster,
        same_side_only=args.same_side_only,
        deer_map=deer_map,
    )
    write_pairs(Path(args.out_pairs), rows)
    write_preview(repo_root, Path(args.preview_dir), rows)

    summary = {
        "pairs": len(rows),
        "out_pairs": args.out_pairs,
        "preview_dir": args.preview_dir,
        "same_side_only": args.same_side_only,
        "deer_map_csv": args.deer_map_csv,
    }
    Path(args.out_summary).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out_summary).open("w") as f:
        json.dump(summary, f, indent=2)

    print("Wrote pairs to", args.out_pairs)
    print("Summary:", summary)


if __name__ == "__main__":
    main()
