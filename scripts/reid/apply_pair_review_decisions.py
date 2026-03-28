import argparse
import csv
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch


repo_path = Path(__file__).resolve().parents[2]
if str(repo_path) not in sys.path:
    sys.path.insert(0, str(repo_path))

from scripts.reid.review_cluster_outliers_cv import ClusterRow, load_embeddings_by_path, load_clusters, write_clusters


_ALLOWED_SIDES = {"left", "right", "unknown"}


@dataclass(frozen=True)
class PairDecision:
    cluster_id: int
    question_path: str
    decision: str


def load_side_predictions(metadata_csv: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not metadata_csv.exists():
        return out
    with metadata_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            image_path = str(row.get("image_path") or "").strip()
            side = str(row.get("side_pred") or "").strip().lower()
            if image_path and side in _ALLOWED_SIDES:
                out[image_path] = side
    return out


def load_user_side_labels(side_labels_csv: Path | None) -> dict[str, str]:
    if side_labels_csv is None or not side_labels_csv.exists():
        return {}
    out: dict[str, str] = {}
    with side_labels_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            image_path = str(row.get("image_path") or "").strip()
            side = str(row.get("side_label") or "").strip().lower()
            if image_path and side in _ALLOWED_SIDES:
                out[image_path] = side
    return out


def image_side(image_path: str, predicted: dict[str, str], user: dict[str, str]) -> str:
    return user.get(image_path, predicted.get(image_path, "unknown"))


def load_pair_decisions(path: Path) -> list[PairDecision]:
    decisions: list[PairDecision] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            decision = str(row.get("decision") or "").strip()
            if not decision:
                continue
            cluster_raw = str(row.get("cluster_id") or "").strip()
            question_path = str(row.get("question_path") or "").strip()
            if not cluster_raw or not question_path:
                continue
            try:
                cluster_id = int(cluster_raw)
            except ValueError:
                continue
            decisions.append(PairDecision(cluster_id=cluster_id, question_path=question_path, decision=decision))
    return decisions


def initialize_members(rows: list[ClusterRow]) -> tuple[dict[int, list[str]], dict[str, int]]:
    members: dict[int, list[str]] = defaultdict(list)
    current_assign: dict[str, int] = {}
    for r in rows:
        members[r.cluster_id].append(r.image_path)
        current_assign[r.image_path] = r.cluster_id
    return dict(members), current_assign


def initialize_sums(
    members: dict[int, list[str]],
    current_assign: dict[str, int],
    emb_by_path: dict[str, torch.Tensor],
    side_predicted: dict[str, str],
    side_user: dict[str, str],
) -> tuple[dict[tuple[int, str], torch.Tensor], dict[tuple[int, str], int]]:
    sums: dict[tuple[int, str], torch.Tensor] = {}
    counts: dict[tuple[int, str], int] = {}
    for cid, paths in members.items():
        for path in paths:
            if current_assign.get(path) != cid:
                continue
            vec = emb_by_path.get(path)
            if vec is None:
                continue
            side = image_side(path, side_predicted, side_user)
            if side not in {"left", "right"}:
                continue
            key = (cid, side)
            if key in sums:
                sums[key] = sums[key] + vec
                counts[key] = counts.get(key, 0) + 1
            else:
                sums[key] = vec.clone()
                counts[key] = 1
    return sums, counts


def normalize(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm()


def pick_target_cluster(
    image_path: str,
    from_cluster_id: int,
    members: dict[int, list[str]],
    emb_by_path: dict[str, torch.Tensor],
    sums: dict[tuple[int, str], torch.Tensor],
    counts: dict[tuple[int, str], int],
    side_predicted: dict[str, str],
    side_user: dict[str, str],
    threshold: float,
    next_new_cluster_id: int,
) -> tuple[int, float | None, str]:
    vec = emb_by_path.get(image_path)
    if vec is None:
        return next_new_cluster_id, None, "new_cluster_missing_embedding"

    side = image_side(image_path, side_predicted, side_user)
    if side not in {"left", "right"}:
        return next_new_cluster_id, None, "new_cluster_side_unknown"

    best_cluster: int | None = None
    best_score = -1.0
    vecn = normalize(vec)
    for cid in members.keys():
        if cid < 0 or cid == from_cluster_id:
            continue
        key = (cid, side)
        if counts.get(key, 0) < 1:
            continue
        centroid = normalize(sums[key])
        score = float((vecn @ centroid).item())
        if score > best_score:
            best_score = score
            best_cluster = cid

    if best_cluster is not None and best_score >= threshold:
        return best_cluster, best_score, "assigned_existing_side_match"

    return next_new_cluster_id, (best_score if best_cluster is not None else None), "new_cluster"


def adjust_sums(
    image_path: str,
    cid: int,
    delta: int,
    emb_by_path: dict[str, torch.Tensor],
    side_predicted: dict[str, str],
    side_user: dict[str, str],
    sums: dict[tuple[int, str], torch.Tensor],
    counts: dict[tuple[int, str], int],
) -> None:
    vec = emb_by_path.get(image_path)
    if vec is None:
        return
    side = image_side(image_path, side_predicted, side_user)
    if side not in {"left", "right"}:
        return
    key = (cid, side)
    if key not in sums:
        if delta > 0:
            sums[key] = vec.clone()
            counts[key] = 1
        return
    sums[key] = sums[key] + (vec * float(delta))
    counts[key] = counts.get(key, 0) + delta
    if counts[key] <= 0:
        del counts[key]
        del sums[key]


def apply_decisions(
    decisions: list[PairDecision],
    members: dict[int, list[str]],
    current_assign: dict[str, int],
    emb_by_path: dict[str, torch.Tensor],
    side_predicted: dict[str, str],
    side_user: dict[str, str],
    threshold: float,
    dry_run: bool,
) -> dict[str, object]:
    sums, counts = initialize_sums(members, current_assign, emb_by_path, side_predicted, side_user)
    moved = 0
    status_counts: Counter[str] = Counter()
    examples: list[str] = []

    next_new_cluster_id = (max(members.keys()) + 1) if members else 0

    for d in decisions:
        if d.decision == "same":
            continue
        if d.decision not in {"different", "side_mismatch", "error_image"}:
            continue

        current = current_assign.get(d.question_path)
        if current is None:
            status_counts["skip_missing_path"] += 1
            continue
        if current != d.cluster_id:
            status_counts["skip_cluster_mismatch"] += 1
            continue

        from_cid = d.cluster_id
        target, score, status = pick_target_cluster(
            d.question_path,
            from_cid,
            members,
            emb_by_path,
            sums,
            counts,
            side_predicted,
            side_user,
            threshold,
            next_new_cluster_id,
        )
        if status.startswith("new_cluster"):
            next_new_cluster_id += 1
        status_counts[status] += 1

        if dry_run:
            moved += 1
            if len(examples) < 8:
                examples.append(
                    f"{d.question_path} : {from_cid} -> {target} ({status}{'' if score is None else f' score={score:.4f}'})"
                )
            continue

        adjust_sums(d.question_path, from_cid, -1, emb_by_path, side_predicted, side_user, sums, counts)
        members[from_cid] = [p for p in members.get(from_cid, []) if p != d.question_path]

        if target not in members:
            members[target] = []

        members[target].append(d.question_path)
        current_assign[d.question_path] = target
        adjust_sums(d.question_path, target, 1, emb_by_path, side_predicted, side_user, sums, counts)

        moved += 1
        if len(examples) < 8:
            examples.append(
                f"{d.question_path} : {from_cid} -> {target} ({status}{'' if score is None else f' score={score:.4f}'})"
            )

    return {
        "moved": moved,
        "status_counts": dict(status_counts),
        "examples": examples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply pair-review decisions to a clusters CSV")
    _ = parser.add_argument(
        "--clusters",
        default="data/reid/unified_round5/clusters_recognizable_plus_manual_round2_k40_corrected_web.csv",
    )
    _ = parser.add_argument(
        "--embeddings",
        default="data/reid/unified_round5/embeddings_recognizable_plus_manual_round2.pt",
    )
    _ = parser.add_argument(
        "--metadata",
        default="data/reid/unified_round5/embeddings_recognizable_plus_manual_round2.csv",
    )
    _ = parser.add_argument(
        "--pair-decisions",
        default="data/reid/unified_round5/pair_review_current_decisions.csv",
    )
    _ = parser.add_argument(
        "--side-labels-csv",
        default="data/reid/unified_round5/side_labels_k40_web.csv",
    )
    _ = parser.add_argument("--assign-threshold", type=float, default=0.98)
    _ = parser.add_argument(
        "--out-clusters",
        default="data/reid/unified_round5/clusters_recognizable_plus_manual_round2_k40_corrected_from_pair_review.csv",
    )
    _ = parser.add_argument("--apply", action="store_true", help="Write output clusters (default is dry-run)")
    args = parser.parse_args()

    a = cast(dict[str, object], vars(args))
    clusters_path = Path(cast(str, a["clusters"]))
    embeddings_path = Path(cast(str, a["embeddings"]))
    metadata_path = Path(cast(str, a["metadata"]))
    pair_decisions_path = Path(cast(str, a["pair_decisions"]))
    side_labels_path = Path(cast(str, a["side_labels_csv"])) if a.get("side_labels_csv") else None
    out_clusters_path = Path(cast(str, a["out_clusters"]))
    threshold = float(cast(float, a["assign_threshold"]))
    apply = bool(cast(bool, a["apply"]))

    if not clusters_path.exists():
        raise FileNotFoundError(f"clusters not found: {clusters_path}")
    if not embeddings_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("missing embeddings/metadata")
    if not pair_decisions_path.exists():
        raise FileNotFoundError(f"pair decisions not found: {pair_decisions_path}")

    rows = load_clusters(clusters_path)
    members, current_assign = initialize_members(rows)
    emb_by_path = load_embeddings_by_path(embeddings_path, metadata_path)
    side_predicted = load_side_predictions(metadata_path)
    side_user = load_user_side_labels(side_labels_path)
    decisions = load_pair_decisions(pair_decisions_path)

    result = apply_decisions(
        decisions,
        members,
        current_assign,
        emb_by_path,
        side_predicted,
        side_user,
        threshold=threshold,
        dry_run=(not apply),
    )

    print("Pair decisions:", len(decisions))
    print("Moved:", result["moved"])
    print("Status counts:", result["status_counts"])
    for line in cast(list[str], result["examples"]):
        print("-", line)

    if not apply:
        print("Dry-run only. Re-run with --apply to write", out_clusters_path)
        return 0

    updated = [ClusterRow(image_path=p, cluster_id=c) for p, c in sorted(current_assign.items())]
    write_clusters(out_clusters_path, updated)
    print("Wrote", out_clusters_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
