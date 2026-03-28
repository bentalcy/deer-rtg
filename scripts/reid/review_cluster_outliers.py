import argparse
import csv
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from PIL import ExifTags, Image
import torch


@dataclass
class ClusterRow:
    image_path: str
    cluster_id: int


def load_clusters(path: Path) -> list[ClusterRow]:
    rows: list[ClusterRow] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            image_path = (row.get("image_path") or "").strip()
            cluster_raw = (row.get("cluster_id") or "").strip()
            if not image_path or not cluster_raw:
                continue
            rows.append(ClusterRow(image_path=image_path, cluster_id=int(cluster_raw)))
    return rows


def load_deer_map(path: Path | None) -> dict[int, str]:
    if path is None or not path.exists():
        return {}
    out: dict[int, str] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            cid = (row.get("cluster_id") or "").strip()
            deer_id = (row.get("deer_id") or row.get("proposed_deer_id") or "").strip()
            if cid and deer_id:
                out[int(cid)] = deer_id
    return out


def write_deer_map(path: Path, deer_map: dict[int, str], members: dict[int, list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cluster_id", "cluster_size", "deer_id"])
        w.writeheader()
        for cid in sorted(members.keys()):
            if cid < 0:
                continue
            w.writerow(
                {
                    "cluster_id": cid,
                    "cluster_size": len(members.get(cid, [])),
                    "deer_id": deer_map.get(cid, f"CLUSTER_{cid:03d}"),
                }
            )


def write_clusters(path: Path, rows: list[ClusterRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_path", "cluster_id"])
        w.writeheader()
        for row in rows:
            w.writerow({"image_path": row.image_path, "cluster_id": row.cluster_id})


def append_outlier_queue(
    path: Path,
    image_path: str,
    from_cluster_id: int,
    deer_id_likely: str,
    assigned_cluster_id: int,
    assigned_score: float | None,
    status: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "image_path",
                "from_cluster_id",
                "deer_id_likely",
                "assigned_cluster_id",
                "assigned_score",
                "status",
            ],
        )
        if not exists:
            w.writeheader()
        w.writerow(
            {
                "timestamp": int(time.time()),
                "image_path": image_path,
                "from_cluster_id": from_cluster_id,
                "deer_id_likely": deer_id_likely,
                "assigned_cluster_id": assigned_cluster_id,
                "assigned_score": "" if assigned_score is None else f"{assigned_score:.6f}",
                "status": status,
            }
        )


def append_decision(
    path: Path,
    cluster_id: int,
    deer_id_likely: str,
    shown: list[str],
    removed: list[str],
    assignments: list[dict[str, str]],
    action: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "cluster_id",
                "deer_id_likely",
                "shown_paths",
                "removed_paths",
                "assignments",
                "action",
                "decision",
            ],
        )
        if not exists:
            w.writeheader()
        w.writerow(
            {
                "timestamp": int(time.time()),
                "cluster_id": cluster_id,
                "deer_id_likely": deer_id_likely,
                "shown_paths": "|".join(shown),
                "removed_paths": "|".join(removed),
                "assignments": json.dumps(assignments, separators=(",", ":")),
                "action": action,
                "decision": "remove" if removed else "keep_all",
            }
        )


def to_abs(repo_root: Path, image_path: str) -> Path:
    p = Path(image_path)
    return p if p.is_absolute() else repo_root / p


def normalize_crop_basename(name: str) -> str:
    return re.sub(r"__dup\d+(?=(_det\d+\.[A-Za-z0-9]+)$)", "", name)


def build_source_lookup(repo_root: Path, actions_globs: list[str]) -> dict[str, Path]:
    lookup: dict[str, Path] = {}
    for pattern in actions_globs:
        for actions_path in repo_root.glob(pattern):
            if not actions_path.is_file():
                continue
            try:
                raw = json.loads(actions_path.read_text())
            except Exception:
                continue
            if not isinstance(raw, list):
                continue
            for row in raw:
                frame_path = str(row.get("frame_path") or "").strip()
                dets = row.get("detections") or []
                if not frame_path or not isinstance(dets, list):
                    continue
                abs_frame = to_abs(repo_root, frame_path)
                for det in dets:
                    crop_new = str((det or {}).get("new_name") or "").strip()
                    if not crop_new:
                        continue
                    base = Path(crop_new).name
                    if base and base not in lookup:
                        lookup[base] = abs_frame
    return lookup


def _dms_to_deg(value):
    try:
        d = float(value[0][0]) / float(value[0][1])
        m = float(value[1][0]) / float(value[1][1])
        s = float(value[2][0]) / float(value[2][1])
        return d + (m / 60.0) + (s / 3600.0)
    except Exception:
        return None


def extract_image_meta(path: Path, source_frame_path: Path | None = None) -> dict[str, str]:
    out: dict[str, str] = {}
    meta_path = source_frame_path if source_frame_path is not None and source_frame_path.exists() else path
    try:
        img = Image.open(meta_path)
        exif = img.getexif()
        if exif:
            date_time = exif.get(306) or exif.get(36867)
            if date_time:
                out["datetime"] = str(date_time)
            gps_info = exif.get(34853)
            if gps_info:
                gps = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps_info.items()}
                lat = _dms_to_deg(gps.get("GPSLatitude")) if gps.get("GPSLatitude") else None
                lon = _dms_to_deg(gps.get("GPSLongitude")) if gps.get("GPSLongitude") else None
                if lat is not None and gps.get("GPSLatitudeRef") == "S":
                    lat = -lat
                if lon is not None and gps.get("GPSLongitudeRef") == "W":
                    lon = -lon
                if lat is not None and lon is not None:
                    out["location"] = f"{lat:.6f}, {lon:.6f}"
    except Exception:
        pass

    if "datetime" not in out:
        m = re.search(r"(\d{4}-\d{2}-\d{2})[ _](\d{2}-\d{2}-\d{2})", path.name)
        if m:
            out["datetime"] = f"{m.group(1)} {m.group(2)}"
    if "datetime" not in out:
        out["datetime"] = "n/a"
    if "location" not in out:
        out["location"] = "n/a"
    return out


def panel_title(image_path: str, source_lookup: dict[str, Path], repo_root: Path, key_idx: int) -> str:
    abs_path = to_abs(repo_root, image_path)
    source = source_lookup.get(abs_path.name)
    if source is None:
        source = source_lookup.get(normalize_crop_basename(abs_path.name))
    meta = extract_image_meta(abs_path, source)
    source_name = source.name if source is not None else "n/a"
    return (
        f"[{key_idx}] {abs_path.name}\n"
        f"time: {meta['datetime']}\n"
        f"location: {meta['location']}\n"
        f"source: {source_name}"
    )


def cluster_size_map(members: dict[int, list[str]], current_assign: dict[str, int], deer_map: dict[int, str]) -> str:
    counts: dict[int, int] = {}
    for cid, paths in members.items():
        if cid < 0:
            continue
        counts[cid] = sum(1 for p in paths if current_assign.get(p) == cid)
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    if not ordered:
        return "none"
    lines: list[str] = []
    for cid, size in ordered:
        deer_id = deer_map.get(cid, f"CLUSTER_{cid:03d}")
        lines.append(f"{deer_id} [{cid}] = {size}")
    return "\n".join(lines)


def waiting_reassign_count(outlier_queue_csv: Path) -> int:
    if not outlier_queue_csv.exists():
        return 0
    latest: dict[str, str] = {}
    with outlier_queue_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            image_path = (row.get("image_path") or "").strip()
            status = (row.get("status") or "").strip()
            if image_path:
                latest[image_path] = status
    return sum(1 for s in latest.values() if s == "needs_reassign")


def load_embeddings_by_path(embeddings_path: Path, metadata_path: Path) -> dict[str, torch.Tensor]:
    emb = torch.load(embeddings_path, map_location="cpu").float()
    emb = emb / emb.norm(dim=-1, keepdim=True)
    rows: list[dict[str, str]] = []
    with metadata_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if emb.size(0) != len(rows):
        raise ValueError("embeddings and metadata length mismatch")
    by: dict[str, torch.Tensor] = {}
    for i, row in enumerate(rows):
        p = (row.get("image_path") or "").strip()
        if p:
            by[p] = emb[i]
    return by


def assign_removed_image(
    image_path: str,
    from_cluster_id: int,
    members: dict[int, list[str]],
    emb_by_path: dict[str, torch.Tensor],
    threshold: float,
) -> tuple[int, float | None, str]:
    vec = emb_by_path.get(image_path)
    if vec is None:
        next_id = (max(members.keys()) + 1) if members else 0
        return next_id, None, "new_cluster_missing_embedding"

    best_cluster = None
    best_score = -1.0
    for cid, paths in members.items():
        if cid < 0 or cid == from_cluster_id:
            continue
        cvecs = [emb_by_path[p] for p in paths if p in emb_by_path]
        if not cvecs:
            continue
        centroid = torch.stack(cvecs, dim=0).mean(dim=0)
        centroid = centroid / centroid.norm()
        score = float((vec @ centroid).item())
        if score > best_score:
            best_score = score
            best_cluster = cid

    if best_cluster is not None and best_score >= threshold:
        return best_cluster, best_score, "assigned_existing"

    next_id = (max(members.keys()) + 1) if members else 0
    return next_id, (best_score if best_cluster is not None else None), "new_cluster"


def review_batch(
    repo_root: Path,
    source_lookup: dict[str, Path],
    cluster_id: int,
    deer_id_likely: str,
    sample_paths: list[str],
    merge_candidates: list[dict[str, Any]],
    cluster_map_text: str,
    waiting_count: int,
    batch_idx: int,
) -> dict[str, Any] | bool:
    n = len(sample_paths)
    cols = 2
    rows = 2
    fig, axes = plt.subplots(rows, cols, figsize=(14, 10))
    axes_flat = axes.flatten()

    status_box = fig.text(0.01, 0.02, "", fontsize=9, va="bottom", family="monospace")

    for i in range(4):
        ax = axes_flat[i]
        ax.axis("off")
        ax.set_facecolor("#f3f4f6")
        if i >= n:
            continue
        p = to_abs(repo_root, sample_paths[i])
        if p.exists():
            img = Image.open(p).convert("RGB")
            ax.imshow(img)
        ax.set_title(panel_title(sample_paths[i], source_lookup, repo_root, i + 1), fontsize=9)

    picked: dict[str, Any] = {"result": {"action": "keep_all", "paths": []}}
    selected: set[int] = set()
    rename_mode = {"active": False, "text": ""}
    merge_mode = {"active": False, "idx": 0}

    def render_title() -> None:
        selected_keys = ",".join(str(i + 1) for i in sorted(selected)) if selected else "none"
        rename_hint = ""
        if rename_mode["active"]:
            rename_hint = (
                f"\nRENAME MODE: type new deer-id, Enter=confirm, Backspace=delete, Esc=cancel | text: {rename_mode['text']}"
            )
        merge_hint = ""
        if merge_mode["active"]:
            if merge_candidates:
                cand = merge_candidates[min(int(merge_mode["idx"]), len(merge_candidates) - 1)]
                merge_hint = (
                    "\nJOIN MODE: [y]=merge into shown candidate, [n]=next candidate, [x]=new cluster, Esc=cancel"
                    f"\nCandidate {int(merge_mode['idx']) + 1}/{len(merge_candidates)}: "
                    f"cluster {cand['cluster_id']} ({cand['deer_id']}) score={cand['score']:.3f}"
                )
            else:
                merge_hint = "\nJOIN MODE: no candidate clusters. Press [x] for new cluster or Esc to cancel."
        header = (
            f"Cluster {cluster_id} | Deer ID: {deer_id_likely} | Batch {batch_idx}\n"
            f"Selected: {selected_keys} | Waiting reassignment: {waiting_count}\n"
            "Keys: 1..4 toggle, c remove, 0 keep, a all-different, k manual-assign, n rename, j join, q quit"
        )
        fig.suptitle(header, fontsize=11, y=0.98)
        status_box.set_text(f"Cluster Map (deer-id [cluster-id] = count)\n{cluster_map_text}{rename_hint}{merge_hint}")
        fig.canvas.draw_idle()

    render_title()

    def on_key(event):
        k = (event.key or "").lower()
        if merge_mode["active"]:
            if k in {"escape", "esc"}:
                merge_mode["active"] = False
                render_title()
                return
            if k == "y":
                if merge_candidates:
                    cand = merge_candidates[min(int(merge_mode["idx"]), len(merge_candidates) - 1)]
                    picked["result"] = {
                        "action": "merge_cluster",
                        "paths": [],
                        "target_cluster_id": int(cand["cluster_id"]),
                        "target_deer_id": str(cand["deer_id"]),
                    }
                    plt.close(fig)
                return
            if k == "n":
                if merge_candidates:
                    merge_mode["idx"] = (int(merge_mode["idx"]) + 1) % len(merge_candidates)
                render_title()
                return
            if k == "x":
                picked["result"] = {"action": "merge_cluster_new", "paths": []}
                plt.close(fig)
                return
            if k in {"1", "2", "3", "4", "5", "6", "7", "8", "9"}:
                idx = int(k) - 1
                if 0 <= idx < len(merge_candidates):
                    merge_mode["idx"] = idx
                render_title()
                return
            return

        if rename_mode["active"]:
            if k in {"escape", "esc"}:
                rename_mode["active"] = False
                rename_mode["text"] = ""
                render_title()
                return
            if k in {"enter", "return"}:
                new_name = rename_mode["text"].strip()
                if new_name:
                    picked["result"] = {"action": "rename_cluster", "paths": [], "name": new_name}
                    plt.close(fig)
                else:
                    rename_mode["active"] = False
                    render_title()
                return
            if k == "backspace":
                rename_mode["text"] = rename_mode["text"][:-1]
                render_title()
                return
            if len(k) == 1 and (k.isalnum() or k in {"_", "-"}):
                rename_mode["text"] += k
                render_title()
                return
            return

        if k == "q":
            picked["result"] = False
            plt.close(fig)
            return
        if k == "0":
            picked["result"] = {"action": "keep_all", "paths": []}
            plt.close(fig)
            return
        if k == "c":
            picked["result"] = {"action": "remove", "paths": [sample_paths[i] for i in sorted(selected)]}
            plt.close(fig)
            return
        if k == "a":
            picked["result"] = {"action": "all_different", "paths": list(sample_paths)}
            plt.close(fig)
            return
        if k in {"n", "r"}:
            rename_mode["active"] = True
            rename_mode["text"] = ""
            merge_mode["active"] = False
            merge_mode["text"] = ""
            render_title()
            return
        if k == "j":
            merge_mode["active"] = True
            merge_mode["idx"] = 0
            rename_mode["active"] = False
            rename_mode["text"] = ""
            render_title()
            return
        if k == "k":
            picked["result"] = {"action": "manual_assign", "paths": [sample_paths[i] for i in sorted(selected)]}
            plt.close(fig)
            return
        if k in {"1", "2", "3", "4"}:
            idx = int(k) - 1
            if idx < n:
                if idx in selected:
                    selected.remove(idx)
                elif len(selected) < 2:
                    selected.add(idx)
                render_title()

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.tight_layout()
    plt.show()
    return picked["result"]


def resolve_target_cluster(target: str, deer_map: dict[int, str], members: dict[int, list[str]]) -> tuple[int, str]:
    t = target.strip()
    if not t:
        next_id = (max(members.keys()) + 1) if members else 0
        deer_map[next_id] = f"DEER_{next_id+1:03d}"
        return next_id, "new_cluster_empty_target"
    if t.isdigit():
        cid = int(t)
        if cid not in members:
            members[cid] = []
        deer_map.setdefault(cid, f"DEER_{cid+1:03d}")
        return cid, "manual_cluster_id"
    for cid, deer_id in deer_map.items():
        if deer_id == t:
            members.setdefault(cid, [])
            return cid, "manual_deer_id"
    next_id = (max(members.keys()) + 1) if members else 0
    members[next_id] = []
    deer_map[next_id] = t
    return next_id, "new_cluster_named"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clusters-csv", default="data/reid/unified_round5/clusters_recognizable_round2_k10.csv")
    parser.add_argument("--deer-map-csv", default="data/reid/unified_round5/deer_id_proposal_recognizable_round2_k10.csv")
    parser.add_argument("--target-cluster-id", type=int, default=None)
    parser.add_argument("--target-deer-id", default="")
    parser.add_argument("--sample-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-clusters-csv", default="data/reid/unified_round5/clusters_recognizable_round2_k10_corrected.csv")
    parser.add_argument("--outlier-queue-csv", default="data/reid/unified_round5/cluster_outlier_queue.csv")
    parser.add_argument("--decisions-csv", default="data/reid/unified_round5/cluster_outlier_review_decisions.csv")
    parser.add_argument("--out-deer-map-csv", default="data/reid/unified_round5/deer_id_proposal_recognizable_round2_k10_corrected.csv")
    parser.add_argument("--embeddings", default="data/reid/unified_round5/embeddings_recognizable_round2.pt")
    parser.add_argument("--metadata", default="data/reid/unified_round5/embeddings_recognizable_round2.csv")
    parser.add_argument("--assign-threshold", type=float, default=0.94)
    parser.add_argument("--join-candidates", type=int, default=9)
    parser.add_argument(
        "--actions-globs",
        default="data/pool/**/actions.json,runs/**/04_tracklets/actions.json",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    actions_globs = [x.strip() for x in args.actions_globs.split(",") if x.strip()]
    source_lookup = build_source_lookup(repo_root, actions_globs)
    all_rows = load_clusters(Path(args.clusters_csv))
    emb_by_path = load_embeddings_by_path(Path(args.embeddings), Path(args.metadata))
    deer_map = load_deer_map(Path(args.deer_map_csv) if args.deer_map_csv else None)

    members: dict[int, list[str]] = {}
    for row in all_rows:
        members.setdefault(row.cluster_id, []).append(row.image_path)

    current_assign: dict[str, int] = {row.image_path: row.cluster_id for row in all_rows}

    selected_clusters = sorted(members.keys())
    if args.target_cluster_id is not None:
        selected_clusters = [c for c in selected_clusters if c == args.target_cluster_id]
    if args.target_deer_id:
        want = args.target_deer_id.strip()
        selected_clusters = [c for c in selected_clusters if deer_map.get(c, "") == want]

    if not selected_clusters:
        print("No clusters selected for review.")
        return

    rng = random.Random(args.seed)
    for cid in selected_clusters:
        deer_id = deer_map.get(cid, f"CLUSTER_{cid:03d}")
        batch = 0
        while True:
            current = [p for p in members.get(cid, []) if current_assign.get(p) == cid]
            if len(current) < 1:
                break

            cluster_vecs = [emb_by_path[p] for p in current if p in emb_by_path]
            merge_candidates: list[dict[str, Any]] = []
            if cluster_vecs:
                c_centroid = torch.stack(cluster_vecs, dim=0).mean(dim=0)
                c_centroid = c_centroid / c_centroid.norm()
                for ocid, opaths in members.items():
                    if ocid == cid or ocid < 0:
                        continue
                    o_current = [p for p in opaths if current_assign.get(p) == ocid]
                    o_vecs = [emb_by_path[p] for p in o_current if p in emb_by_path]
                    if not o_vecs:
                        continue
                    o_centroid = torch.stack(o_vecs, dim=0).mean(dim=0)
                    o_centroid = o_centroid / o_centroid.norm()
                    score = float((c_centroid @ o_centroid).item())
                    merge_candidates.append(
                        {
                            "cluster_id": ocid,
                            "deer_id": deer_map.get(ocid, f"CLUSTER_{ocid:03d}"),
                            "score": score,
                        }
                    )
                merge_candidates.sort(key=lambda x: x["score"], reverse=True)
                merge_candidates = merge_candidates[: max(1, args.join_candidates)]

            map_text = cluster_size_map(members, current_assign, deer_map)
            pending_count = waiting_reassign_count(Path(args.outlier_queue_csv))

            k = min(max(1, args.sample_size), 4, len(current))
            sample = rng.sample(current, k)
            batch += 1
            result = review_batch(
                repo_root,
                source_lookup,
                cid,
                deer_id,
                sample,
                merge_candidates,
                map_text,
                pending_count,
                batch,
            )
            if result is False:
                print("Stopped by user.")
                write_clusters(
                    Path(args.out_clusters_csv),
                    [ClusterRow(image_path=path, cluster_id=cluster) for path, cluster in sorted(current_assign.items())],
                )
                write_deer_map(Path(args.out_deer_map_csv), deer_map, members)
                print("Wrote", args.out_clusters_csv)
                return

            if not isinstance(result, dict):
                append_decision(Path(args.decisions_csv), cid, deer_id, sample, [], [], "invalid_result")
                continue

            action = str(result.get("action", "keep_all"))
            removed = list(result.get("paths", []))
            assignments: list[dict[str, str]] = []

            if action == "rename_cluster":
                new_name = str(result.get("name", "")).strip()
                if new_name:
                    deer_map[cid] = new_name
                    deer_id = new_name
                append_decision(Path(args.decisions_csv), cid, deer_id, sample, [], [], action)
                continue

            if action == "merge_cluster":
                target_cluster = result.get("target_cluster_id")
                if target_cluster is None:
                    append_decision(Path(args.decisions_csv), cid, deer_id, sample, [], [], "merge_cluster_invalid")
                    continue
                target_cluster = int(target_cluster)
                members.setdefault(target_cluster, [])
                deer_map.setdefault(target_cluster, str(result.get("target_deer_id") or f"CLUSTER_{target_cluster:03d}"))
                status = "merge_cluster_candidate"
                if target_cluster == cid:
                    append_decision(Path(args.decisions_csv), cid, deer_id, sample, [], [], "merge_cluster_self_skip")
                    continue
                moved = [p for p in members.get(cid, []) if current_assign.get(p) == cid]
                for p in moved:
                    current_assign[p] = target_cluster
                    members.setdefault(target_cluster, []).append(p)
                members[cid] = [p for p in members.get(cid, []) if current_assign.get(p) == cid]
                assignments = [
                    {
                        "image_path": p,
                        "assigned_cluster_id": str(target_cluster),
                        "assigned_score": "",
                        "status": status,
                    }
                    for p in moved
                ]
                append_decision(Path(args.decisions_csv), cid, deer_id, sample, moved, assignments, action)
                continue

            if action == "merge_cluster_new":
                moved = [p for p in members.get(cid, []) if current_assign.get(p) == cid]
                if not moved:
                    append_decision(Path(args.decisions_csv), cid, deer_id, sample, [], [], "merge_cluster_new_empty")
                    continue
                next_id = (max(members.keys()) + 1) if members else 0
                members[next_id] = []
                deer_map[next_id] = f"DEER_{next_id+1:03d}"
                for p in moved:
                    current_assign[p] = next_id
                    members[next_id].append(p)
                members[cid] = []
                assignments = [
                    {
                        "image_path": p,
                        "assigned_cluster_id": str(next_id),
                        "assigned_score": "",
                        "status": "merge_cluster_new",
                    }
                    for p in moved
                ]
                append_decision(Path(args.decisions_csv), cid, deer_id, sample, moved, assignments, action)
                continue

            if action == "manual_assign":
                if len(removed) != 1:
                    print("Manual assign requires selecting exactly one image with keys 1..4.")
                    append_decision(Path(args.decisions_csv), cid, deer_id, sample, [], [], "manual_assign_invalid")
                    continue
                target = input("Assign selected image to cluster-id or deer-id (new name creates cluster): ").strip()
                target_cluster, status = resolve_target_cluster(target, deer_map, members)
                p = removed[0]
                members[cid] = [x for x in members.get(cid, []) if x != p]
                members.setdefault(target_cluster, []).append(p)
                current_assign[p] = target_cluster
                assignments.append(
                    {
                        "image_path": p,
                        "assigned_cluster_id": str(target_cluster),
                        "assigned_score": "",
                        "status": status,
                    }
                )
                append_outlier_queue(
                    Path(args.outlier_queue_csv),
                    p,
                    cid,
                    deer_id,
                    target_cluster,
                    None,
                    status,
                )
                append_decision(Path(args.decisions_csv), cid, deer_id, sample, removed, assignments, action)
                continue

            if action == "all_different":
                used_targets: set[int] = set()
                for p in removed:
                    members[cid] = [x for x in members.get(cid, []) if x != p]
                    target_cluster, score, status = assign_removed_image(
                        p,
                        from_cluster_id=cid,
                        members=members,
                        emb_by_path=emb_by_path,
                        threshold=args.assign_threshold,
                    )
                    if target_cluster in used_targets:
                        target_cluster = (max(members.keys()) + 1) if members else 0
                        members[target_cluster] = []
                        deer_map[target_cluster] = f"DEER_{target_cluster+1:03d}"
                        status = "new_cluster_all_different"
                        score = None
                    used_targets.add(target_cluster)
                    members.setdefault(target_cluster, []).append(p)
                    current_assign[p] = target_cluster
                    assignments.append(
                        {
                            "image_path": p,
                            "assigned_cluster_id": str(target_cluster),
                            "assigned_score": "" if score is None else f"{score:.6f}",
                            "status": status,
                        }
                    )
                    append_outlier_queue(
                        Path(args.outlier_queue_csv),
                        p,
                        cid,
                        deer_id,
                        target_cluster,
                        score,
                        status,
                    )
                append_decision(Path(args.decisions_csv), cid, deer_id, sample, removed, assignments, action)
                continue

            for removed_path in removed:
                if removed_path in members.get(cid, []):
                    members[cid] = [p for p in members[cid] if p != removed_path]
                target_cluster, score, status = assign_removed_image(
                    removed_path,
                    from_cluster_id=cid,
                    members=members,
                    emb_by_path=emb_by_path,
                    threshold=args.assign_threshold,
                )
                members.setdefault(target_cluster, []).append(removed_path)
                current_assign[removed_path] = target_cluster
                assignments.append(
                    {
                        "image_path": removed_path,
                        "assigned_cluster_id": str(target_cluster),
                        "assigned_score": "" if score is None else f"{score:.6f}",
                        "status": status,
                    }
                )
                append_outlier_queue(
                    Path(args.outlier_queue_csv),
                    removed_path,
                    cid,
                    deer_id,
                    target_cluster,
                    score,
                    status,
                )

            append_decision(Path(args.decisions_csv), cid, deer_id, sample, removed, assignments, action)
            if not removed:
                break

    corrected = [ClusterRow(image_path=path, cluster_id=cluster) for path, cluster in sorted(current_assign.items())]
    write_clusters(Path(args.out_clusters_csv), corrected)
    write_deer_map(Path(args.out_deer_map_csv), deer_map, members)
    print("Wrote", args.out_clusters_csv)
    print("Wrote", args.out_deer_map_csv)
    print("Reviewed clusters:", len(selected_clusters))
    print("Outlier queue:", args.outlier_queue_csv)


if __name__ == "__main__":
    main()
