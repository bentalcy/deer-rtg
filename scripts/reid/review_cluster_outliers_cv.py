import argparse
import csv
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
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
            if image_path and cluster_raw:
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


def write_clusters(path: Path, rows: list[ClusterRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_path", "cluster_id"])
        w.writeheader()
        for r in rows:
            w.writerow({"image_path": r.image_path, "cluster_id": r.cluster_id})


def write_deer_map(path: Path, deer_map: dict[int, str], members: dict[int, list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cluster_id", "cluster_size", "deer_id"])
        w.writeheader()
        for cid in sorted(k for k in members.keys() if k >= 0):
            w.writerow(
                {
                    "cluster_id": cid,
                    "cluster_size": len(members.get(cid, [])),
                    "deer_id": deer_map.get(cid, f"CLUSTER_{cid:03d}"),
                }
            )


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


def extract_meta(path: Path, source_path: Path | None) -> tuple[str, str]:
    meta_path = source_path if source_path is not None and source_path.exists() else path
    dt = "n/a"
    loc = "n/a"
    try:
        exif = Image.open(meta_path).getexif()
        if exif:
            date_time = exif.get(306) or exif.get(36867)
            if date_time:
                dt = str(date_time)
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
                    loc = f"{lat:.6f}, {lon:.6f}"
    except Exception:
        pass
    if dt == "n/a":
        m = re.search(r"(\d{4}-\d{2}-\d{2})[ _](\d{2}-\d{2}-\d{2})", path.name)
        if m:
            dt = f"{m.group(1)} {m.group(2)}"
    return dt, loc


def load_embeddings_by_path(embeddings_path: Path, metadata_path: Path) -> dict[str, torch.Tensor]:
    emb = torch.load(embeddings_path, map_location="cpu").float()
    emb = emb / emb.norm(dim=-1, keepdim=True)
    rows = list(csv.DictReader(metadata_path.open(newline="")))
    if emb.size(0) != len(rows):
        raise ValueError("embeddings and metadata length mismatch")
    out: dict[str, torch.Tensor] = {}
    for i, row in enumerate(rows):
        p = (row.get("image_path") or "").strip()
        if p:
            out[p] = emb[i]
    return out


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


def resolve_target_cluster(target: str, deer_map: dict[int, str], members: dict[int, list[str]]) -> tuple[int, str]:
    t = target.strip()
    if not t:
        nid = (max(members.keys()) + 1) if members else 0
        members[nid] = []
        deer_map[nid] = f"DEER_{nid+1:03d}"
        return nid, "new_cluster_empty_target"
    if t.isdigit():
        cid = int(t)
        members.setdefault(cid, [])
        deer_map.setdefault(cid, f"CLUSTER_{cid:03d}")
        return cid, "manual_cluster_id"
    for cid, name in deer_map.items():
        if name == t:
            members.setdefault(cid, [])
            return cid, "manual_deer_id"
    nid = (max(members.keys()) + 1) if members else 0
    members[nid] = []
    deer_map[nid] = t
    return nid, "new_cluster_named"


def waiting_reassign_count(outlier_queue_csv: Path) -> int:
    if not outlier_queue_csv.exists():
        return 0
    latest: dict[str, str] = {}
    for row in csv.DictReader(outlier_queue_csv.open(newline="")):
        p = (row.get("image_path") or "").strip()
        s = (row.get("status") or "").strip()
        if p:
            latest[p] = s
    return sum(1 for s in latest.values() if s == "needs_reassign")


def cluster_map_lines(members: dict[int, list[str]], current_assign: dict[str, int], deer_map: dict[int, str]) -> list[str]:
    counts: list[tuple[int, int]] = []
    for cid, paths in members.items():
        if cid < 0:
            continue
        c = sum(1 for p in paths if current_assign.get(p) == cid)
        counts.append((cid, c))
    counts.sort(key=lambda kv: (-kv[1], kv[0]))
    return [f"{deer_map.get(cid, f'CLUSTER_{cid:03d}')} [{cid}] = {count}" for cid, count in counts]


def make_canvas(
    repo_root: Path,
    source_lookup: dict[str, Path],
    cluster_id: int,
    deer_id: str,
    sample: list[str],
    selected: set[int],
    waiting_count: int,
    cmap_lines: list[str],
    mode: str,
    candidates: list[dict],
    candidate_idx: int,
) -> "cv2.typing.MatLike":
    h, w = 980, 1720
    left_w = 1140
    canvas = 255 * torch.ones((h, w, 3), dtype=torch.uint8).numpy()
    canvas[:] = (245, 246, 248)

    cv2.rectangle(canvas, (0, 0), (left_w - 1, h - 1), (230, 230, 230), 1)
    cv2.rectangle(canvas, (left_w, 0), (w - 1, h - 1), (220, 220, 220), 1)

    title = f"Cluster {cluster_id} | Deer: {deer_id} | waiting reassignment: {waiting_count}"
    cv2.putText(canvas, title, (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (35, 35, 35), 2, cv2.LINE_AA)
    ctrl = "keys: 1-4 toggle | c remove | 0 keep | a all-diff | k assign selected | n rename | j join | q quit"
    cv2.putText(canvas, ctrl, (18, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (45, 45, 45), 1, cv2.LINE_AA)

    cell_w, cell_h = 540, 410
    starts = [(20, 80), (580, 80), (20, 500), (580, 500)]
    for i in range(4):
        x, y = starts[i]
        cv2.rectangle(canvas, (x, y), (x + cell_w, y + cell_h), (180, 180, 180), 1)
        if i >= len(sample):
            continue
        p = to_abs(repo_root, sample[i])
        if p.exists():
            img = cv2.imread(str(p))
            if img is not None:
                ih, iw = img.shape[:2]
                scale = min((cell_w - 10) / max(1, iw), (cell_h - 70) / max(1, ih))
                nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
                img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
                ox = x + (cell_w - nw) // 2
                oy = y + 6
                canvas[oy : oy + nh, ox : ox + nw] = img

        key_tag = f"[{i+1}]"
        label_color = (20, 120, 20) if i in selected else (50, 50, 50)
        cv2.putText(canvas, key_tag, (x + 8, y + cell_h - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.65, label_color, 2, cv2.LINE_AA)

        source = source_lookup.get(Path(sample[i]).name) or source_lookup.get(normalize_crop_basename(Path(sample[i]).name))
        dt, loc = extract_meta(p, source if isinstance(source, Path) else None)
        cv2.putText(canvas, Path(sample[i]).name[:62], (x + 56, y + cell_h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"time: {dt}", (x + 8, y + cell_h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (55, 55, 55), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"loc: {loc}", (x + 8, y + cell_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (55, 55, 55), 1, cv2.LINE_AA)

    px = left_w + 14
    cv2.putText(canvas, "Cluster Map (name [id] = count)", (px, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (35, 35, 35), 1, cv2.LINE_AA)
    y = 50
    for line in cmap_lines:
        cv2.putText(canvas, line[:74], (px, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (60, 60, 60), 1, cv2.LINE_AA)
        y += 18
        if y > h - 180:
            break

    by = h - 140
    if mode == "join":
        cv2.putText(canvas, "JOIN MODE: y=choose shown, n=next, 1..9 jump, x=new cluster, esc=cancel", (px, by), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 120), 1, cv2.LINE_AA)
        if candidates:
            c = candidates[candidate_idx]
            txt = f"candidate {candidate_idx+1}/{len(candidates)} -> {c['deer_id']} [{c['cluster_id']}] score={c['score']:.3f}"
            cv2.putText(canvas, txt[:74], (px, by + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 120), 1, cv2.LINE_AA)
    elif mode == "assign":
        cv2.putText(canvas, "ASSIGN MODE: 1..9 pick candidate, x=new cluster, esc=cancel", (px, by), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 50, 20), 1, cv2.LINE_AA)
    elif mode == "rename":
        cv2.putText(canvas, "RENAME MODE: type name, Enter=save, Backspace=del, Esc=cancel", (px, by), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 90, 20), 1, cv2.LINE_AA)

    return canvas


def compute_merge_candidates(
    cid: int,
    members: dict[int, list[str]],
    current_assign: dict[str, int],
    emb_by_path: dict[str, torch.Tensor],
    deer_map: dict[int, str],
    limit: int,
) -> list[dict]:
    current = [p for p in members.get(cid, []) if current_assign.get(p) == cid and p in emb_by_path]
    if not current:
        return []
    c_cent = torch.stack([emb_by_path[p] for p in current], dim=0).mean(dim=0)
    c_cent = c_cent / c_cent.norm()
    out: list[dict] = []
    for ocid, opaths in members.items():
        if ocid == cid or ocid < 0:
            continue
        o = [p for p in opaths if current_assign.get(p) == ocid and p in emb_by_path]
        if not o:
            continue
        o_cent = torch.stack([emb_by_path[p] for p in o], dim=0).mean(dim=0)
        o_cent = o_cent / o_cent.norm()
        score = float((c_cent @ o_cent).item())
        out.append({"cluster_id": ocid, "deer_id": deer_map.get(ocid, f"CLUSTER_{ocid:03d}"), "score": score})
    out.sort(key=lambda x: x["score"], reverse=True)
    return out[: max(1, limit)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clusters-csv", default="data/reid/unified_round5/clusters_recognizable_round2_k10_corrected.csv")
    parser.add_argument("--deer-map-csv", default="data/reid/unified_round5/deer_id_proposal_recognizable_round2_k10_corrected.csv")
    parser.add_argument("--sample-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-clusters-csv", default="data/reid/unified_round5/clusters_recognizable_round2_k10_corrected.csv")
    parser.add_argument("--outlier-queue-csv", default="data/reid/unified_round5/cluster_outlier_queue.csv")
    parser.add_argument("--decisions-csv", default="data/reid/unified_round5/cluster_outlier_review_decisions_v3.csv")
    parser.add_argument("--out-deer-map-csv", default="data/reid/unified_round5/deer_id_proposal_recognizable_round2_k10_corrected.csv")
    parser.add_argument("--embeddings", default="data/reid/unified_round5/embeddings_recognizable_round2.pt")
    parser.add_argument("--metadata", default="data/reid/unified_round5/embeddings_recognizable_round2.csv")
    parser.add_argument("--assign-threshold", type=float, default=0.94)
    parser.add_argument("--join-candidates", type=int, default=9)
    parser.add_argument("--batches-per-cluster", type=int, default=1)
    parser.add_argument("--min-review-cluster-size", type=int, default=2)
    parser.add_argument("--auto-assign-singletons", action="store_true", default=True)
    parser.add_argument("--no-auto-assign-singletons", dest="auto_assign_singletons", action="store_false")
    parser.add_argument("--actions-globs", default="data/pool/**/actions.json,runs/**/04_tracklets/actions.json")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    actions_globs = [x.strip() for x in args.actions_globs.split(",") if x.strip()]
    source_lookup = build_source_lookup(repo_root, actions_globs)
    all_rows = load_clusters(Path(args.clusters_csv))
    deer_map = load_deer_map(Path(args.deer_map_csv))
    emb_by_path = load_embeddings_by_path(Path(args.embeddings), Path(args.metadata))

    members: dict[int, list[str]] = {}
    for r in all_rows:
        members.setdefault(r.cluster_id, []).append(r.image_path)
    current_assign = {r.image_path: r.cluster_id for r in all_rows}

    rng = random.Random(args.seed)
    cv2.namedWindow("cluster_review", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("cluster_review", 1720, 980)

    for cid in sorted(list(members.keys())):
        cluster_batches = 0
        while True:
            current = [p for p in members.get(cid, []) if current_assign.get(p) == cid]
            if len(current) == 1 and args.auto_assign_singletons:
                deer_id = deer_map.get(cid, f"CLUSTER_{cid:03d}")
                p = current[0]
                members[cid] = []
                tgt, score, status = assign_removed_image(p, cid, members, emb_by_path, args.assign_threshold)
                members.setdefault(tgt, []).append(p)
                current_assign[p] = tgt
                append_outlier_queue(Path(args.outlier_queue_csv), p, cid, deer_id, tgt, score, status)
                assign_rows = [
                    {
                        "image_path": p,
                        "assigned_cluster_id": str(tgt),
                        "assigned_score": "" if score is None else f"{score:.6f}",
                        "status": status,
                    }
                ]
                append_decision(Path(args.decisions_csv), cid, deer_id, [p], [p], assign_rows, "singleton_auto_assign")
                continue
            if len(current) < max(1, args.min_review_cluster_size):
                break
            if cluster_batches >= max(1, args.batches_per_cluster):
                break
            k = min(max(1, args.sample_size), 4, len(current))
            sample = rng.sample(current, k)
            selected: set[int] = set()
            mode = "normal"
            rename_text = ""
            merge_candidates = compute_merge_candidates(cid, members, current_assign, emb_by_path, deer_map, args.join_candidates)
            cand_idx = 0
            action_taken = False

            while True:
                waiting = waiting_reassign_count(Path(args.outlier_queue_csv))
                cmap = cluster_map_lines(members, current_assign, deer_map)
                deer_id = deer_map.get(cid, f"CLUSTER_{cid:03d}")
                canvas = make_canvas(
                    repo_root,
                    source_lookup,
                    cid,
                    deer_id,
                    sample,
                    selected,
                    waiting,
                    cmap,
                    mode,
                    merge_candidates,
                    cand_idx,
                )
                if mode == "rename":
                    cv2.putText(canvas, f"rename text: {rename_text}", (1154, 910), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 90, 20), 2, cv2.LINE_AA)
                cv2.imshow("cluster_review", canvas)
                key = cv2.waitKey(0) & 0xFF

                if mode == "rename":
                    if key == 27:
                        mode = "normal"
                        rename_text = ""
                        continue
                    if key in (13, 10):
                        t = rename_text.strip()
                        if t:
                            deer_map[cid] = t
                            append_decision(Path(args.decisions_csv), cid, deer_map[cid], sample, [], [], "rename_cluster")
                        mode = "normal"
                        rename_text = ""
                        continue
                    if key in (8, 127):
                        rename_text = rename_text[:-1]
                        continue
                    ch = chr(key) if 32 <= key <= 126 else ""
                    if ch and (ch.isalnum() or ch in "_-"):
                        rename_text += ch
                    continue

                if mode == "join":
                    if key == 27:
                        mode = "normal"
                        continue
                    if key == ord("y"):
                        if merge_candidates:
                            tgt = int(merge_candidates[cand_idx]["cluster_id"])
                            moved = [p for p in members.get(cid, []) if current_assign.get(p) == cid]
                            for p in moved:
                                current_assign[p] = tgt
                                members.setdefault(tgt, []).append(p)
                            members[cid] = []
                            assign_rows = [{"image_path": p, "assigned_cluster_id": str(tgt), "assigned_score": "", "status": "merge_cluster_candidate"} for p in moved]
                            append_decision(Path(args.decisions_csv), cid, deer_id, sample, moved, assign_rows, "merge_cluster")
                        action_taken = True
                        mode = "normal"
                        break
                    if key == ord("n"):
                        if merge_candidates:
                            cand_idx = (cand_idx + 1) % len(merge_candidates)
                        continue
                    if key == ord("x"):
                        nid = (max(members.keys()) + 1) if members else 0
                        members[nid] = []
                        deer_map[nid] = f"DEER_{nid+1:03d}"
                        moved = [p for p in members.get(cid, []) if current_assign.get(p) == cid]
                        for p in moved:
                            current_assign[p] = nid
                            members[nid].append(p)
                        members[cid] = []
                        assign_rows = [{"image_path": p, "assigned_cluster_id": str(nid), "assigned_score": "", "status": "merge_cluster_new"} for p in moved]
                        append_decision(Path(args.decisions_csv), cid, deer_id, sample, moved, assign_rows, "merge_cluster_new")
                        action_taken = True
                        mode = "normal"
                        break
                    if ord("1") <= key <= ord("9"):
                        idx = key - ord("1")
                        if 0 <= idx < len(merge_candidates):
                            cand_idx = idx
                        continue
                    continue

                if mode == "assign":
                    if key == 27:
                        mode = "normal"
                        continue
                    if key == ord("x"):
                        if not selected:
                            mode = "normal"
                            continue
                        removed = [sample[i] for i in sorted(selected)]
                        assign_rows = []
                        for p in removed:
                            members[cid] = [x for x in members.get(cid, []) if x != p]
                            nid = (max(members.keys()) + 1) if members else 0
                            members[nid] = [p]
                            deer_map[nid] = f"DEER_{nid+1:03d}"
                            current_assign[p] = nid
                            assign_rows.append({"image_path": p, "assigned_cluster_id": str(nid), "assigned_score": "", "status": "manual_assign_new"})
                            append_outlier_queue(Path(args.outlier_queue_csv), p, cid, deer_id, nid, None, "manual_assign_new")
                        append_decision(Path(args.decisions_csv), cid, deer_id, sample, removed, assign_rows, "manual_assign")
                        action_taken = True
                        mode = "normal"
                        break
                    if ord("1") <= key <= ord("9"):
                        idx = key - ord("1")
                        if idx < len(merge_candidates) and selected:
                            tgt = int(merge_candidates[idx]["cluster_id"])
                            removed = [sample[i] for i in sorted(selected)]
                            assign_rows = []
                            for p in removed:
                                members[cid] = [x for x in members.get(cid, []) if x != p]
                                members.setdefault(tgt, []).append(p)
                                current_assign[p] = tgt
                                assign_rows.append({"image_path": p, "assigned_cluster_id": str(tgt), "assigned_score": "", "status": "manual_assign"})
                                append_outlier_queue(Path(args.outlier_queue_csv), p, cid, deer_id, tgt, None, "manual_assign")
                            append_decision(Path(args.decisions_csv), cid, deer_id, sample, removed, assign_rows, "manual_assign")
                            action_taken = True
                            mode = "normal"
                            break
                    continue

                if key == ord("q"):
                    corrected = [ClusterRow(image_path=p, cluster_id=c) for p, c in sorted(current_assign.items())]
                    write_clusters(Path(args.out_clusters_csv), corrected)
                    write_deer_map(Path(args.out_deer_map_csv), deer_map, members)
                    cv2.destroyAllWindows()
                    print("Saved and stopped by user")
                    return
                if key in (ord("1"), ord("2"), ord("3"), ord("4")):
                    idx = key - ord("1")
                    if idx < len(sample):
                        if idx in selected:
                            selected.remove(idx)
                        elif len(selected) < 2:
                            selected.add(idx)
                    continue
                if key == ord("0"):
                    append_decision(Path(args.decisions_csv), cid, deer_id, sample, [], [], "keep_all")
                    action_taken = True
                    break
                if key == ord("n"):
                    mode = "rename"
                    rename_text = ""
                    continue
                if key == ord("j"):
                    mode = "join"
                    cand_idx = 0
                    continue
                if key == ord("k"):
                    mode = "assign"
                    continue
                if key == ord("a"):
                    removed = list(sample)
                    assign_rows = []
                    used_targets: set[int] = set()
                    for p in removed:
                        members[cid] = [x for x in members.get(cid, []) if x != p]
                        tgt, score, status = assign_removed_image(p, cid, members, emb_by_path, args.assign_threshold)
                        if tgt in used_targets:
                            tgt = (max(members.keys()) + 1) if members else 0
                            members[tgt] = []
                            deer_map[tgt] = f"DEER_{tgt+1:03d}"
                            status = "new_cluster_all_different"
                            score = None
                        used_targets.add(tgt)
                        members.setdefault(tgt, []).append(p)
                        current_assign[p] = tgt
                        assign_rows.append({"image_path": p, "assigned_cluster_id": str(tgt), "assigned_score": "" if score is None else f"{score:.6f}", "status": status})
                        append_outlier_queue(Path(args.outlier_queue_csv), p, cid, deer_id, tgt, score, status)
                    append_decision(Path(args.decisions_csv), cid, deer_id, sample, removed, assign_rows, "all_different")
                    action_taken = True
                    break
                if key == ord("c"):
                    if not selected:
                        continue
                    removed = [sample[i] for i in sorted(selected)]
                    assign_rows = []
                    for p in removed:
                        members[cid] = [x for x in members.get(cid, []) if x != p]
                        tgt, score, status = assign_removed_image(p, cid, members, emb_by_path, args.assign_threshold)
                        members.setdefault(tgt, []).append(p)
                        current_assign[p] = tgt
                        assign_rows.append({"image_path": p, "assigned_cluster_id": str(tgt), "assigned_score": "" if score is None else f"{score:.6f}", "status": status})
                        append_outlier_queue(Path(args.outlier_queue_csv), p, cid, deer_id, tgt, score, status)
                    append_decision(Path(args.decisions_csv), cid, deer_id, sample, removed, assign_rows, "remove")
                    action_taken = True
                    break

            if action_taken:
                cluster_batches += 1

    corrected = [ClusterRow(image_path=p, cluster_id=c) for p, c in sorted(current_assign.items())]
    write_clusters(Path(args.out_clusters_csv), corrected)
    write_deer_map(Path(args.out_deer_map_csv), deer_map, members)
    cv2.destroyAllWindows()
    print("Wrote", args.out_clusters_csv)
    print("Wrote", args.out_deer_map_csv)


if __name__ == "__main__":
    main()
