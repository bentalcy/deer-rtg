import argparse
import csv
import json
import random
import re
import sys
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from random import Random
from typing import cast
from urllib.parse import parse_qs, urlparse

import torch
from torch.nn import functional as F

if __package__ in {None, ""}:
    repo_path = Path(__file__).resolve().parents[2]
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))

from scripts.reid.review_cluster_outliers_cv import (
    ClusterRow,
    append_decision,
    append_outlier_queue,
    cluster_map_lines,
    load_clusters,
    load_deer_map,
    load_embeddings_by_path,
    waiting_reassign_count,
    write_clusters,
    write_deer_map,
)


_CAPTURE_TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2})[ _](\d{2})-(\d{2})-(\d{2})")
_ALLOWED_SIDES = {"left", "right", "unknown"}
_SEQ_SUFFIX_RE = re.compile(r"^(.*?)(?:_det\d+)?$")
_SEQ_ID_RE = re.compile(r"^(.*?)(\d+)$")
_PAD_SUFFIX = "__pad"


def image_capture_time(image_path: str) -> str:
    name = Path(image_path).name
    match = _CAPTURE_TS_RE.search(name)
    if match is None:
        return ""
    date_part, hour, minute, second = match.groups()
    return f"{date_part} {hour}:{minute}:{second}"


def parse_sequence_key(image_path: str) -> tuple[str, int] | None:
    name = Path(image_path).stem
    match = _SEQ_SUFFIX_RE.match(name)
    base = match.group(1) if match else name
    seq = _SEQ_ID_RE.match(base)
    if seq is None:
        return None
    prefix = seq.group(1)
    number_raw = seq.group(2)
    if not prefix or not number_raw:
        return None
    return prefix, int(number_raw)


def build_sequence_index(rows: list[ClusterRow]) -> dict[str, list[tuple[int, str]]]:
    out: dict[str, list[tuple[int, str]]] = {}
    for row in rows:
        key = parse_sequence_key(row.image_path)
        if key is None:
            continue
        prefix, number = key
        out.setdefault(prefix, []).append((number, row.image_path))
    for prefix in out:
        out[prefix].sort(key=lambda pair: pair[0])
    return out


def load_side_predictions(metadata_path: Path) -> dict[str, str]:
    if not metadata_path.exists():
        return {}
    out: dict[str, str] = {}
    with metadata_path.open(newline="") as f:
        rows = csv.DictReader(f)
        for row in rows:
            image_path = str(row.get("image_path") or "").strip()
            side = str(row.get("side_pred") or "").strip().lower()
            if not image_path or side not in _ALLOWED_SIDES:
                continue
            out[image_path] = side
    return out


def load_user_side_labels(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    with path.open(newline="") as f:
        rows = csv.DictReader(f)
        for row in rows:
            image_path = str(row.get("image_path") or "").strip()
            side = str(row.get("side_label") or "").strip().lower()
            if not image_path or side not in _ALLOWED_SIDES:
                continue
            out[image_path] = side
    return out


def write_user_side_labels(path: Path, labels: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "side_label"])
        writer.writeheader()
        for image_path in sorted(labels.keys()):
            side = labels[image_path]
            if side not in _ALLOWED_SIDES:
                continue
            writer.writerow({"image_path": image_path, "side_label": side})


def load_removed_images(path: Path) -> list[str]:
    if not path.exists():
        return []
    out: list[str] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            image_path = str(row.get("image_path") or "").strip()
            if image_path:
                out.append(image_path)
    return out


def append_removed_images(path: Path, image_paths: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "image_path", "reason"])
        if not existing:
            writer.writeheader()
        ts = str(int(time.time()))
        for image_path in image_paths:
            writer.writerow({"timestamp": ts, "image_path": image_path, "reason": "removed_from_pool"})


def load_recent_outlier_moves(path: Path, limit: int = 10) -> list[dict[str, str]]:
    if not path.exists():
        return []
    rows: list[dict[str, str]] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "timestamp": str(row.get("timestamp") or ""),
                    "image_path": str(row.get("image_path") or ""),
                    "from_cluster_id": str(row.get("from_cluster_id") or ""),
                    "assigned_cluster_id": str(row.get("assigned_cluster_id") or ""),
                    "status": str(row.get("status") or ""),
                    "assigned_score": str(row.get("assigned_score") or ""),
                }
            )
    if limit <= 0:
        return rows
    return rows[-limit:]


@dataclass
class ReviewConfig:
    clusters_csv: Path
    deer_map_csv: Path
    out_clusters_csv: Path
    outlier_queue_csv: Path
    decisions_csv: Path
    out_deer_map_csv: Path
    side_labels_csv: Path
    removed_images_csv: Path
    embeddings: Path
    metadata: Path
    sample_size: int
    seed: int
    assign_threshold: float
    join_candidates: int
    batches_per_cluster: int
    min_review_cluster_size: int
    auto_assign_singletons: bool


class ClusterReviewEngine:
    def __init__(self, cfg: ReviewConfig) -> None:
        self.cfg: ReviewConfig = cfg
        self.rows: list[ClusterRow] = load_clusters(cfg.clusters_csv)
        self.deer_map: dict[int, str] = load_deer_map(cfg.deer_map_csv)
        self.emb_by_path: dict[str, torch.Tensor] = load_embeddings_by_path(cfg.embeddings, cfg.metadata)
        self.predicted_side_by_path: dict[str, str] = load_side_predictions(cfg.metadata)
        self.user_side_labels: dict[str, str] = load_user_side_labels(cfg.side_labels_csv)
        self.removed_images: set[str] = set(load_removed_images(cfg.removed_images_csv))
        self.members: dict[int, list[str]] = {}
        for row in self.rows:
            self.members.setdefault(row.cluster_id, []).append(row.image_path)
        self.current_assign: dict[str, int] = {row.image_path: row.cluster_id for row in self.rows}
        self.cluster_order: list[int] = sorted(self.members.keys())
        self.batches_done: dict[int, int] = {cid: 0 for cid in self.cluster_order}
        self.cluster_idx: int = 0
        self.rng: Random = random.Random(cfg.seed)
        self.finished: bool = False
        self.current_cluster_id: int | None = None
        self.current_sample: list[str] = []
        self.current_candidates: list[dict[str, object]] = []
        self.last_action_message: str = ""
        self.recent_moves: list[dict[str, str]] = load_recent_outlier_moves(cfg.outlier_queue_csv, limit=10)
        self.sequence_index: dict[str, list[tuple[int, str]]] = build_sequence_index(self.rows)
        self._advance()

    def _active_paths(self, cluster_id: int) -> list[str]:
        return [
            p
            for p in self.members.get(cluster_id, [])
            if self.current_assign.get(p) == cluster_id and p not in self.removed_images
        ]

    def _register_cluster_if_needed(self, cluster_id: int) -> None:
        if cluster_id in self.cluster_order:
            return
        self.cluster_order.append(cluster_id)
        self.cluster_order.sort()
        _ = self.batches_done.setdefault(cluster_id, 0)

    def _record_move(
        self,
        image_path: str,
        from_cluster_id: int,
        assigned_cluster_id: int,
        status: str,
        assigned_score: float | None,
    ) -> None:
        row = {
            "timestamp": str(int(time.time())),
            "image_path": image_path,
            "from_cluster_id": str(from_cluster_id),
            "assigned_cluster_id": str(assigned_cluster_id),
            "status": status,
            "assigned_score": "" if assigned_score is None else f"{assigned_score:.6f}",
        }
        self.recent_moves.append(row)
        if len(self.recent_moves) > 10:
            self.recent_moves = self.recent_moves[-10:]

    def _append_outlier_move(
        self,
        image_path: str,
        from_cluster_id: int,
        deer_id_likely: str,
        assigned_cluster_id: int,
        assigned_score: float | None,
        status: str,
    ) -> None:
        append_outlier_queue(
            self.cfg.outlier_queue_csv,
            image_path,
            from_cluster_id,
            deer_id_likely,
            assigned_cluster_id,
            assigned_score,
            status,
        )
        self._record_move(image_path, from_cluster_id, assigned_cluster_id, status, assigned_score)

    def _next_cluster(self) -> int | None:
        while self.cluster_idx < len(self.cluster_order):
            cid = self.cluster_order[self.cluster_idx]
            active = self._active_paths(cid)

            if len(active) == 1 and self.cfg.auto_assign_singletons:
                deer_id = self.deer_map.get(cid, f"CLUSTER_{cid:03d}")
                path = active[0]
                target, score, status = self._assign_removed_image(path, cid)
                if status == "assigned_existing_side_match":
                    self.members[cid] = []
                    self._register_cluster_if_needed(target)
                    self.members.setdefault(target, []).append(path)
                    self.current_assign[path] = target
                    self._append_outlier_move(path, cid, deer_id, target, score, status)
                    assign_rows = [
                        {
                            "image_path": path,
                            "assigned_cluster_id": str(target),
                            "assigned_score": "" if score is None else f"{score:.6f}",
                            "status": status,
                        }
                    ]
                    append_decision(self.cfg.decisions_csv, cid, deer_id, [path], [path], assign_rows, "singleton_auto_assign")
                    self.save_outputs()
                    continue

            if len(active) < max(1, self.cfg.min_review_cluster_size):
                self.cluster_idx += 1
                continue
            if self.batches_done.get(cid, 0) >= max(1, self.cfg.batches_per_cluster):
                self.cluster_idx += 1
                continue
            return cid

        return None

    def _advance(self) -> None:
        cid = self._next_cluster()
        if cid is None:
            self.finished = True
            self.current_cluster_id = None
            self.current_sample = []
            self.current_candidates = []
            return

        self.finished = False
        self._set_current_cluster(cid)

    def _set_current_cluster(self, cluster_id: int) -> None:
        active = self._active_paths(cluster_id)
        self.current_cluster_id = cluster_id
        self.current_sample = list(active)
        self.current_candidates = compute_candidates(
            cluster_id,
            self.members,
            self.current_assign,
            self.emb_by_path,
            self.deer_map,
            self.cfg.join_candidates,
        )

    def action_select_cluster(self, target_cluster_id: int) -> None:
        if target_cluster_id not in self.members:
            raise ValueError("unknown target cluster")
        self._register_cluster_if_needed(target_cluster_id)
        self.finished = False
        self.cluster_idx = self.cluster_order.index(target_cluster_id)
        self._set_current_cluster(target_cluster_id)
        self.last_action_message = f"Selected cluster {target_cluster_id}."

    def _finish_batch(self) -> None:
        if self.current_cluster_id is None:
            return
        self.batches_done[self.current_cluster_id] = self.batches_done.get(self.current_cluster_id, 0) + 1
        self.save_outputs()
        self._advance()

    def _image_side(self, image_path: str) -> str:
        return self.user_side_labels.get(image_path, self.predicted_side_by_path.get(image_path, "unknown"))

    def _assign_removed_image(self, image_path: str, from_cluster_id: int) -> tuple[int, float | None, str]:
        vec = self.emb_by_path.get(image_path)
        if vec is None:
            next_id = (max(self.members.keys()) + 1) if self.members else 0
            return next_id, None, "new_cluster_missing_embedding"

        source_side = self._image_side(image_path)
        if source_side not in {"left", "right"}:
            next_id = (max(self.members.keys()) + 1) if self.members else 0
            return next_id, None, "new_cluster_side_unknown"

        best_cluster = None
        best_score = -1.0
        for candidate_cluster_id, paths in self.members.items():
            if candidate_cluster_id < 0 or candidate_cluster_id == from_cluster_id:
                continue
            active_paths = [p for p in paths if self.current_assign.get(p) == candidate_cluster_id and p in self.emb_by_path]
            if not active_paths:
                continue
            same_side_paths = [p for p in active_paths if self._image_side(p) == source_side]
            if not same_side_paths:
                continue
            centroid_raw = torch.stack([self.emb_by_path[p] for p in same_side_paths], dim=0).mean(dim=0)
            centroid = cast(torch.Tensor, centroid_raw / centroid_raw.norm())
            score = float(cast(float, (vec @ centroid).item()))
            if score > best_score:
                best_score = score
                best_cluster = candidate_cluster_id

        if best_cluster is not None and best_score >= self.cfg.assign_threshold:
            return best_cluster, best_score, "assigned_existing_side_match"

        next_id = (max(self.members.keys()) + 1) if self.members else 0
        return next_id, (best_score if best_cluster is not None else None), "new_cluster"

    def _selected_paths(self, selected_indices: list[int], selected_paths: list[str]) -> list[str]:
        cid = self._current_id()
        active_set = set(self._active_paths(cid))
        picked: list[str] = []
        for path in selected_paths:
            if path in active_set and path not in picked:
                picked.append(path)
        if picked:
            return picked
        for idx in selected_indices:
            if 0 <= idx < len(self.current_sample):
                path = self.current_sample[idx]
                if path in active_set and path not in picked:
                    picked.append(path)
        return picked

    def _deer_id(self) -> str:
        if self.current_cluster_id is None:
            return ""
        return self.deer_map.get(self.current_cluster_id, f"CLUSTER_{self.current_cluster_id:03d}")

    def _current_id(self) -> int:
        if self.current_cluster_id is None:
            raise ValueError("no active cluster")
        return self.current_cluster_id

    def save_outputs(self) -> None:
        corrected = [ClusterRow(image_path=p, cluster_id=c) for p, c in sorted(self.current_assign.items())]
        write_clusters(self.cfg.out_clusters_csv, corrected)
        write_deer_map(self.cfg.out_deer_map_csv, self.deer_map, self.members)

    def action_keep_all(self) -> None:
        cid = self._current_id()
        append_decision(self.cfg.decisions_csv, cid, self._deer_id(), self.current_sample, [], [], "keep_all")
        self.last_action_message = f"Kept all {len(self.current_sample)} shown images in cluster {cid}."
        self._finish_batch()

    def action_rename(self, new_name: str) -> None:
        cid = self._current_id()
        name = new_name.strip()
        if not name:
            return
        self.deer_map[cid] = name
        append_decision(self.cfg.decisions_csv, cid, self.deer_map[cid], self.current_sample, [], [], "rename_cluster")
        self.last_action_message = f"Renamed cluster {cid} to {name}."
        self.save_outputs()

    def action_merge(self, target_cluster_id: int) -> None:
        cid = self._current_id()
        deer_id = self._deer_id()
        moved = [p for p in self.members.get(cid, []) if self.current_assign.get(p) == cid]
        for path in moved:
            self.current_assign[path] = target_cluster_id
            self._register_cluster_if_needed(target_cluster_id)
            self.members.setdefault(target_cluster_id, []).append(path)
        self.members[cid] = []
        assign_rows = [{"image_path": p, "assigned_cluster_id": str(target_cluster_id), "assigned_score": "", "status": "merge_cluster_candidate"} for p in moved]
        append_decision(self.cfg.decisions_csv, cid, deer_id, self.current_sample, moved, assign_rows, "merge_cluster")
        self.last_action_message = f"Merged cluster {cid} into cluster {target_cluster_id} ({len(moved)} images)."
        self._finish_batch()

    def action_merge_new(self) -> None:
        cid = self._current_id()
        deer_id = self._deer_id()
        new_id = (max(self.members.keys()) + 1) if self.members else 0
        self._register_cluster_if_needed(new_id)
        self.members[new_id] = []
        self.deer_map[new_id] = f"DEER_{new_id + 1:03d}"
        moved = [p for p in self.members.get(cid, []) if self.current_assign.get(p) == cid]
        for path in moved:
            self.current_assign[path] = new_id
            self.members[new_id].append(path)
        self.members[cid] = []
        assign_rows = [{"image_path": p, "assigned_cluster_id": str(new_id), "assigned_score": "", "status": "merge_cluster_new"} for p in moved]
        append_decision(self.cfg.decisions_csv, cid, deer_id, self.current_sample, moved, assign_rows, "merge_cluster_new")
        self.last_action_message = f"Moved {len(moved)} images from cluster {cid} to new cluster {new_id}."
        self._finish_batch()

    def action_remove_selected(self, selected_indices: list[int], selected_paths: list[str]) -> None:
        cid = self._current_id()
        deer_id = self._deer_id()
        removed = self._selected_paths(selected_indices, selected_paths)
        if not removed:
            return
        assign_rows: list[dict[str, str]] = []
        for path in removed:
            self.members[cid] = [x for x in self.members.get(cid, []) if x != path]
            target, score, status = self._assign_removed_image(path, cid)
            self._register_cluster_if_needed(target)
            self.members.setdefault(target, []).append(path)
            self.current_assign[path] = target
            assign_rows.append(
                {
                    "image_path": path,
                    "assigned_cluster_id": str(target),
                    "assigned_score": "" if score is None else f"{score:.6f}",
                    "status": status,
                }
            )
            self._append_outlier_move(path, cid, deer_id, target, score, status)
        append_decision(self.cfg.decisions_csv, cid, deer_id, self.current_sample, removed, assign_rows, "remove")
        targets = ", ".join(
            sorted({f"{row['assigned_cluster_id']} ({row['status']})" for row in assign_rows})
        )
        self.last_action_message = f"Removed {len(removed)} image(s) from cluster {cid} -> {targets}."
        self.save_outputs()
        self._set_current_cluster(cid)

    def action_all_different(self, selected_indices: list[int], selected_paths: list[str]) -> None:
        cid = self._current_id()
        deer_id = self._deer_id()
        removed = self._selected_paths(selected_indices, selected_paths)
        if not removed:
            removed = list(self.current_sample)
        assign_rows: list[dict[str, str]] = []
        used_targets: set[int] = set()
        for path in removed:
            self.members[cid] = [x for x in self.members.get(cid, []) if x != path]
            target, score, status = self._assign_removed_image(path, cid)
            if target in used_targets:
                target = (max(self.members.keys()) + 1) if self.members else 0
                self._register_cluster_if_needed(target)
                self.members[target] = []
                self.deer_map[target] = f"DEER_{target + 1:03d}"
                status = "new_cluster_all_different"
                score = None
            used_targets.add(target)
            self._register_cluster_if_needed(target)
            self.members.setdefault(target, []).append(path)
            self.current_assign[path] = target
            assign_rows.append(
                {
                    "image_path": path,
                    "assigned_cluster_id": str(target),
                    "assigned_score": "" if score is None else f"{score:.6f}",
                    "status": status,
                }
            )
            self._append_outlier_move(path, cid, deer_id, target, score, status)
        append_decision(self.cfg.decisions_csv, cid, deer_id, self.current_sample, removed, assign_rows, "all_different")
        self.last_action_message = f"Processed all-different for {len(removed)} image(s) in cluster {cid}."
        self._finish_batch()

    def action_manual_assign(self, selected_indices: list[int], selected_paths: list[str], target_cluster_id: int) -> None:
        cid = self._current_id()
        deer_id = self._deer_id()
        removed = self._selected_paths(selected_indices, selected_paths)
        if not removed:
            return
        assign_rows: list[dict[str, str]] = []
        for path in removed:
            self.members[cid] = [x for x in self.members.get(cid, []) if x != path]
            self._register_cluster_if_needed(target_cluster_id)
            self.members.setdefault(target_cluster_id, []).append(path)
            self.current_assign[path] = target_cluster_id
            assign_rows.append({"image_path": path, "assigned_cluster_id": str(target_cluster_id), "assigned_score": "", "status": "manual_assign"})
            self._append_outlier_move(path, cid, deer_id, target_cluster_id, None, "manual_assign")
        append_decision(self.cfg.decisions_csv, cid, deer_id, self.current_sample, removed, assign_rows, "manual_assign")
        self.last_action_message = f"Manually assigned {len(removed)} image(s) from cluster {cid} to {target_cluster_id}."
        self._finish_batch()

    def action_manual_assign_new(self, selected_indices: list[int], selected_paths: list[str]) -> None:
        cid = self._current_id()
        deer_id = self._deer_id()
        removed = self._selected_paths(selected_indices, selected_paths)
        if not removed:
            return
        assign_rows: list[dict[str, str]] = []
        for path in removed:
            self.members[cid] = [x for x in self.members.get(cid, []) if x != path]
            new_id = (max(self.members.keys()) + 1) if self.members else 0
            self._register_cluster_if_needed(new_id)
            self.members[new_id] = [path]
            self.deer_map[new_id] = f"DEER_{new_id + 1:03d}"
            self.current_assign[path] = new_id
            assign_rows.append({"image_path": path, "assigned_cluster_id": str(new_id), "assigned_score": "", "status": "manual_assign_new"})
            self._append_outlier_move(path, cid, deer_id, new_id, None, "manual_assign_new")
        append_decision(self.cfg.decisions_csv, cid, deer_id, self.current_sample, removed, assign_rows, "manual_assign")
        self.last_action_message = f"Assigned {len(removed)} image(s) from cluster {cid} to new clusters."
        self._finish_batch()

    def action_skip_batch(self) -> None:
        cid = self._current_id()
        self.last_action_message = f"Skipped current batch for cluster {cid}."
        self._finish_batch()

    def action_set_side_label(self, image_path: str, side_label: str) -> None:
        side = side_label.strip().lower()
        if side not in _ALLOWED_SIDES:
            raise ValueError("invalid side label")
        if image_path not in self.current_assign:
            raise ValueError("unknown image path")
        self.user_side_labels[image_path] = side
        write_user_side_labels(self.cfg.side_labels_csv, self.user_side_labels)
        self.last_action_message = f"Set side label for selected image to {side}."

    def action_remove_from_pool(self, selected_indices: list[int], selected_paths: list[str]) -> None:
        cid = self._current_id()
        removed = self._selected_paths(selected_indices, selected_paths)
        if not removed:
            return
        for path in removed:
            self.removed_images.add(path)
            _ = self.current_assign.pop(path, None)
            self.members[cid] = [p for p in self.members.get(cid, []) if p != path]
        append_removed_images(self.cfg.removed_images_csv, removed)
        self.last_action_message = f"Removed {len(removed)} image(s) from pool."
        self.save_outputs()
        self._set_current_cluster(cid)

    def _sample_entry(self, path: str) -> dict[str, str]:
        return {
            "path": path,
            "captured_at": image_capture_time(path),
            "side_label": self.user_side_labels.get(path, self.predicted_side_by_path.get(path, "unknown")),
            "side_label_source": (
                "user"
                if path in self.user_side_labels
                else ("predicted" if path in self.predicted_side_by_path else "unknown")
            ),
        }

    def state(self) -> dict[str, object]:
        total_clusters = len(self.cluster_order)
        reviewed = sum(1 for cid in self.cluster_order if self.batches_done.get(cid, 0) >= max(1, self.cfg.batches_per_cluster))
        waiting = waiting_reassign_count(self.cfg.outlier_queue_csv)
        cluster_id = self.current_cluster_id
        deer_id = self._deer_id() if cluster_id is not None else ""
        active_paths = self._active_paths(cluster_id) if cluster_id is not None else []
        sequence_suggestions: list[dict[str, object]] = []
        if cluster_id is not None and len(active_paths) == 1:
            only_path = active_paths[0]
            key = parse_sequence_key(only_path)
            if key is not None:
                prefix, number = key
                candidates = self.sequence_index.get(prefix, [])
                ranked: list[tuple[int, int, str]] = []
                for other_num, other_path in candidates:
                    if other_path == only_path:
                        continue
                    other_cluster = self.current_assign.get(other_path)
                    if other_cluster is None or other_cluster == cluster_id:
                        continue
                    if len(self._active_paths(other_cluster)) < 2:
                        continue
                    ranked.append((abs(other_num - number), other_cluster, other_path))
                ranked.sort(key=lambda row: row[0])
                seen: set[int] = set()
                for diff, other_cluster, other_path in ranked:
                    if other_cluster in seen:
                        continue
                    seen.add(other_cluster)
                    sequence_suggestions.append(
                        {
                            "cluster_id": other_cluster,
                            "deer_id": self.deer_map.get(other_cluster, f"CLUSTER_{other_cluster:03d}"),
                            "example_path": other_path,
                            "number_diff": diff,
                        }
                    )
                    if len(sequence_suggestions) >= 5:
                        break
        lines = cluster_map_lines(self.members, self.current_assign, self.deer_map)
        cluster_rows = [
            {
                "cluster_id": cid,
                "deer_id": self.deer_map.get(cid, f"CLUSTER_{cid:03d}"),
                "active_size": len(self._active_paths(cid)),
                "batch_done": self.batches_done.get(cid, 0),
            }
            for cid in sorted(self.cluster_order)
        ]
        return {
            "finished": self.finished,
            "last_action": self.last_action_message,
            "recent_moves": self.recent_moves,
            "summary": {
                "clusters_total": total_clusters,
                "clusters_done": reviewed,
                "waiting_reassign": waiting,
            },
            "clusters": cluster_rows,
            "current": {
                "cluster_id": cluster_id,
                "deer_id": deer_id,
                "sample": self.current_sample,
                "sample_entries": [self._sample_entry(path) for path in self.current_sample],
                "all_entries": [self._sample_entry(path) for path in active_paths],
                "candidates": self.current_candidates,
                "cluster_size": (len(self._active_paths(cluster_id)) if cluster_id is not None else 0),
                "batch_done": (self.batches_done.get(cluster_id, 0) if cluster_id is not None else 0),
                "sequence_suggestions": sequence_suggestions,
            },
            "cluster_map_lines": lines[:48],
        }


HTML = """<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Cluster Review Web UI</title>
  <style>
    :root { --bg:#f5f1e7; --ink:#222; --muted:#655; --panel:#fff; --accent:#24543d; --warn:#8b3d23; --border:#e2dac8; }
    * { box-sizing: border-box; }
    body { margin:0; font-family: \"Avenir Next\", Avenir, \"Segoe UI\", sans-serif; color:var(--ink); background: radial-gradient(circle at top right,#eadfcb,var(--bg)); }
    header { padding:12px 16px; border-bottom:1px solid var(--border); background:#fbf7ee; display:flex; gap:8px; flex-wrap:wrap; }
    .pill { background:#efe7d8; border:1px solid var(--border); border-radius:999px; padding:4px 10px; font-size:12px; }
    .layout { display:grid; grid-template-columns: 1fr 360px; gap:12px; padding:12px; }
    .panel { background:var(--panel); border:1px solid var(--border); border-radius:12px; padding:10px; }
    .grid { display:grid; grid-template-columns: repeat(2, minmax(220px, 1fr)); gap:10px; }
    .card { border:1px solid var(--border); border-radius:10px; padding:8px; background:#fcfaf4; }
    .card .when { margin-top:6px; font-size:12px; color:var(--muted); font-variant-numeric: tabular-nums; }
    .card .side-controls { margin-top:6px; display:flex; align-items:center; gap:6px; flex-wrap:wrap; }
    .card .side-title { font-size:12px; color:var(--muted); }
    .card .side-source { font-size:11px; color:var(--muted); }
    .card .side-btn { padding:4px 8px; font-size:11px; border-radius:8px; }
    .card .side-btn.active { background:#24543d; color:#fff; border-color:#24543d; }
    .card .meta { font-size:12px; color:var(--muted); margin-top:4px; word-break:break-all; }
    img { width:100%; height:260px; object-fit:cover; border-radius:8px; background:#efe5d3; }
    .actions { display:flex; gap:8px; flex-wrap:wrap; margin:8px 0; }
    button { border:1px solid var(--border); border-radius:10px; padding:8px 10px; cursor:pointer; background:#fff; }
    button.primary { background:var(--accent); color:#fff; border-color:var(--accent); }
    button.warn { background:var(--warn); color:#fff; border-color:var(--warn); }
    .hint-inline { font-size:11px; color:var(--muted); align-self:center; margin-right:10px; }
    input, select { border:1px solid var(--border); border-radius:8px; padding:6px 8px; }
    .map { max-height: calc(100vh - 210px); overflow:auto; font-size:12px; color:var(--muted); white-space:pre-wrap; }
    .section-title { margin:10px 0 6px; font-size:13px; color:var(--muted); }
    .moves-table { width:100%; border-collapse:collapse; font-size:11px; margin-top:8px; }
    .moves-table th, .moves-table td { border:1px solid var(--border); padding:4px 6px; text-align:left; vertical-align:top; }
    .moves-table th { background:#f6efe0; }
    .moves-path { max-width:240px; word-break:break-all; }
    .suggestions { font-size:12px; color:var(--muted); }
    .suggestions .row { display:flex; gap:8px; align-items:center; margin-top:6px; }
    .suggestions .pill { background:#efe7d8; border:1px solid var(--border); border-radius:999px; padding:2px 8px; font-size:11px; }
    .suggestions .path { word-break:break-all; }
    @media (max-width: 1100px) { .layout { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <header>
    <span class=\"pill\" id=\"s1\">clusters 0/0</span>
    <span class=\"pill\" id=\"s2\">cluster -</span>
    <span class=\"pill\" id=\"s3\">waiting 0</span>
    <span class=\"pill\" id=\"s4\">batch 0</span>
    <span class=\"pill\" id=\"s5\">sample 0</span>
  </header>
  <div class=\"layout\">
    <div class=\"panel\">
      <div class=\"actions\">
        <select id=\"clusterPicker\"></select>
        <button onclick=\"jumpCluster()\">Go To Cluster</button>
        <span class=\"hint-inline\">pick any cluster and review it directly</span>
      </div>
      <div class=\"actions\">
        <button class=\"primary\" onclick=\"doAction('keep_all')\">Keep All</button>
        <span class=\"hint-inline\">accept this whole sample</span>
        <button class=\"warn\" onclick=\"doAction('remove_selected')\">Remove Selected</button>
        <span class=\"hint-inline\">reassign checked images</span>
        <button class=\"warn\" onclick=\"doAction('all_different')\">All Different</button>
        <span class=\"hint-inline\">split checked to separate IDs</span>
        <button onclick=\"doAction('skip_batch')\">Skip Batch</button>
        <span class=\"hint-inline\">next sample, no changes</span>
        <button class=\"warn\" onclick=\"removeFromPool()\">Remove From Pool</button>
        <span class=\"hint-inline\">exclude checked images entirely</span>
      </div>
      <div class=\"actions\">
        <input id=\"rename\" placeholder=\"rename deer id\" />
        <button onclick=\"renameCluster()\">Rename</button>
        <span class=\"hint-inline\">rename current cluster</span>
      </div>
      <div class=\"actions\">
        <select id=\"candidate\"></select>
        <button onclick=\"mergeCandidate()\">Merge Cluster</button>
        <span class=\"hint-inline\">move all to selected candidate</span>
        <button onclick=\"doAction('merge_new')\">Merge to New</button>
        <span class=\"hint-inline\">move all to a new deer ID</span>
      </div>
      <div class=\"actions\">
        <select id=\"manualTarget\"></select>
        <button onclick=\"manualAssign()\">Assign Selected</button>
        <span class=\"hint-inline\">assign checked to selected cluster</span>
        <button onclick=\"doAction('manual_assign_new')\">Assign Selected New</button>
        <span class=\"hint-inline\">each checked gets a new ID</span>
      </div>
      <div class=\"grid\" id=\"grid\"></div>
    </div>
    <div class=\"panel\">
      <div class=\"map\" id=\"map\"></div>
      <div id=\"status\" style=\"margin-top:8px; font-size:12px; color:var(--muted);\"></div>
      <div class=\"section-title\">recent moves (last 10)</div>
      <table class=\"moves-table\">
        <thead>
          <tr><th>from</th><th>to</th><th>status</th><th>score</th><th>image</th></tr>
        </thead>
        <tbody id=\"movesBody\"></tbody>
      </table>
      <div class=\"section-title\">singleton filename suggestions</div>
      <div class=\"suggestions\" id=\"seqSuggestions\">Only shown when the selected cluster has a single image.</div>
    </div>
  </div>
  <div style=\"padding:0 12px 12px; font-size:12px; color:var(--muted); border-top:1px dashed var(--border);\" id=\"debug\">UI bootstrap: loading...</div>
<script>
var state = null;
var selected = {};

window.onerror = function(message, source, lineno, colno) {
  var dbg = document.getElementById('debug');
  if (dbg) {
    dbg.textContent = 'JS error: ' + String(message) + ' @' + String(lineno) + ':' + String(colno);
  }
};

function img(path) {
  return '/image?path=' + encodeURIComponent(path);
}

function selectedPaths() {
  var out = [];
  for (var key in selected) {
    if (Object.prototype.hasOwnProperty.call(selected, key)) {
      out.push(String(key));
    }
  }
  out.sort();
  return out;
}

function render() {
  if (!state) {
    return;
  }
  var s = state.summary || {};
  var c = state.current || {};
  var cid = (c.cluster_id !== null && c.cluster_id !== undefined) ? c.cluster_id : '-';
  var sampleEntries = c.sample_entries || [];
  if (!sampleEntries.length && (c.sample || []).length) {
    for (var legacyIdx = 0; legacyIdx < c.sample.length; legacyIdx += 1) {
      sampleEntries.push({ path: c.sample[legacyIdx], captured_at: '' });
    }
  }
  var sample = [];
  for (var sampleIdx = 0; sampleIdx < sampleEntries.length; sampleIdx += 1) {
    sample.push(sampleEntries[sampleIdx].path || '');
  }
  var candidates = c.candidates || [];
  var clusters = (state.clusters || []).slice(0);
  function clusterIdValue(row) {
    var raw = row ? row.cluster_id : null;
    var parsed = parseInt(String(raw), 10);
    return isNaN(parsed) ? Number.MAX_SAFE_INTEGER : parsed;
  }
  clusters.sort(function(a, b) {
    return clusterIdValue(a) - clusterIdValue(b);
  });

  document.getElementById('s1').textContent = 'clusters ' + String(s.clusters_done || 0) + '/' + String(s.clusters_total || 0);
  document.getElementById('s2').textContent = 'cluster ' + String(cid) + ' | ' + String(c.deer_id || '');
  document.getElementById('s3').textContent = 'waiting ' + String(s.waiting_reassign || 0);
  document.getElementById('s4').textContent = 'batch ' + String(c.batch_done || 0);
  document.getElementById('s5').textContent = 'sample ' + String(sample.length);

  var picker = document.getElementById('clusterPicker');
  var pickerValue = picker ? picker.value : '';
  if (picker) {
    picker.innerHTML = '';
    for (var clusterIdx = 0; clusterIdx < clusters.length; clusterIdx += 1) {
      var clusterRow = clusters[clusterIdx];
      var clusterOpt = document.createElement('option');
      clusterOpt.value = String(clusterRow.cluster_id);
      clusterOpt.textContent = String(clusterRow.deer_id) + ' [' + String(clusterRow.cluster_id) + '] size=' + String(clusterRow.active_size);
      if (String(clusterRow.cluster_id) === String(cid)) {
        clusterOpt.selected = true;
      }
      picker.appendChild(clusterOpt);
    }
    if (pickerValue && picker.value !== pickerValue) {
      picker.value = pickerValue;
    }
  }

  var manualTarget = document.getElementById('manualTarget');
  var manualValue = manualTarget ? manualTarget.value : '';
  if (manualTarget) {
    manualTarget.innerHTML = '';
    for (var manualIdx = 0; manualIdx < clusters.length; manualIdx += 1) {
      var manualRow = clusters[manualIdx];
      var manualOpt = document.createElement('option');
      manualOpt.value = String(manualRow.cluster_id);
      manualOpt.textContent = String(manualRow.deer_id) + ' [' + String(manualRow.cluster_id) + ']';
      manualTarget.appendChild(manualOpt);
    }
    if (manualValue && manualTarget.value !== manualValue) {
      manualTarget.value = manualValue;
    }
  }

  var grid = document.getElementById('grid');
  grid.innerHTML = '';
  for (var idx = 0; idx < sample.length; idx += 1) {
    var entry = sampleEntries[idx] || {};
    var path = entry.path || sample[idx] || '';
    var card = document.createElement('div');
    card.className = 'card';
    var box = document.createElement('input');
    box.type = 'checkbox';
    box.checked = !!selected[path];
    box.setAttribute('data-path', path);
    box.onchange = function() {
      var p = String(this.getAttribute('data-path') || '');
      if (this.checked) {
        selected[p] = true;
      } else {
        delete selected[p];
      }
    };
    var image = document.createElement('img');
    image.src = img(path);
    var sideControls = document.createElement('div');
    sideControls.className = 'side-controls';
    var sideTitle = document.createElement('div');
    sideTitle.className = 'side-title';
    sideTitle.textContent = 'side:';
    var sideLeft = document.createElement('button');
    sideLeft.className = 'side-btn' + ((entry.side_label || 'unknown') === 'left' ? ' active' : '');
    sideLeft.textContent = 'Left';
    sideLeft.onclick = (function(imagePath) {
      return function() { setSideLabel(imagePath, 'left'); };
    })(path);
    var sideRight = document.createElement('button');
    sideRight.className = 'side-btn' + ((entry.side_label || 'unknown') === 'right' ? ' active' : '');
    sideRight.textContent = 'Right';
    sideRight.onclick = (function(imagePath) {
      return function() { setSideLabel(imagePath, 'right'); };
    })(path);
    var sideUnknown = document.createElement('button');
    sideUnknown.className = 'side-btn' + ((entry.side_label || 'unknown') === 'unknown' ? ' active' : '');
    sideUnknown.textContent = 'Unknown';
    sideUnknown.onclick = (function(imagePath) {
      return function() { setSideLabel(imagePath, 'unknown'); };
    })(path);
    var sideSource = document.createElement('div');
    sideSource.className = 'side-source';
    sideSource.textContent = 'source: ' + String(entry.side_label_source || 'unknown');
    sideControls.appendChild(sideTitle);
    sideControls.appendChild(sideLeft);
    sideControls.appendChild(sideRight);
    sideControls.appendChild(sideUnknown);
    sideControls.appendChild(sideSource);
    var when = document.createElement('div');
    when.className = 'when';
    when.textContent = entry.captured_at ? ('date/time: ' + String(entry.captured_at)) : 'date/time: -';
    var meta = document.createElement('div');
    meta.className = 'meta';
    meta.textContent = path;
    card.appendChild(box);
    card.appendChild(image);
    card.appendChild(sideControls);
    card.appendChild(when);
    card.appendChild(meta);
    grid.appendChild(card);
  }

  var cand = document.getElementById('candidate');
  cand.innerHTML = '';
  for (var j = 0; j < candidates.length; j += 1) {
    var row = candidates[j];
    var opt = document.createElement('option');
    opt.value = String(row.cluster_id);
    opt.textContent = String(row.deer_id) + ' [' + String(row.cluster_id) + '] score=' + Number(row.score).toFixed(3);
    cand.appendChild(opt);
  }

  document.getElementById('map').textContent = (state.cluster_map_lines || []).join('\\n');
  var statusParts = [];
  if (state.last_action) {
    statusParts.push(String(state.last_action));
  }
  if (state.finished) {
    statusParts.push('Review finished. Outputs saved.');
  } else if (!sample.length) {
    statusParts.push('No sample in current cluster. Try Skip Batch or check input files.');
  }
  document.getElementById('status').textContent = statusParts.join(' | ');

  var movesBody = document.getElementById('movesBody');
  if (movesBody) {
    movesBody.innerHTML = '';
    var recentMoves = state.recent_moves || [];
    for (var m = recentMoves.length - 1; m >= 0; m -= 1) {
      var move = recentMoves[m] || {};
      var tr = document.createElement('tr');
      var tdFrom = document.createElement('td');
      tdFrom.textContent = String(move.from_cluster_id || '-');
      var tdTo = document.createElement('td');
      tdTo.textContent = String(move.assigned_cluster_id || '-');
      var tdStatus = document.createElement('td');
      tdStatus.textContent = String(move.status || '');
      var tdScore = document.createElement('td');
      tdScore.textContent = String(move.assigned_score || '');
      var tdImage = document.createElement('td');
      tdImage.className = 'moves-path';
      tdImage.textContent = String(move.image_path || '');
      tr.appendChild(tdFrom);
      tr.appendChild(tdTo);
      tr.appendChild(tdStatus);
      tr.appendChild(tdScore);
      tr.appendChild(tdImage);
      movesBody.appendChild(tr);
    }
  }
  var seqBox = document.getElementById('seqSuggestions');
  if (seqBox) {
    var suggestions = (c.sequence_suggestions || []);
    if (!suggestions.length) {
      seqBox.textContent = 'No sequence suggestions for this cluster.';
    } else {
      seqBox.innerHTML = '';
      for (var sIdx = 0; sIdx < suggestions.length; sIdx += 1) {
        var s = suggestions[sIdx] || {};
        var row = document.createElement('div');
        row.className = 'row';
        var pill = document.createElement('span');
        pill.className = 'pill';
        pill.textContent = 'cluster ' + String(s.cluster_id || '-');
        var text = document.createElement('span');
        text.textContent = String(s.deer_id || '') + ' diff=' + String(s.number_diff || '-');
        var path = document.createElement('span');
        path.className = 'path';
        path.textContent = String(s.example_path || '');
        row.appendChild(pill);
        row.appendChild(text);
        row.appendChild(path);
        seqBox.appendChild(row);
      }
    }
  }
  document.getElementById('debug').textContent = 'UI ready. cluster_images=' + String(sample.length) + ', candidates=' + String(candidates.length);
}

function refresh() {
  try {
    var req = new XMLHttpRequest();
    req.open('GET', '/api/state', true);
    req.onreadystatechange = function() {
      if (req.readyState !== 4) {
        return;
      }
      if (req.status < 200 || req.status >= 300) {
        document.getElementById('debug').textContent = 'API error: ' + String(req.status);
        return;
      }
      state = JSON.parse(req.responseText);
      if (!state.current || !state.current.sample) {
        selected = {};
      }
      render();
    };
    req.send(null);
  } catch (err) {
    document.getElementById('debug').textContent = 'Failed to load state: ' + String(err);
  }
}

function post(payload, keepSelection) {
  var req = new XMLHttpRequest();
  req.open('POST', '/api/action', true);
  req.setRequestHeader('Content-Type', 'application/json');
  req.onreadystatechange = function() {
    if (req.readyState !== 4) {
      return;
    }
    if (req.status < 200 || req.status >= 300) {
      document.getElementById('status').textContent = req.responseText || 'action failed';
      return;
    }
    if (!keepSelection) {
      selected = {};
    }
    refresh();
  };
  req.send(JSON.stringify(payload));
}

function doAction(action) {
  post({ action: action, selected_paths: selectedPaths() }, false);
}

function renameCluster() {
  var value = document.getElementById('rename').value || '';
  post({ action: 'rename_cluster', deer_id: value }, false);
}

function mergeCandidate() {
  var target = document.getElementById('candidate').value;
  post({ action: 'merge_cluster', target_cluster_id: Number(target) }, false);
}

function jumpCluster() {
  var target = document.getElementById('clusterPicker').value;
  post({ action: 'select_cluster', target_cluster_id: Number(target) }, false);
}

function manualAssign() {
  var target = Number(document.getElementById('manualTarget').value);
  post({ action: 'manual_assign', selected_paths: selectedPaths(), target_cluster_id: target }, false);
}

function removeFromPool() {
  if (!confirm('Remove selected images from the pool entirely?')) {
    return;
  }
  post({ action: 'remove_from_pool', selected_paths: selectedPaths() }, false);
}

function setSideLabel(path, sideLabel) {
  post({ action: 'set_side_label', image_path: path, side_label: sideLabel }, true);
}

refresh();
</script>
</body>
</html>
"""


def repo_relative(repo_root: Path, raw_path: str) -> Path | None:
    candidate = Path(raw_path)
    target = candidate if candidate.is_absolute() else (repo_root / candidate)
    resolved_target = target.resolve()
    resolved_root = repo_root.resolve()
    try:
        _ = resolved_target.relative_to(resolved_root)
    except ValueError:
        return None
    return resolved_target


def padded_variant(path: Path) -> Path | None:
    if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        return None
    candidate = path.with_name(f"{path.stem}{_PAD_SUFFIX}{path.suffix}")
    if candidate.exists():
        return candidate
    return None


def compute_candidates(
    cluster_id: int,
    members: dict[int, list[str]],
    current_assign: dict[str, int],
    emb_by_path: dict[str, torch.Tensor],
    deer_map: dict[int, str],
    limit: int,
) -> list[dict[str, object]]:
    current = [p for p in members.get(cluster_id, []) if current_assign.get(p) == cluster_id and p in emb_by_path]
    if not current:
        return []
    current_centroid = torch.stack([emb_by_path[p] for p in current], dim=0).mean(dim=0)
    current_centroid = F.normalize(current_centroid, dim=0)
    out: list[dict[str, object]] = []
    for other_cluster_id, other_paths in members.items():
        if other_cluster_id == cluster_id or other_cluster_id < 0:
            continue
        active_other = [p for p in other_paths if current_assign.get(p) == other_cluster_id and p in emb_by_path]
        if not active_other:
            continue
        other_centroid = torch.stack([emb_by_path[p] for p in active_other], dim=0).mean(dim=0)
        other_centroid = F.normalize(other_centroid, dim=0)
        score = float(cast(float, (current_centroid @ other_centroid).item()))
        out.append(
            {
                "cluster_id": other_cluster_id,
                "deer_id": deer_map.get(other_cluster_id, f"CLUSTER_{other_cluster_id:03d}"),
                "score": score,
            }
        )
    out.sort(key=lambda row: float(cast(float, row["score"])), reverse=True)
    return out[: max(1, limit)]


def create_handler(engine: ClusterReviewEngine, repo_root: Path):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, content_type: str = "text/plain") -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            _ = self.wfile.write(body)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send(200, HTML.encode("utf-8"), "text/html")
                return
            if parsed.path == "/api/state":
                self._send(200, json.dumps(engine.state()).encode("utf-8"), "application/json")
                return
            if parsed.path == "/image":
                qs = parse_qs(parsed.query)
                raw_path = (qs.get("path", [""])[0] or "").strip()
                safe = repo_relative(repo_root, raw_path)
                if safe is None or not safe.exists() or safe.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    self._send(404, b"not found")
                    return
                padded = padded_variant(safe)
                target = padded if padded is not None else safe
                content_type = "image/png" if target.suffix.lower() == ".png" else "image/jpeg"
                self._send(200, target.read_bytes(), content_type)
                return
            self._send(404, b"not found")

        def do_POST(self) -> None:
            if self.path != "/api/action":
                self._send(404, b"not found")
                return

            length = int(self.headers.get("Content-Length", "0"))
            raw = cast(object, json.loads(self.rfile.read(length) or b"{}"))
            if not isinstance(raw, dict):
                self._send(400, b"invalid payload")
                return
            payload = cast(dict[str, object], raw)
            action = str(payload.get("action", "")).strip()
            selected_raw = payload.get("selected_indices", [])
            selected: list[int] = []
            if isinstance(selected_raw, list):
                for value in cast(list[object], selected_raw):
                    if isinstance(value, int):
                        selected.append(value)
            selected_paths_raw = payload.get("selected_paths", [])
            selected_paths: list[str] = []
            if isinstance(selected_paths_raw, list):
                for value in cast(list[object], selected_paths_raw):
                    if isinstance(value, str):
                        path = value.strip()
                        if path:
                            selected_paths.append(path)

            def parse_int_field(name: str) -> int | None:
                value = payload.get(name)
                if isinstance(value, int):
                    return value
                if isinstance(value, str) and value.strip().isdigit():
                    return int(value.strip())
                return None

            try:
                if action == "keep_all":
                    engine.action_keep_all()
                elif action == "remove_selected":
                    engine.action_remove_selected(selected, selected_paths)
                elif action == "all_different":
                    engine.action_all_different(selected, selected_paths)
                elif action == "rename_cluster":
                    deer_id = str(payload.get("deer_id", ""))
                    engine.action_rename(deer_id)
                elif action == "merge_cluster":
                    target = parse_int_field("target_cluster_id")
                    if target is None or target < 0:
                        raise ValueError("invalid target cluster")
                    engine.action_merge(target)
                elif action == "merge_new":
                    engine.action_merge_new()
                elif action == "manual_assign":
                    target = parse_int_field("target_cluster_id")
                    if target is None or target < 0:
                        raise ValueError("invalid target cluster")
                    engine.action_manual_assign(selected, selected_paths, target)
                elif action == "manual_assign_new":
                    engine.action_manual_assign_new(selected, selected_paths)
                elif action == "skip_batch":
                    engine.action_skip_batch()
                elif action == "remove_from_pool":
                    engine.action_remove_from_pool(selected, selected_paths)
                elif action == "select_cluster":
                    target = parse_int_field("target_cluster_id")
                    if target is None:
                        raise ValueError("invalid target cluster")
                    engine.action_select_cluster(target)
                elif action == "set_side_label":
                    image_path = str(payload.get("image_path", "")).strip()
                    side_label = str(payload.get("side_label", "")).strip().lower()
                    if not image_path:
                        raise ValueError("missing image_path")
                    engine.action_set_side_label(image_path, side_label)
                else:
                    self._send(400, b"unknown action")
                    return
            except ValueError as exc:
                self._send(400, str(exc).encode("utf-8"))
                return

            self._send(200, b"ok")

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--clusters-csv", default="data/reid/unified_round5/clusters_recognizable_plus_manual_round2_k40_corrected_web.csv")
    _ = parser.add_argument("--deer-map-csv", default="data/reid/unified_round5/deer_id_proposal_recognizable_plus_manual_round2_k40_corrected_web.csv")
    _ = parser.add_argument("--sample-size", type=int, default=4)
    _ = parser.add_argument("--seed", type=int, default=42)
    _ = parser.add_argument("--out-clusters-csv", default="data/reid/unified_round5/clusters_recognizable_plus_manual_round2_k40_corrected_web.csv")
    _ = parser.add_argument("--outlier-queue-csv", default="data/reid/unified_round5/cluster_outlier_queue_k40_web.csv")
    _ = parser.add_argument("--decisions-csv", default="data/reid/unified_round5/cluster_outlier_review_decisions_k40_web.csv")
    _ = parser.add_argument("--out-deer-map-csv", default="data/reid/unified_round5/deer_id_proposal_recognizable_plus_manual_round2_k40_corrected_web.csv")
    _ = parser.add_argument("--side-labels-csv", default="data/reid/unified_round5/side_labels_k40_web.csv")
    _ = parser.add_argument("--removed-images-csv", default="data/reid/unified_round5/removed_images_k40_web.csv")
    _ = parser.add_argument("--embeddings", default="data/reid/unified_round5/embeddings_recognizable_plus_manual_round2.pt")
    _ = parser.add_argument("--metadata", default="data/reid/unified_round5/embeddings_recognizable_plus_manual_round2.csv")
    _ = parser.add_argument("--assign-threshold", type=float, default=0.98)
    _ = parser.add_argument("--join-candidates", type=int, default=9)
    _ = parser.add_argument("--batches-per-cluster", type=int, default=2)
    _ = parser.add_argument("--min-review-cluster-size", type=int, default=2)
    _ = parser.add_argument("--auto-assign-singletons", action="store_true", default=True)
    _ = parser.add_argument("--no-auto-assign-singletons", dest="auto_assign_singletons", action="store_false")
    _ = parser.add_argument("--host", default="127.0.0.1")
    _ = parser.add_argument("--port", type=int, default=8011)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    values = cast(dict[str, object], vars(args))

    def get_str(name: str) -> str:
        value = values.get(name)
        if not isinstance(value, str):
            raise ValueError(f"invalid --{name.replace('_', '-')}")
        return value

    def get_int(name: str) -> int:
        value = values.get(name)
        if not isinstance(value, int):
            raise ValueError(f"invalid --{name.replace('_', '-')}")
        return value

    def get_float(name: str) -> float:
        value = values.get(name)
        if isinstance(value, (int, float)):
            return float(value)
        raise ValueError(f"invalid --{name.replace('_', '-')}")

    def get_bool(name: str) -> bool:
        value = values.get(name)
        if not isinstance(value, bool):
            raise ValueError(f"invalid --{name.replace('_', '-')}")
        return value

    clusters_csv = get_str("clusters_csv")
    deer_map_csv = get_str("deer_map_csv")
    out_clusters_csv = get_str("out_clusters_csv")
    outlier_queue_csv = get_str("outlier_queue_csv")
    decisions_csv = get_str("decisions_csv")
    out_deer_map_csv = get_str("out_deer_map_csv")
    side_labels_csv = get_str("side_labels_csv")
    removed_images_csv = get_str("removed_images_csv")
    embeddings = get_str("embeddings")
    metadata = get_str("metadata")
    sample_size = get_int("sample_size")
    seed = get_int("seed")
    assign_threshold = get_float("assign_threshold")
    join_candidates = get_int("join_candidates")
    batches_per_cluster = get_int("batches_per_cluster")
    min_review_cluster_size = get_int("min_review_cluster_size")
    auto_assign_singletons = get_bool("auto_assign_singletons")
    host = get_str("host")
    port = get_int("port")

    cfg = ReviewConfig(
        clusters_csv=Path(clusters_csv),
        deer_map_csv=Path(deer_map_csv),
        out_clusters_csv=Path(out_clusters_csv),
        outlier_queue_csv=Path(outlier_queue_csv),
        decisions_csv=Path(decisions_csv),
        out_deer_map_csv=Path(out_deer_map_csv),
        side_labels_csv=Path(side_labels_csv),
        removed_images_csv=Path(removed_images_csv),
        embeddings=Path(embeddings),
        metadata=Path(metadata),
        sample_size=sample_size,
        seed=seed,
        assign_threshold=assign_threshold,
        join_candidates=join_candidates,
        batches_per_cluster=batches_per_cluster,
        min_review_cluster_size=min_review_cluster_size,
        auto_assign_singletons=auto_assign_singletons,
    )

    if not cfg.clusters_csv.exists():
        raise FileNotFoundError(f"clusters csv not found: {cfg.clusters_csv}")
    if not cfg.embeddings.exists() or not cfg.metadata.exists():
        raise FileNotFoundError("embeddings/metadata not found")

    engine = ClusterReviewEngine(cfg)
    repo_root = Path(__file__).resolve().parents[2]
    server = HTTPServer((host, port), create_handler(engine, repo_root))
    print(f"Cluster review UI running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
