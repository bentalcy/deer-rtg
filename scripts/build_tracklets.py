import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import torch
from tqdm import tqdm
from ultralytics import YOLO


DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def iou(box_a: list[int], box_b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0

    a_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    b_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = a_area + b_area - inter_area
    return inter_area / union if union > 0 else 0.0


@dataclass
class Track:
    track_id: int
    bbox: list[int]
    missed: int = 0


def detect_deer(model: YOLO, img_bgr, conf: float, iou_nms: float | None) -> list[dict]:
    predict_args: dict = {
        "device": DEVICE,
        "conf": conf,
        "verbose": False,
    }
    if iou_nms is not None:
        predict_args["iou"] = iou_nms

    results = model.predict(img_bgr, **predict_args)
    if results is None:
        return []

    results_list = results if isinstance(results, list) else [results]
    dets = []
    for r in results_list:
        if r.boxes is None:
            continue
        names = getattr(r, "names", {})
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            dets.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "conf": float(box.conf[0]),
                    "cls": int(box.cls[0]),
                    "cls_name": names.get(int(box.cls[0]), str(int(box.cls[0]))),
                }
            )
    return dets


def greedy_match_tracks(
    tracks: list[Track],
    detections: list[dict],
    iou_thresh: float,
    max_missed: int,
    next_track_id: int,
) -> tuple[list[Track], list[dict], int]:
    unmatched_tracks = tracks[:]
    unmatched_dets = detections[:]
    assigned: list[tuple[Track, dict]] = []

    # Greedy matching by IoU
    while True:
        best = (0.0, None, None)
        for t in unmatched_tracks:
            for d in unmatched_dets:
                score = iou(t.bbox, d["bbox"])
                if score > best[0]:
                    best = (score, t, d)
        best_iou, best_track, best_det = best
        if best_track is None or best_det is None or best_iou < iou_thresh:
            break

        unmatched_tracks.remove(best_track)
        unmatched_dets.remove(best_det)
        assigned.append((best_track, best_det))

    updated_tracks: list[Track] = []
    for t, d in assigned:
        updated_tracks.append(Track(track_id=t.track_id, bbox=d["bbox"], missed=0))
        d["track_id"] = t.track_id

    for t in unmatched_tracks:
        t.missed += 1
        if t.missed <= max_missed:
            updated_tracks.append(t)

    for d in unmatched_dets:
        d["track_id"] = next_track_id
        updated_tracks.append(Track(track_id=next_track_id, bbox=d["bbox"], missed=0))
        next_track_id += 1

    return updated_tracks, detections, next_track_id


def crop_and_write(img_bgr, bbox: list[int], out_path: Path) -> bool:
    x1, y1, x2, y2 = bbox
    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(out_path), crop))


def build_tracklets(
    frames_root: str,
    out_root: str,
    model_path: str,
    conf: float = 0.4,
    iou_nms: float | None = 0.3,
    match_iou: float = 0.3,
    max_missed: int = 2,
    actions_path: str | None = None,
) -> None:
    frames_dir = Path(frames_root)
    out_dir = Path(out_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_path).to(DEVICE)
    frame_files = sorted(frames_dir.rglob("*.jpg"))

    tracks: list[Track] = []
    next_track_id = 0
    actions: list[dict] = []

    for frame_path in tqdm(frame_files, desc="Tracking"):
        img = cv2.imread(str(frame_path))
        if img is None:
            continue

        dets = detect_deer(model, img, conf=conf, iou_nms=iou_nms)
        tracks, dets, next_track_id = greedy_match_tracks(
            tracks,
            dets,
            iou_thresh=match_iou,
            max_missed=max_missed,
            next_track_id=next_track_id,
        )

        frame_action = {
            "frame_path": str(frame_path),
            "detections": [],
        }
        for det_i, det in enumerate(dets):
            track_id = det["track_id"]
            out_path = out_dir / f"track_{track_id:03d}" / f"{frame_path.stem}__t{track_id:03d}__d{det_i:02d}.jpg"
            ok = crop_and_write(img, det["bbox"], out_path)
            if not ok:
                continue
            frame_action["detections"].append(
                {
                    "track_id": track_id,
                    "bbox": det["bbox"],
                    "conf": det["conf"],
                    "cls": det["cls"],
                    "cls_name": det["cls_name"],
                    "new_name": str(out_path),
                }
            )
        actions.append(frame_action)

    if actions_path is not None:
        actions_file = Path(actions_path)
        actions_file.parent.mkdir(parents=True, exist_ok=True)
        with open(actions_file, "w") as f:
            json.dump(actions, f, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--iou-nms", type=float, default=0.3)
    parser.add_argument("--match-iou", type=float, default=0.3)
    parser.add_argument("--max-missed", type=int, default=2)
    parser.add_argument("--actions-path", default=None)
    args = parser.parse_args()

    build_tracklets(
        frames_root=args.frames_root,
        out_root=args.out_root,
        model_path=args.model,
        conf=args.conf,
        iou_nms=args.iou_nms,
        match_iou=args.match_iou,
        max_missed=args.max_missed,
        actions_path=args.actions_path,
    )
