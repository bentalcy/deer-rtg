import cv2
import torch
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
from typing import Optional
from PIL import Image

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


class DeerDetector:
    def __init__(self, model_name="atonus/peura", conf=0.3, iou=None):
        self.device = DEVICE
        self.model = YOLO(model_name).to(self.device)
        self.conf = conf
        self.iou = iou

    def _detect(self, img_bgr):
        predict_args = {
            "device": self.device,
            "conf": self.conf,
            "verbose": False,
        }
        if self.iou is not None:
            predict_args["iou"] = self.iou

        results = self.model.predict(img_bgr, **predict_args)
        if results is None:
            return []
        detections = []
        if isinstance(results, list):
            results_list = results
        else:
            results_list = [results]

        for r in results_list:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                cls_name = r.names.get(cls, str(cls))
                detections.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "conf": conf,
                        "cls": cls,
                        "cls_name": cls_name,
                    }
                )

        return detections

    @staticmethod
    def _iou(box_a, box_b):
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

    def _nms(self, detections, iou_thresh=0.5):
        if not detections:
            return []
        dets = sorted(detections, key=lambda d: d["conf"], reverse=True)
        keep = []
        while dets:
            best = dets.pop(0)
            keep.append(best)
            remaining = []
            for det in dets:
                if self._iou(best["bbox"], det["bbox"]) <= iou_thresh:
                    remaining.append(det)
            dets = remaining
        return keep

    def _detect_tiled(self, img_bgr, tile_size=640, overlap=0.2, nms_iou=0.5):
        h, w = img_bgr.shape[:2]
        stride = max(1, int(tile_size * (1 - overlap)))
        detections = []
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                x2 = min(x + tile_size, w)
                y2 = min(y + tile_size, h)
                tile = img_bgr[y:y2, x:x2]
                if tile.size == 0:
                    continue
                tile_dets = self._detect(tile)
                for det in tile_dets:
                    tx1, ty1, tx2, ty2 = det["bbox"]
                    detections.append(
                        {
                            "bbox": [x + tx1, y + ty1, x + tx2, y + ty2],
                            "conf": det["conf"],
                            "cls": det["cls"],
                            "cls_name": det["cls_name"],
                        }
                    )

        return self._nms(detections, iou_thresh=nms_iou)

    def detect_on_image(
        self,
        img_bgr,
        split_wide=False,
        split_aspect=2.2,
        split_rel_area=0.4,
        split_min_dets=2,
        tile_detect=False,
        tile_size=640,
        tile_overlap=0.2,
        tile_nms_iou=0.5,
        max_dets=None,
        max_dets_by="area",
    ):
        """Return list of detections for a single BGR image."""
        if tile_detect:
            detections = self._detect_tiled(
                img_bgr,
                tile_size=tile_size,
                overlap=tile_overlap,
                nms_iou=tile_nms_iou,
            )
        else:
            detections = self._detect(img_bgr)
        if not split_wide:
            return self._limit_dets(detections, max_dets, max_dets_by)

        h, w = img_bgr.shape[:2]
        updated = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            bw = max(0, x2 - x1)
            bh = max(0, y2 - y1)
            if bw == 0 or bh == 0:
                continue
            aspect = bw / bh
            rel_area = (bw * bh) / (w * h)
            if aspect <= split_aspect and rel_area <= split_rel_area:
                updated.append(det)
                continue

            crop = img_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                updated.append(det)
                continue

            split_dets = self._detect(crop)
            if len(split_dets) < split_min_dets:
                updated.append(det)
                continue

            for sdet in split_dets:
                sx1, sy1, sx2, sy2 = sdet["bbox"]
                updated.append(
                    {
                        "bbox": [x1 + sx1, y1 + sy1, x1 + sx2, y1 + sy2],
                        "conf": sdet["conf"],
                        "cls": sdet["cls"],
                        "cls_name": sdet["cls_name"],
                    }
                )

        return self._limit_dets(updated, max_dets, max_dets_by)

    @staticmethod
    def _limit_dets(detections, max_dets, max_dets_by):
        if max_dets is None:
            return detections
        if max_dets <= 0:
            return []
        if max_dets_by == "conf":
            key_fn = lambda d: d["conf"]
        else:
            key_fn = lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1])
        return sorted(detections, key=key_fn, reverse=True)[:max_dets]


def load_clip_classifier():
    """Lazy-load a zero-shot classifier to filter non-deer crops."""
    from transformers import pipeline

    # Prefer MPS if available; otherwise fall back to CPU.
    if torch.backends.mps.is_available():
        device_arg = "mps"
    else:
        device_arg = "cpu"

    return pipeline(
        "zero-shot-image-classification",
        model="openai/clip-vit-base-patch32",
        device=device_arg,
    )


def is_likely_deer(img_bgr, classifier, threshold=0.6):
    """Use CLIP zero-shot to keep only crops likely to be full-body deer."""
    labels = [
        "full body deer",
        "deer head closeup",
        "lion",
        "tiger",
        "cat",
        "dog",
        "bird",
        "hawk",
        "squirrel",
        "giraffe",
        "other animal",
    ]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    result = classifier(pil_img, candidate_labels=labels)[0]
    # Keep only when top label is "full body deer" above threshold and not a head-only hit
    return result["label"] == "full body deer" and result["score"] >= threshold


def extract_deer_crops_from_frames(
    frames_root,
    out_root,
    min_conf=0.3,
    model_name="atonus/peura",
    iou=None,
    filter_nondeer: bool = False,
    clip_threshold: float = 0.6,
    min_rel_area: float = 0.02,
    max_rel_area: Optional[float] = None,
    min_aspect: float = 0.3,
    max_aspect: float = 3.5,
    actions_path: Optional[str] = None,
    split_wide: bool = False,
    split_aspect: float = 2.2,
    split_rel_area: float = 0.4,
    split_min_dets: int = 2,
    tile_detect: bool = False,
    tile_size: int = 640,
    tile_overlap: float = 0.2,
    tile_nms_iou: float = 0.5,
    max_dets: Optional[int] = None,
    max_dets_by: str = "area",
):
    frames_root = Path(frames_root)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    detector = DeerDetector(model_name=model_name, conf=min_conf, iou=iou)
    classifier = load_clip_classifier() if filter_nondeer else None
    frame_files = [
        p
        for p in frames_root.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    actions = []

    print(f"Found {len(frame_files)} frames to scan.")

    for frame_path in tqdm(frame_files):
        img = cv2.imread(str(frame_path))
        if img is None:
            print(f"Failed to read {frame_path}")
            continue

        detections = detector.detect_on_image(
            img,
            split_wide=split_wide,
            split_aspect=split_aspect,
            split_rel_area=split_rel_area,
            split_min_dets=split_min_dets,
            tile_detect=tile_detect,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            tile_nms_iou=tile_nms_iou,
            max_dets=max_dets,
            max_dets_by=max_dets_by,
        )
        frame_actions = {
            "frame_path": str(frame_path),
            "detections": [],
        }

        stem = frame_path.stem
        subdir = out_root / frame_path.parent.relative_to(frames_root)
        subdir.mkdir(parents=True, exist_ok=True)

        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det["bbox"]
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            if w == 0 or h == 0:
                continue

            # Filter by size relative to frame to avoid tiny crops (e.g., eyes)
            rel_area = (w * h) / (img.shape[0] * img.shape[1])
            if rel_area < min_rel_area:
                continue
            if max_rel_area is not None and rel_area > max_rel_area:
                continue

            # Filter by aspect ratio to avoid extreme slivers
            aspect = w / h
            if aspect < min_aspect or aspect > max_aspect:
                continue

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            if classifier is not None:
                if not is_likely_deer(crop, classifier, threshold=clip_threshold):
                    continue

            out_name = f"{stem}_det{i:02d}.jpg"
            out_path = subdir / out_name
            cv2.imwrite(str(out_path), crop)
            frame_actions["detections"].append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "conf": det["conf"],
                    "cls": det["cls"],
                    "cls_name": det["cls_name"],
                    "new_name": str(out_path),
                }
            )

        actions.append(frame_actions)

    if actions_path is not None:
        import json

        actions_file = Path(actions_path)
        actions_file.parent.mkdir(parents=True, exist_ok=True)
        with open(actions_file, "w") as f:
            json.dump(actions, f, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-root", default="./tmp_sample_frames")
    parser.add_argument("--out-root", default="./tmp_crops")
    parser.add_argument("--min-conf", type=float, default=0.4)
    parser.add_argument("--model", default="atonus/peura", help="YOLO model name or local path")
    parser.add_argument("--iou", type=float, default=None, help="IoU threshold for NMS")
    parser.add_argument("--filter-nondeer", action="store_true", help="Filter crops with CLIP zero-shot")
    parser.add_argument("--clip-threshold", type=float, default=0.6, help="CLIP score threshold for deer")
    parser.add_argument("--min-rel-area", type=float, default=0.02, help="Min bbox area as fraction of frame")
    parser.add_argument("--max-rel-area", type=float, default=None, help="Max bbox area as fraction of frame")
    parser.add_argument("--min-aspect", type=float, default=0.3, help="Min width/height for bbox")
    parser.add_argument("--max-aspect", type=float, default=3.5, help="Max width/height for bbox")
    parser.add_argument("--actions-path", default=None)
    parser.add_argument("--split-wide", action="store_true", help="Re-detect within wide/large boxes")
    parser.add_argument("--split-aspect", type=float, default=2.2, help="Aspect ratio threshold to split")
    parser.add_argument("--split-rel-area", type=float, default=0.4, help="Relative area threshold to split")
    parser.add_argument("--split-min-dets", type=int, default=2, help="Minimum detections in crop to split")
    parser.add_argument("--tile-detect", action="store_true", help="Run detection over tiled windows")
    parser.add_argument("--tile-size", type=int, default=640, help="Tile size for detection")
    parser.add_argument("--tile-overlap", type=float, default=0.2, help="Tile overlap fraction")
    parser.add_argument("--tile-nms-iou", type=float, default=0.5, help="NMS IoU for tiled detections")
    parser.add_argument("--max-dets", type=int, default=None, help="Max detections per frame")
    parser.add_argument("--max-dets-by", default="area", choices=["area", "conf"], help="Sort key for max dets")
    args = parser.parse_args()

    extract_deer_crops_from_frames(
        frames_root=args.frames_root,
        out_root=args.out_root,
        min_conf=args.min_conf,
        model_name=args.model,
        iou=args.iou,
        filter_nondeer=args.filter_nondeer,
        clip_threshold=args.clip_threshold,
        min_rel_area=args.min_rel_area,
        max_rel_area=args.max_rel_area,
        min_aspect=args.min_aspect,
        max_aspect=args.max_aspect,
        actions_path=args.actions_path,
        split_wide=args.split_wide,
        split_aspect=args.split_aspect,
        split_rel_area=args.split_rel_area,
        split_min_dets=args.split_min_dets,
        tile_detect=args.tile_detect,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        tile_nms_iou=args.tile_nms_iou,
        max_dets=args.max_dets,
        max_dets_by=args.max_dets_by,
    )
