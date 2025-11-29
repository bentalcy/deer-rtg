import cv2
import torch
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
from typing import Optional
from PIL import Image

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


class DeerDetector:
    def __init__(self, model_name="atonus/peura", conf=0.3):
        self.device = DEVICE
        self.model = YOLO(model_name).to(self.device)
        self.conf = conf

    def detect_on_image(self, img_bgr):
        """Return list of detections for a single BGR image."""
        results = self.model.predict(
            img_bgr,
            device=self.device,
            conf=self.conf,
            verbose=False,
        )
        detections = []

        for r in results:
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
    filter_nondeer: bool = False,
    clip_threshold: float = 0.6,
    min_rel_area: float = 0.02,
    max_rel_area: Optional[float] = None,
    min_aspect: float = 0.3,
    max_aspect: float = 3.5,
):
    frames_root = Path(frames_root)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    detector = DeerDetector(model_name=model_name, conf=min_conf)
    classifier = load_clip_classifier() if filter_nondeer else None
    frame_files = [p for p in frames_root.rglob("*.jpg")]

    print(f"Found {len(frame_files)} frames to scan.")

    for frame_path in tqdm(frame_files):
        img = cv2.imread(str(frame_path))
        if img is None:
            print(f"Failed to read {frame_path}")
            continue

        detections = detector.detect_on_image(img)
        if not detections:
            continue

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-root", default="./tmp_sample_frames")
    parser.add_argument("--out-root", default="./tmp_crops")
    parser.add_argument("--min-conf", type=float, default=0.4)
    parser.add_argument("--model", default="atonus/peura", help="YOLO model name or local path")
    parser.add_argument("--filter-nondeer", action="store_true", help="Filter crops with CLIP zero-shot")
    parser.add_argument("--clip-threshold", type=float, default=0.6, help="CLIP score threshold for deer")
    parser.add_argument("--min-rel-area", type=float, default=0.02, help="Min bbox area as fraction of frame")
    parser.add_argument("--max-rel-area", type=float, default=None, help="Max bbox area as fraction of frame")
    parser.add_argument("--min-aspect", type=float, default=0.3, help="Min width/height for bbox")
    parser.add_argument("--max-aspect", type=float, default=3.5, help="Max width/height for bbox")
    args = parser.parse_args()

    extract_deer_crops_from_frames(
        frames_root=args.frames_root,
        out_root=args.out_root,
        min_conf=args.min_conf,
        model_name=args.model,
        filter_nondeer=args.filter_nondeer,
        clip_threshold=args.clip_threshold,
        min_rel_area=args.min_rel_area,
        max_rel_area=args.max_rel_area,
        min_aspect=args.min_aspect,
        max_aspect=args.max_aspect,
    )
