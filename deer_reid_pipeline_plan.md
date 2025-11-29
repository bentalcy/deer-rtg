# Deer Re-Identification Pipeline (Hugging Face + macOS M-series)

Goal: given ~5,000 short videos of ~100 individual deer, automatically detect **which specific deer** appears in each video using batch processing on your Mac (36 GB RAM, 20-core GPU).

The pipeline is 2-stage:

1. **Detection** – find deer in frames (bounding boxes).
2. **Re-ID (re-identification)** – say *which* deer this is by comparing visual features to a labeled gallery of known individuals.

---

## 1. Architecture Overview

### 1.1. Models

- **Deer detector (object detection)**  
  - Hugging Face: `atonus/peura`  
  - Type: YOLOv8n fine-tuned specifically for deer vs background.  
  - Role: find deer bounding boxes in each frame.

- **Deer re-identification backbone (feature extractor)**  
  - Hugging Face: `conservationxlabs/miewid-msv3`  
  - Type: wildlife re-identification model (EfficientNetV2 backbone) trained with contrastive learning over many animal species.  
  - Role: convert each deer crop into a high-dimensional embedding vector such that **same individual ~ similar embeddings**, different individuals ~ dissimilar.

### 1.2. Data Flow

For each video:

1. Sample frames (e.g. 1–2 fps).
2. Run `peura` → deer bounding boxes.
3. Crop each box → deer image patch.
4. Run `miewid-msv3` → embedding vector.
5. Compare embedding to a **gallery** of known deer embeddings (built from labeled examples).
6. If similarity above threshold → assign that deer ID; else mark as “unknown”.

Example output per video:

```json
{
  "video": "VID_000123.mp4",
  "detections": [
    {
      "frame": 120,
      "time_sec": 4.0,
      "deer_id": "deer_017",
      "similarity": 0.84,
      "bbox": [100, 50, 300, 280]
    },
    {
      "frame": 135,
      "time_sec": 4.5,
      "deer_id": "deer_017",
      "similarity": 0.81,
      "bbox": [110, 55, 310, 285]
    }
  ],
  "deer_in_video": ["deer_017"]
}
```

---

## 2. Environment Setup (macOS, Python, MPS)

### 2.1. System Requirements

- macOS on Apple Silicon (M-series) – your 36 GB is plenty.
- Python 3.10+.
- `ffmpeg` for video handling (via Homebrew).

```bash
# Homebrew (if not installed)
# /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

brew install ffmpeg

# Create a clean virtualenv
python3 -m venv ~/venv/deer_reid
source ~/venv/deer_reid/bin/activate

pip install --upgrade pip

# Core packages (CPU index URL is fine; it still uses MPS on Mac)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Detection + transformers + utils
pip install ultralytics transformers opencv-python numpy pillow tqdm matplotlib
```

Quick sanity check of MPS:

```python
import torch
print("MPS available:", torch.backends.mps.is_available())
```

You want this to print `True`. Device string will be `"mps"`.

---

## 3. Directory Layout

Recommended minimal layout:

```text
project_root/
  videos_raw/          # all original videos (.mp4, .mov, ...)
  frames_gallery/      # extracted frames you will manually label for known deer
  frames_gallery_crops/# deer crops extracted by YOLO
  gallery_labels.csv   # mapping from (crop) image path -> deer_id
  gallery_embeds.npz   # saved embeddings for the gallery (generated later)
  gallery_meta.json    # metadata about deer IDs and counts
  results/             # output JSON/CSV per video
  scripts/             # all Python scripts below
```

Example `gallery_labels.csv` (you will create/edit this):

```csv
frame_path,deer_id
frames_gallery_crops/VID0001_frame000120_det00.jpg,deer_001
frames_gallery_crops/VID0001_frame000150_det00.jpg,deer_001
frames_gallery_crops/VID0002_frame000090_det00.jpg,deer_002
```

You do **not** need to label all frames, just **good representative crops** for each individual (20–50 per deer is a reasonable starting point).

---

## 4. Step 1 – Extract Frames from Videos

First script: sample frames from every video to help you create the initial labeled gallery and debug detection.

Create `scripts/extract_frames.py`:

```python
import cv2
from pathlib import Path
from tqdm import tqdm

VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi")


def extract_frames_from_video(video_path, out_dir, stride=30):
    """
    Extract every `stride`-th frame from a video.

    Example: stride=30 at 30 FPS is roughly 1 frame per second.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return 0

    frame_idx = 0
    saved = 0
    stem = Path(video_path).stem  # video filename without extension

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % stride == 0:
            out_name = f"{stem}_frame{frame_idx:06d}.jpg"
            out_path = out_dir / out_name
            cv2.imwrite(str(out_path), frame)
            saved += 1

        frame_idx += 1

    cap.release()
    return saved


def main(videos_root, out_root, stride=30):
    videos_root = Path(videos_root)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    video_files = [
        p for p in videos_root.rglob("*")
        if p.suffix.lower() in VIDEO_EXTS
    ]

    print(f"Found {len(video_files)} videos.")

    for vid in tqdm(video_files):
        rel = vid.relative_to(videos_root)
        vid_out_dir = out_root / rel.parent
        vid_out_dir.mkdir(parents=True, exist_ok=True)
        n = extract_frames_from_video(vid, vid_out_dir, stride=stride)
        print(f"{vid}: saved {n} frames")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--videos-root", default="../videos_raw")
    parser.add_argument("--out-root", default="../frames_gallery")
    parser.add_argument("--stride", type=int, default=30)
    args = parser.parse_args()

    main(args.videos_root, args.out_root, args.stride)
```

Run:

```bash
cd project_root/scripts
python extract_frames.py --videos-root ../videos_raw --out-root ../frames_gallery --stride 30
```

You now have sampled frames in `frames_gallery/`. You can either:

- Label directly on these frames, or  
- First run detection and label on the crops (recommended).

---

## 5. Step 2 – Deer Detection and Crop Extraction (YOLO `atonus/peura`)

Script to load `peura` and produce crops for any deer detection in a frame. This is used both for building the gallery and later for video processing.

Create `scripts/deer_detector.py`:

```python
import cv2
import torch
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


class DeerDetector:
    def __init__(self, model_name="atonus/peura", conf=0.3):
        self.device = DEVICE
        self.model = YOLO(model_name).to(self.device)
        self.conf = conf

    def detect_on_image(self, img_bgr):
        """
        img_bgr: HxWx3 BGR image (OpenCV)
        returns: list of dicts with bbox and confidence
        """
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


def extract_deer_crops_from_frames(frames_root, out_root, min_conf=0.3):
    frames_root = Path(frames_root)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    detector = DeerDetector(conf=min_conf)

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
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            out_name = f"{stem}_det{i:02d}.jpg"
            out_path = subdir / out_name
            cv2.imwrite(str(out_path), crop)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-root", default="../frames_gallery")
    parser.add_argument("--out-root", default="../frames_gallery_crops")
    parser.add_argument("--min-conf", type=float, default=0.3)
    args = parser.parse_args()

    extract_deer_crops_from_frames(
        frames_root=args.frames_root,
        out_root=args.out_root,
        min_conf=args.min_conf,
    )
```

Run:

```bash
python deer_detector.py --frames-root ../frames_gallery --out-root ../frames_gallery_crops --min-conf 0.3
```

Now you have **deer crops** in `frames_gallery_crops/`. These are easier to label per individual.

Update `gallery_labels.csv` so that `frame_path` points to these crops.

---

## 6. Step 3 – Build the Re-ID Gallery (MiewID Embeddings)

Now we create a **gallery of embeddings** using `conservationxlabs/miewid-msv3`.

Create `scripts/build_gallery.py`:

```python
import csv
import json
import cv2
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModel
import torchvision.transforms as T
from tqdm import tqdm
from numpy.linalg import norm

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def load_reid_model(model_name="conservationxlabs/miewid-msv3"):
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.to(DEVICE)
    model.eval()
    return model


PREPROCESS = T.Compose([
    T.ToPILImage(),
    T.Resize((440, 440)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


def embed_image(model, img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    x = PREPROCESS(img_rgb).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(x)

    # Model may return a dict or tensor; handle both
    if isinstance(out, dict):
        feats = out.get("last_hidden_state", None)
        if feats is None:
            feats = next(iter(out.values()))
        feat = feats.mean(dim=1).cpu().numpy()[0]
    else:
        feat = out.cpu().numpy()[0]

    # L2-normalize
    feat = feat / (norm(feat) + 1e-8)
    return feat


def build_gallery(crops_root, labels_csv,
                  out_path="gallery_embeds.npz",
                  meta_path="gallery_meta.json"):
    """
    crops_root: root folder for deer crops
    labels_csv: CSV mapping crop image path -> deer_id
    """
    crops_root = Path(crops_root)
    label_map = {}

    # Load labels
    with open(labels_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_path = Path(row["frame_path"])
            if not frame_path.is_absolute():
                frame_path = crops_root / frame_path
            deer_id = row["deer_id"]
            label_map[str(frame_path)] = deer_id

    print(f"Loaded {len(label_map)} labeled crops.")

    model = load_reid_model()
    deer_to_embs = {}

    for img_path_str, deer_id in tqdm(label_map.items()):
        img_path = Path(img_path_str)
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Failed to read image: {img_path}")
            continue

        emb = embed_image(model, img)
        deer_to_embs.setdefault(deer_id, []).append(emb)

    # Convert to arrays for saving
    npz_dict = {}
    meta = {"deer_ids": []}

    for deer_id, emb_list in deer_to_embs.items():
        arr = np.stack(emb_list, axis=0)
        key = f"deer_{deer_id}"
        npz_dict[key] = arr
        meta["deer_ids"].append({
            "key": key,
            "deer_id": deer_id,
            "count": int(arr.shape[0]),
        })

    np.savez(out_path, **npz_dict)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved gallery embeddings to {out_path}")
    print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--crops-root", default="../frames_gallery_crops")
    parser.add_argument("--labels-csv", default="../gallery_labels.csv")
    parser.add_argument("--out-path", default="../gallery_embeds.npz")
    parser.add_argument("--meta-path", default="../gallery_meta.json")
    args = parser.parse_args()

    build_gallery(
        crops_root=args.crops_root,
        labels_csv=args.labels_csv,
        out_path=args.out_path,
        meta_path=args.meta_path,
    )
```

Run once you have some labeled crops:

```bash
python build_gallery.py   --crops-root ../frames_gallery_crops   --labels-csv ../gallery_labels.csv   --out-path ../gallery_embeds.npz   --meta-path ../gallery_meta.json
```

At this point you have a **re-ID gallery** of known individuals.

---

## 7. Step 4 – Batch Process Videos and Assign Deer IDs

Now wire everything together: detection + re-ID + similarity search + per-video summary.

Create `scripts/process_videos.py`:

```python
import json
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from numpy.linalg import norm
from transformers import AutoModel
import torchvision.transforms as T
from ultralytics import YOLO

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi")


# 1. Detector
class DeerDetector:
    def __init__(self, model_name="atonus/peura", conf=0.3):
        self.device = DEVICE
        self.model = YOLO(model_name).to(self.device)
        self.conf = conf

    def detect_on_frame(self, frame_bgr):
        results = self.model.predict(
            frame_bgr,
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


# 2. Re-ID model
PREPROCESS = T.Compose([
    T.ToPILImage(),
    T.Resize((440, 440)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


def load_reid_model(model_name="conservationxlabs/miewid-msv3"):
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.to(DEVICE)
    model.eval()
    return model


def embed_image(model, img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    x = PREPROCESS(img_rgb).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(x)

    if isinstance(out, dict):
        feats = out.get("last_hidden_state", None)
        if feats is None:
            feats = next(iter(out.values()))
        feat = feats.mean(dim=1).cpu().numpy()[0]
    else:
        feat = out.cpu().numpy()[0]

    feat = feat / (norm(feat) + 1e-8)
    return feat


# 3. Gallery loading + similarity search
def load_gallery(npz_path, meta_path):
    data = np.load(npz_path)
    with open(meta_path) as f:
        meta = json.load(f)

    gallery = {}
    for entry in meta["deer_ids"]:
        key = entry["key"]
        deer_id = entry["deer_id"]
        arr = data[key]  # shape [N, D]
        gallery[deer_id] = arr

    return gallery


def find_best_match(query_emb, gallery):
    best_id, best_sim = None, -1.0
    for deer_id, arr in gallery.items():
        # arr: [N, D]; query_emb: [D]
        sims = arr @ query_emb
        idx = int(np.argmax(sims))
        sim = float(sims[idx])
        if sim > best_sim:
            best_sim, best_id = sim, deer_id
    return best_id, best_sim


# 4. Main video processing
def process_video(video_path, detector, reid_model, gallery,
                  frame_stride=15, sim_thresh=0.75):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Failed to open {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = 0
    matches = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_stride == 0:
            detections = detector.detect_on_frame(frame)
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                emb = embed_image(reid_model, crop)
                deer_id, sim = find_best_match(emb, gallery)
                if deer_id is not None and sim >= sim_thresh:
                    time_sec = frame_idx / fps
                    matches.append({
                        "frame": frame_idx,
                        "time_sec": time_sec,
                        "deer_id": deer_id,
                        "similarity": sim,
                        "bbox": [x1, y1, x2, y2],
                    })

        frame_idx += 1

    cap.release()
    return matches


def batch_process_videos(videos_root, out_root,
                         gallery_npz, gallery_meta,
                         frame_stride=15, sim_thresh=0.75,
                         conf=0.3):
    videos_root = Path(videos_root)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    video_files = [
        p for p in videos_root.rglob("*")
        if p.suffix.lower() in VIDEO_EXTS
    ]

    print(f"Found {len(video_files)} videos.")

    detector = DeerDetector(conf=conf)
    reid_model = load_reid_model()
    gallery = load_gallery(gallery_npz, gallery_meta)

    for vid in tqdm(video_files):
        rel = vid.relative_to(videos_root)
        out_dir = out_root / rel.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_json = out_dir / f"{vid.stem}_deer_ids.json"

        matches = process_video(
            video_path=vid,
            detector=detector,
            reid_model=reid_model,
            gallery=gallery,
            frame_stride=frame_stride,
            sim_thresh=sim_thresh,
        )

        deer_ids = sorted({m["deer_id"] for m in matches})
        result = {
            "video": str(vid),
            "deer_in_video": deer_ids,
            "matches": matches,
        }

        with open(out_json, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--videos-root", default="../videos_raw")
    parser.add_argument("--out-root", default="../results")
    parser.add_argument("--gallery-npz", default="../gallery_embeds.npz")
    parser.add_argument("--gallery-meta", default="../gallery_meta.json")
    parser.add_argument("--frame-stride", type=int, default=15)
    parser.add_argument("--sim-thresh", type=float, default=0.75)
    parser.add_argument("--conf", type=float, default=0.3)
    args = parser.parse_args()

    batch_process_videos(
        videos_root=args.videos_root,
        out_root=args.out_root,
        gallery_npz=args.gallery_npz,
        gallery_meta=args.gallery_meta,
        frame_stride=args.frame_stride,
        sim_thresh=args.sim_thresh,
        conf=args.conf,
    )
```

Run:

```bash
python process_videos.py   --videos-root ../videos_raw   --out-root ../results   --gallery-npz ../gallery_embeds.npz   --gallery-meta ../gallery_meta.json   --frame-stride 15   --sim-thresh 0.75   --conf 0.3
```

Each video will get a JSON file in `results/` describing which deer were found and where.

---

## 8. Tuning and Next Steps

1. **Thresholds**  
   - `sim_thresh` (default 0.75) controls how strict the match is.  
     - Too low → many false matches.  
     - Too high → many “unknown” even when it is actually the same deer.

2. **Frame stride**  
   - Lower stride (e.g. 5) = more frames, better coverage, but slower.  
   - Higher stride (e.g. 30) = fewer frames, faster, higher risk of missing short appearances.

3. **Gallery size**  
   - More labeled crops per deer improve robustness across lighting, angles, distance.  
   - Start with ~20 per deer, then add more for “confused pairs”.

4. **Potential later upgrades**  
   - Hard-negative mining: add crops from “confusable” deer to sharpen separation.  
   - Replace per-crop sets by per-deer centroids (average embedding per deer).  
   - Temporal smoothing: enforce consistency across neighboring frames.

This gives you a complete end-to-end batch pipeline using Hugging Face models and Python on your Mac. Adjust folders, thresholds, and frame stride as needed.
