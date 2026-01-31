# Deer Re-Identification Pipeline (Hugging Face + macOS M-series)

Goal: given ~5,000 short videos of ~100 individual deer, automatically detect **which specific deer** appears in each video using batch processing on your Mac (36 GB RAM, 20-core GPU).

## Project Status

- **Current goal**: Run POC 1 on `videos/food1_1.mp4` (sample frames, detect deer, record JSON trail).
- **Current state**:
- Plan captured in this README; pipeline phases and directory stages are outlined.
- Scripts exist for frame extraction and deer detection in `scripts/`.
- Frame extraction supports single video input, frame cap, and actions JSON.
- POC 1 frames extracted for `food1_1.mp4` in `runs/poc1_food1_1/02_frames/`.
- Local detector weights available at `models/peura/train/weights/best.pt`.
- POC 1 detection outputs saved under `runs/poc1_food1_1/03_crops/`.
- POC 1 requires sampling every 10th frame with a hard cap of 10 frames.
- Detection uses `atonus/peura` via `ultralytics` on macOS MPS.
- JSON trails are expected for frames and detections.
- Input is a flat root under `videos/`.
- Output staging for POC 1 is not run yet.
- **TODO**:
- [x] Provide local weights for `atonus/peura` in `models/`.
- [x] Extract frames from `videos/food1_1.mp4` with stride=10 and cap=10, write frame actions JSON.
- [x] Run deer detection on sampled frames and write detection actions JSON.
- [x] Confirm whether deer are present in sampled frames.
- **How to resume**:
- `uv venv .venv && source .venv/bin/activate`
- `uv pip install -r requirements.txt`
- Next file: `README.md`
- **Notes / decisions**:
- Use `videos/food1_1.mp4` for POC 1.
- Frame actions JSON: `runs/poc1_food1_1/02_frames/actions.json`.
- Frame sampling config: every 1.0 seconds, cap at 10 frames.
- Detection actions JSON: `runs/poc1_food1_1/03_crops/actions.json`.
- **Crop quality proposals (ordered by likely success)**:
- Add tighter post-processing on detections: tune NMS IoU lower, enforce stricter min/max bbox area and aspect ratio to reject wide multi-animal boxes.
- Add a segmentation refinement step (SAM or lightweight deer mask) inside each box, then crop to the tightest mask to isolate a single animal.
- Add a two-stage detector pass: if a box is wide/large, re-run detection on the crop (or tiled crop) to split merged animals.
- Add tracklet-based filtering: track detections over frames and drop boxes that merge/split across frames (likely multi-animal).
- Add confidence + proximity heuristics: if boxes overlap heavily, keep higher-confidence; if a box is very wide, re-run detection at higher resolution on that crop.

## Current Understanding and Open Questions (Recorded)

### Current Understanding

- Objective: shift the plan from a simple pipeline to a research-driven, multi-phase modeling strategy focused on best identification results over a long period.
- Scope: likely rewrite all sections.
- Pipeline: more than two steps required; exact structure to be researched.
- Models: not fixed; model selection is part of the research.
- Output formats: not important yet; defer.
- Constraints: must use Python + Hugging Face + GPU.
- You will later install agents/skills to teach me HF carefully; I should be patient and avoid mistakes.

### Open Questions

1) What “results” means (accuracy target, top-k, per-video correctness, etc.)?
2) What ground-truth labels and metadata exist today?
3) Data scale assumptions (video count, length, resolution, FPS)?
4) GPU specifics (M-series vs CUDA, memory, multi-GPU)?
5) Include active learning / human-in-the-loop labeling?
6) Privacy or data-handling constraints (no cloud, etc.)?
7) Open-source HF only, or fine-tune + private HF repos allowed?
8) Timeline granularity for the long-period plan?

### Additional Items to Consider (Recorded)

- Goal is to identify one or more deer in each video.
- Pipeline starts by looping over videos.
- First goal per frame: determine whether there is a deer present.
- Deer are social animals; expect multiple deer in a frame.
- If a video contains no deer, mark it with `NO_DEER`.
- If deer are present, identify each individual deer (multiple IDs per video possible).
- Include a pipeline step to collect the best image(s) for each individual.
- Male deer have antlers only part of the year and can look different the next season.
- Identify individuals by dot patterns on their sides; left and right sides do not match.
- Start with an empty library; learn one individual at a time and build the deer library incrementally.
- Auto-add a new individual only when confidence it is new is very high (e.g., >95%).
- Otherwise place the image in a `waiting_for_confirmation/` directory for human review.
- Human review tools should compare against the current library, show confidence, and show likely confused individuals.
- This auto-add vs. review threshold and workflow is an open discussion to refine later.

## Recent Research Summaries (Recorded)

### NORPPA: NOvel Ringed Seal Re-Identification by Pelage Pattern Aggregation (WACV Workshops 2024)

- Summary: CBIR-style pipeline for patterned animals. Steps: tone mapping, instance segmentation (Mask R-CNN), pelage pattern extraction (U-Net), affine-invariant local features (HesAffNet + HardNet), Fisher Vector aggregation, cosine distance retrieval. Supports top-k review and a similarity threshold for unseen individuals.
- Results (SealID dataset, 57 seals, 2080 images): TOP-1 77.64%, TOP-5 85.27%; outperforms HotSpotter and prior ringed-seal methods.
- Why relevant: emphasizes pattern-first extraction, local feature aggregation, and top-k human review; includes explicit unknown-individual thresholding and no retraining when adding new individuals.

### Combining Feature Aggregation and Geometric Similarity for Re-Identification of Patterned Animals (arXiv 2023)

- Summary: Species-agnostic method combining (1) aggregated local pattern appearance via Fisher vectors and (2) geometric consistency of matched keypoints via RANSAC. Uses Mask R-CNN + U-Net pattern extraction, HesAffNet + HardNet embeddings, and a combined similarity rule.
- Results: On SealID, best combined similarity achieves TOP-1 83.4% (better than NORPPA and HotSpotter). On whale sharks, combined method improves top-1 to 61% (73% with whale-shark-specific pattern extraction).
- Why relevant: explicitly addresses geometric arrangement of patterns and improves robustness across different pattern types; offers a retrieval + verification structure compatible with human review.

### Siamese Network Based Pelage Pattern Matching for Ringed Seal Re-Identification (WACV Workshops 2020)

- Summary: Pattern extraction with Sato tubeness filter, then siamese/triplet patch matching. Full pipeline includes Deeplab segmentation, patch correspondence search, and topology-preserving matching to rank identities. Adds rotation-invariance in the patch encoder.
- Results: Pattern extraction improves patch matching (Top-1 74.6% vs 66.4% without). Full ReID with pattern extraction + rotation + topology: Top-1 67.8%, Top-5 88.6%.
- Why relevant: demonstrates that explicit pattern extraction and patch-level matching can outperform raw-image embeddings; offers topology checks to avoid false matches.

### Towards Multi-Species Animal Re-Identification (WSCG 2024)

- Summary: Evaluates person ReID models (OSNet/PCB/ResNet50 variants) transferred to animals across 20+ datasets plus new ToadID and ZooMix. Uses disjoint train/query/gallery splits and ranks (R-1/R-5/R-10). Finds OSNet AIN generalizes best with pretraining.
- Results: OSNet AIN shows strong Rank-1 on ToadID (85%) but poor on many others; combining related datasets can improve performance; open-world not addressed (closed-world evaluation).
- Why relevant: shows baseline performance and limits of generic ReID backbones; informs whether to use patterned-feature pipelines vs. generic ReID embeddings, and highlights data split/evaluation pitfalls.

## Pipeline Options (Recorded)

### Input Directory Handling (High-Level Options)

**Option 1 — Flat Input Root**
Input: a single directory with videos directly inside.
- Step 1: validate directory exists and scan for video files.
- Step 2: create run output folder under `results/` using timestamp or run id.
- Step 3: iterate videos in sorted order and enqueue for processing.

## Additional Input/Output Rules (Recorded)

- Use Option 1 (flat input root) as the plan baseline.
- Every touched file (video or extracted image) must have an informative name.
- Image filenames must be updated to add relevant data to the name.
- Because names can get too long, maintain a JSON file as a per-image action database.
- Maintain a second JSON file as the library database.
- Library JSON should be a single file.
- Pipeline is a set of directories; items move across directories as stages complete.
- Filenames should be informative but not too long; rename as items move through the pipeline.
- Filenames are not stable across runs; they change by stage.
- Filename must include original file name plus additional fields (kept short).
- JSON fields should be driven by directory/stage; different stages can have different fields.
- Left/right sides are explicitly labeled.
- Storage should be efficient but debuggable; cleanup rules should be well-defined.
- New individual auto-add threshold: 95%+ certainty that it is not already in the library; otherwise request human help.
- Model/version tracking is not required at this time.

## Pipeline Phases After Option D (High-Level)

### Phase 1 — Input Intake + Naming + Run Metadata

- Validate flat input directory and enumerate video files.
- Create a run output folder and run metadata.
- Apply informative naming rules for videos and derived frames.
- Initialize per-image action JSON database and library JSON database.

### Phase 2 — Video Scan + Detection + Tracking

- Loop through videos; detect deer frames.
- Track individuals across frames to form tracklets.
- Mark videos with `NO_DEER` when applicable.

### Phase 3 — Best-Frame Selection + Side Handling

- Select clearest frame per tracklet.
- Determine side (left/right) and treat sides as separate views.
- Rename extracted images with added metadata.
- Record all actions and decisions in per-image JSON database.

### Phase 4 — Pattern Extraction + Embedding + Matching

- Segment deer, extract pattern region.
- Generate HF embeddings and local pattern features.
- Perform similarity search with geometric verification.
- Produce top-k candidates and confusion list with confidence.

### Phase 5 — Open-Set Decision + Library Growth

- Auto-add new individuals only if new-ID confidence >95%.
- Otherwise add to `waiting_for_confirmation/`.
- Update library JSON database with new individuals and best images.

## Directory-Based Pipeline (Draft v3)

### Naming Rules (Short, Informative, Renamed per Stage)

- Each stage moves files into a new directory and renames them.
- Filenames always include the original video base name (shortened if needed).
- Add only the minimum tokens needed for the stage; omit unused tokens.
- Prefer short tokens: `f` (frame), `d` (det), `t` (track), `s` (side), `b` (best), `p` (pattern), `e` (embed).

Example token order (adjust per stage):

```
{orig}__f{frame}__d{det}__t{track}__s{side}__b{best}.jpg
```

### Stage 0 — Input Root (Flat)

Directory: `input_videos/`
- Contents: original videos only.
- Rename rule: keep original filename.
- Stage JSON: `input_videos/index.json`
  - Fields: `orig_name`, `ext`, `size_bytes`, `duration_sec`, `fps`, `ingest_time`.

### Stage 1 — Intake + Run Staging

Directory: `runs/<run_id>/01_intake/`
- Contents: staged video copies or symlinks.
- Rename rule: `{orig}__v{seq}.ext` (seq is per-run order).
- Stage JSON: `runs/<run_id>/01_intake/actions.json`
  - Fields: `orig_name`, `stage_name`, `new_name`, `video_index`, `notes`.

### Stage 2 — Frame Extraction

Directory: `runs/<run_id>/02_frames/`
- Contents: extracted frames.
- Rename rule: `{orig}__f{frame}.jpg`.
- Stage JSON: `runs/<run_id>/02_frames/actions.json`
  - Fields: `orig_name`, `frame`, `time_sec`, `source_video`, `new_name`.

### Stage 3 — Detection Crops

Directory: `runs/<run_id>/03_crops/`
- Contents: deer crops per frame.
- Rename rule: `{orig}__f{frame}__d{det}.jpg`.
- Stage JSON: `runs/<run_id>/03_crops/actions.json`
  - Fields: `orig_name`, `frame`, `det`, `bbox`, `det_conf`, `source_frame`, `new_name`.

### Stage 4 — Tracking / Tracklets

Directory: `runs/<run_id>/04_tracklets/`
- Contents: per-tracklet frames or summaries.
- Rename rule: `{orig}__t{track}__f{frame}.jpg`.
- Stage JSON: `runs/<run_id>/04_tracklets/actions.json`
  - Fields: `orig_name`, `track`, `frame`, `track_conf`, `source_crop`, `new_name`.

### Stage 5 — Best Frame Selection + Side Label

Directory: `runs/<run_id>/05_best/`
- Contents: best frame per tracklet + side label.
- Rename rule: `{orig}__t{track}__s{side}__b1.jpg`.
- Stage JSON: `runs/<run_id>/05_best/actions.json`
  - Fields: `orig_name`, `track`, `side`, `quality_score`, `reason`, `source_frame`, `new_name`.

### Stage 6 — Pattern Extraction (Optional if used)

Directory: `runs/<run_id>/06_pattern/`
- Contents: pattern-extracted images.
- Rename rule: `{orig}__t{track}__s{side}__p.jpg`.
- Stage JSON: `runs/<run_id>/06_pattern/actions.json`
  - Fields: `orig_name`, `track`, `side`, `pattern_method`, `source_best`, `new_name`.

### Stage 7 — Embeddings + Matching

Directory: `runs/<run_id>/07_embed/`
- Contents: embedding vectors (npz/pt) and match results.
- Rename rule: `{orig}__t{track}__s{side}__e.npz`.
- Stage JSON: `runs/<run_id>/07_embed/actions.json`
  - Fields: `orig_name`, `track`, `side`, `embed_path`, `top_k`, `scores`, `confused_ids`.

### Stage 8 — Open-Set Decision

Directory: `runs/<run_id>/08_decisions/`
- Contents: per-track decision records.
- Rename rule: `{orig}__t{track}__s{side}__decision.json`.
- Stage JSON: `runs/<run_id>/08_decisions/actions.json`
  - Fields: `orig_name`, `track`, `side`, `decision` (KNOWN/NEW/REVIEW), `confidence`, `candidate_ids`.

### Stage 9 — Waiting for Confirmation

Directory: `runs/<run_id>/waiting_for_confirmation/`
- Contents: images requiring human review.
- Rename rule: `{orig}__t{track}__s{side}__review.jpg`.
- Stage JSON: `runs/<run_id>/waiting_for_confirmation/actions.json`
  - Fields: `orig_name`, `track`, `side`, `confidence`, `top_k`, `confused_ids`, `review_status`.

### Stage 10 — Library

Directory: `library/`
- Contents: per-deer folders and best images per side.
- Rename rule: `{deer_id}__s{side}__best.jpg` (short and stable inside library).
- Library JSON: `library/library.json`
- Library JSON: `library/library.json`
  - Fields: `deer_id`, `created_at`, `best_left`, `best_right`, `all_images`, `notes`.
  - Concrete schema (single file):
    - `library_version`: string
    - `updated_at`: ISO timestamp
    - `deer`: list of deer entries
      - `deer_id`: string
      - `created_at`: ISO timestamp
      - `best_left`: image path or null
      - `best_right`: image path or null
      - `images_left`: list of image paths
      - `images_right`: list of image paths
      - `source_runs`: list of run ids
      - `notes`: string (optional)

### NO_DEER Handling

Directory: `runs/<run_id>/no_deer/`
- Contents: marker files or per-video JSON.
- Stage JSON: `runs/<run_id>/no_deer/actions.json`
  - Fields: `orig_name`, `video`, `reason`, `frames_scanned`.

## Accuracy and Efficiency Adjustments From Literature

- Use fast retrieval + verification: embeddings shortlist top-k, then geometric verification on the shortlist (patterned-animals paper).
- Make segmentation a required step before pattern extraction; use it to reduce background bias (ringed-seal papers).
- Best-frame selection should prioritize pattern visibility/contrast, not just sharpness.
- Maintain multi-shot gallery per individual and per side; avoid single-shot brittleness.
- If using person-ReID backbones (OSNet family), keep square inputs and disjoint train/query/gallery splits.
- Pattern extraction can be species-agnostic but may need deer-specific fine-tuning if accuracy is weak.

## Planning Updates Requested (Recorded)

### Video Sampling Clarity

- Explicitly define sampling: every 10th frame (N = 10).
- Add a max-frames-per-video cap to avoid full conversion (max 10 per video).
- Record sampling config in per-stage JSON.

### Best-Frame Selection Details

- For a detection at frame F, scan a local window (F-k to F+k).
- Score candidates by: pattern visibility (primary), crop size, sharpness, occlusion, side visibility.
- Save best-frame decision rationale in stage JSON.

### Library JSON

- Define a minimal, explicit library JSON schema (per-deer, per-side, best images).
- Keep lineage to run/tracklet without bloating filenames.

### Missing Pieces

- Deep-dive needed on unfilled components once approved (sampling, best-frame, library JSON, and remaining pipeline gaps).

## POC (Draft, Recorded)

### POC 1 — Plumbing + Feasibility Checks (1 Video)

Goal: confirm local plumbing works and core modeling steps run on one video.

Scope:
- 1 video only.
- Sampling: every Nth frame, hard cap = 10 frames total.

Checks:
1) Extract frames (every 10th frame, cap=10) into `runs/<run_id>/02_frames/`.
2) Run deer detection locally on those frames.
3) Answer: can we identify deer yes/no in sampled frames.
4) JSON trail created for frames and detections.

Success = “ready for next step” if all checks pass end-to-end.

### POC 2 — Detection at Small Scale (10 Videos)

Goal: confirm we can reliably detect deer presence across a small batch.

Scope:
- 10 videos.
- Sampling: every 10th frame, hard cap = 10 frames per video.

Checks:
1) Extract frames per video with cap=10.
2) Run detection and record `NO_DEER` vs `DEER_PRESENT` per video.
3) Confirm runtime, stability, and basic accuracy of deer presence detection.

Success = “ready for next step” if detection runs across all 10 videos and outputs `NO_DEER`/`DEER_PRESENT` per video.

### POC Steps

1) Sampling + detection
   - Sample frames at fixed FPS (e.g., 1 fps) and cap frames/video.
   - Run detector on sampled frames only.

2) Best-frame selection test
   - For each detection at frame F, scan F-k…F+k.
   - Score frames by pattern visibility, crop size, sharpness, occlusion.
   - Compare best-frame-only vs random vs all frames.

3) Embedding model test (HF backbone)
   - Compute embeddings on crops.
   - Evaluate top-1/top-3 within left/right galleries.

4) Geometric verification test
   - Apply geometric verification to top-k matches.
   - Measure false match reduction vs embeddings alone.

5) Open-set gating (95%)
   - Hold out one deer as unknown.
   - Measure rejection rate at 95% new-ID threshold.

6) Tracklet aggregation
   - Aggregate embeddings across frames in a tracklet.
   - Compare tracklet-mean vs single-frame accuracy.

### POC Success Criteria (Initial)

- POC success is defined as "ready for next step" (plumbing works end-to-end).

**Option 2 — Structured Input Root**
Input: a root directory with subfolders (site/date/camera).
- Step 1: validate root, scan recursively for video files.
- Step 2: preserve relative paths in output (mirror subfolders in `results/`).
- Step 3: process in a stable order (by subfolder then filename).

**Option 3 — Manifest-Driven Input**
Input: a root directory plus a manifest file listing video paths and metadata.
- Step 1: validate root and manifest, resolve all paths.
- Step 2: create run output folder and per-video records with metadata.
- Step 3: process in manifest order; missing files get logged and skipped.

### Option D (Suggestion) — Hybrid Pattern-First + Tracklet Aggregation + Open-Set HF Embeddings

- Pattern-first retrieval with geometric verification.
- Multi-object tracking to build per-deer tracklets and aggregate evidence.
- HF-backed embedding model + open-set gating for new individuals.
- Steps (high-level): video loop + detection → tracking/tracklets → select clearest frame per tracklet + side → instance segmentation → pattern extraction → HF embedding → local similarity + geometric verification → top-k ranking → auto-add if new-ID confidence >95% else `waiting_for_confirmation/` with confusion list + scores → update library (left/right sides treated separately).
- Notes: use the clearest frame for ID; left/right sides do not match; human review tool should show confidence and likely confusions.

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
