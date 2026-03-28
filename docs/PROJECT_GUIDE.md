# Deer Re-Identification Project Guide

> For AI coding assistants. Read `AGENT_HANDOFF.md` first for the current task.

---

## Goal

Identify individual deer by their flank spot patterns. New trail camera image → match against a gallery of enrolled known deer → ranked results with confidence.

- Left and right flanks have **different** patterns — treated as separate views, never matched across sides
- Antlers change seasonally — do NOT use for ID
- Runs on macOS Apple Silicon (M-series, 36GB RAM, MPS GPU)

---

## Current State

The enrollment/identification tool is **complete** (4 scripts, all tested). See `AGENT_HANDOFF.md` for bugs and next steps.

The clustering codebase (`scripts/reid/`) exists but is no longer the active path — it was abandoned because 253 near-duplicate frames from 6 encounters could not produce reliable identity clusters. The correct approach is enrollment-based matching.

---

## Architecture

```
Phase A (done): Video → Frame Extraction → YOLO Detection → Tracklets → Flank Gate → Flank Pool

Phase B (active):
  Enrollment:      images/ → enrollment_ui.py → user assigns deer IDs → gallery.json
  Identification:  new photo (pre-cropped) → MegaDescriptor embed → cosine similarity → ranked results
```

### Models

| Model | Purpose | Location |
|-------|---------|----------|
| YOLOv8n (`atonus/peura`) | Deer detection | `models/peura/train/weights/best.pt` |
| CLIP ViT-B/32 | Gate classifier backbone | via `openai/clip-vit-base-patch32` |
| MegaDescriptor-L-384 | Re-ID embeddings | `hf_hub:BVRA/MegaDescriptor-L-384` via timm |
| Flank gate | Keep flank-visible crops | `data/models/flank_gate.pt` (threshold 0.56) |
| Side gate | Predict left/right/unknown | `data/models/side_gate/` (threshold 0.55) |

---

## Hard Constraints

### Left ≠ Right (non-negotiable)
Left and right flank patterns are biologically different for the same deer. A left-side query must only compare against left-side gallery entries — never cross sides. `rank_matches()` enforces this strictly.

### Show full images to users
Users cannot recognize individual deer from a crop alone. Any review or enrollment UI must show the **full original trail camera photo**, not just the crop.

### Source images
- Target deer: `images/FD/` (265 images) and `images/DEER-IMG-2023/` (394 images) — raw ranger photos, unlabeled
- Playground data (`data/reid/`, 253 crops from practice videos) — **different deer population**, valid for hypothesis testing only

### Side determination
Treat side gate output as a pre-fill suggestion only — always let the user confirm or override.

---

## Key File Locations

| What | Where |
|------|-------|
| Current task / bugs | `AGENT_HANDOFF.md` |
| Gallery | `data/gallery/gallery.json` |
| Flank crops (253, playground) | `data/reid/flank_pool/` |
| Instance index (has `side_pred`) | `data/reid/index.csv` |
| Cluster name suggestions (enrollment UI only) | `data/reid/clusters.csv` |
| MegaDescriptor load pattern | `scripts/gallery_utils.py:load_megadescriptor()` |
| HTTP server pattern | `scripts/enrollment_ui.py` |
| Flank gate model | `data/models/flank_gate.pt` |
| YOLO weights | `models/peura/train/weights/best.pt` |

---

## Environment

```bash
source .venv/bin/activate
.venv/bin/python -m pytest -q    # always run before and after changes
ruff check . && ruff format .
```

---

## Rules

- One change at a time. Small diffs.
- Never break CLI flags, CSV column names, or `actions.json` schema.
- No new dependencies without justification.
- Every changed behavior needs a test.
- Read `AGENTS.md` for the full list.

---

## Glossary

| Term | Meaning |
|------|---------|
| Tracklet | Sequence of crops of the same deer across consecutive frames |
| Flank gate | Binary classifier: keep only crops showing the deer's flank |
| Side gate | Classifier: left / right / unknown orientation |
| Encounter | Burst of images from same deer in a short time window |
| MegaDescriptor | SOTA wildlife re-ID backbone (Swin Transformer), used for gallery embeddings |
| Prototype | Mean of all embeddings for one deer+side, L2-normalized |
