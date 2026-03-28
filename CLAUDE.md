# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Deer re-identification pipeline: detect, track, and identify individual deer across ~5,000 short videos using ~100 known deer. Runs on macOS M-series (36 GB RAM, 20-core GPU). All pipeline stages write outputs to `runs/{run_id}/` and aggregated re-ID data to `data/reid/`.

## Environment Setup

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Device auto-selection: code checks `torch.backends.mps.is_available()` for M-series GPU, falls back to CPU.

## Commands

```bash
pytest -q                                 # run all tests
pytest tests/test_review_cluster_outliers_web.py  # run single test file
ruff check . && ruff format . && mypy .   # lint + type check
```

**Key pipeline entry points:**
```bash
python scripts/extract_frames.py --videos-root <dir> --out-root runs/<run_id>
python scripts/build_tracklets.py --run-dir runs/<run_id>
python scripts/reid/build_index.py
python scripts/reid/embed_dinov2.py
python scripts/reid/cluster_hdbscan.py && python scripts/reid/cluster_nn.py
python scripts/reid/build_review_queue.py
python scripts/reid/review_cluster_outliers_web.py   # main review UI (Flask)
python scripts/reid/dev_review_cluster_outliers_web.py  # with hot reload
```

## Pipeline Architecture

Data flows through numbered stage directories inside each run:

```
Raw Videos
  → [extract_frames.py]    → runs/{run_id}/02_frames/ + actions.json
  → [build_tracklets.py]   → runs/{run_id}/04_tracklets/track_NNN/ + actions.json
  → [score_flank_gate.py]  → filters by flank visibility (model: data/models/flank_gate.pt, threshold 0.56)
  → [reid/build_index.py]  → data/reid/index.csv  (aggregates all runs)
  → [reid/embed_dinov2.py] → data/reid/embeddings.pt (shape [N, 768]) + embeddings.csv
  → [cluster_hdbscan.py + cluster_nn.py]  → dual-view clustering CSVs + summary JSONs
  → [reid/build_review_queue.py]          → data/reid/review_queue.csv (disagreement medoid pairs)
  → [review_cluster_outliers_web.py]      → data/reid/clusters_corrected.csv + decisions.csv
  → [reid/build_splits.py]               → data/reid/splits.csv (encounter-based, no leakage)
```

Every stage writes an `actions.json` alongside outputs capturing metadata (frame indices, detection bboxes, track IDs, flank scores, etc.). Downstream scripts read these JSONs — never break their schema.

## Key Design Constraints

- **Flank crops only** — antlers vary by season; left/right sides are treated as separate identities (never matched across sides).
- **Dual-view clustering** — HDBSCAN (density) + nearest-neighbor (threshold) run independently; disagreements go to human review, not automated resolution.
- **Encounter-based splits** — all crops from one encounter stay in the same split to prevent leakage. `build_splits.py` enforces this; don't relax it.
- **Model checkpoint format**: `{"model_name", "label_to_idx", "state_dict", ...}` — keep this schema when saving/loading gate models.
- **Gate classifiers use CLIP** (`openai/clip-vit-base-patch32`) embeddings + a small head. DINOv2 (`dinov2_vitb14`) is used only for re-ID embeddings.

## Code Standards (from AGENTS.md)

- Behave like a junior engineer: one change at a time, minimal blast radius, ask when unsure.
- Preserve all contracts: CLI flags, CSV column names, JSON schemas, output file paths.
- Nesting ≤ 3 levels; separate refactors from behavior changes.
- Prefer boring solutions; avoid new dependencies.
- Every changed behavior needs a test; run tests before and after.
- State rollback path in PR description.

## Current Goal

**Build an enrollment-based identification tool — 4 scripts, no training required.**

The previous clustering → ArcFace plan was abandoned. See `AGENT_HANDOFF.md` for full context.

Goal: user manually enrolls known deer into a reference gallery, then `identify_deer.py` matches any new photo against the gallery using **MegaDescriptor-L-384** cosine similarity (threshold ~0.57).

**Hypothesis validation complete (task #5):** MegaDescriptor-L-384 achieves 14/15 (93%) vs DINOv2's 13/15 (87%) on user-judged pairs. MegaDescriptor correctly ignores collar/injury shortcuts that fooled DINOv2. All new scripts must use MegaDescriptor.

### Scripts to build (none exist yet)
| Script | Purpose |
|--------|---------|
| `scripts/gallery_utils.py` | Shared: gallery I/O, cosine similarity, rank_matches, embed_single |
| `scripts/enroll_deer.py` | CLI: enroll one deer crop into the gallery |
| `scripts/identify_deer.py` | CLI: match a new photo against the gallery, print ranked results |
| `scripts/enrollment_ui.py` | Web UI: batch-enroll from the 253 existing crops (mirrors `review_ui.py` pattern) |

### Embedding model
```python
import timm
model = timm.create_model("hf_hub:BVRA/MegaDescriptor-L-384", pretrained=True, num_classes=0)
cfg = timm.data.resolve_model_data_config(model)
transform = timm.data.create_transform(**cfg, is_training=False)
# L2-normalize all outputs before cosine similarity
# Threshold: ~0.57 (re-calibrate once gallery has >5 deer)
```
See `scripts/test_megadescriptor_hypothesis.py` for the canonical load+embed pattern.

### What already exists and is reusable
- HTTP server pattern for UI: see `scripts/reid/review_ui.py` — stdlib `BaseHTTPRequestHandler`, no Flask
- Old DINOv2 embeddings (`data/reid/embeddings.pt`) are playground-only, not used by new scripts

### Gallery format
`data/gallery/gallery.json`:
```json
{"5A": {"left": {"embeddings": [[...768 floats...]], "image_paths": ["..."]}, "right": {...}}}
```
Confidence = `(cosine_similarity + 1) / 2 * 100`
