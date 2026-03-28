# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Deer re-identification pipeline: rangers label trail camera photos in a browser UI, then `identify_deer.py` matches new photos against the gallery using MegaDescriptor-L-384 cosine similarity. Runs on macOS M-series (36 GB RAM). Pipeline stages write outputs to `runs/{run_id}/` and aggregated re-ID data to `data/reid/`.

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

**Key entry points:**
```bash
python scripts/enrollment_ui.py --host 127.0.0.1 --port 8765   # ranger labeling UI
python scripts/enroll_deer.py --image <path> --deer-id <id> --side <left|right>
python scripts/identify_deer.py --image <path>
```

## Pipeline Architecture

```
Phase A (complete): Videos → extract_frames.py → build_tracklets.py → score_flank_gate.py → data/reid/

Phase B (active):
  Enrollment:      images/ → enrollment_ui.py → ranger assigns deer IDs → data/gallery/gallery.json
  Identification:  new photo (pre-cropped) → MegaDescriptor embed → cosine similarity → ranked results
```

Every stage writes an `actions.json` alongside outputs. Downstream scripts read these JSONs — never break their schema.

## Key Design Constraints

- **Flank crops only** — antlers vary by season; left/right sides are treated as separate identities (never matched across sides).
- **Model checkpoint format**: `{"model_name", "label_to_idx", "state_dict", ...}` — keep this schema when saving/loading gate models.
- **Gate classifiers use CLIP** (`openai/clip-vit-base-patch32`) embeddings + a small head.
- **Re-ID embeddings use MegaDescriptor-L-384** — all new scripts must use this model, not DINOv2.

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

### Scripts (all built and tested)
| Script | Purpose |
|--------|---------|
| `scripts/gallery_utils.py` | Shared: gallery I/O, cosine similarity, rank_matches, embed_single |
| `scripts/enroll_deer.py` | CLI: enroll one deer crop into the gallery |
| `scripts/identify_deer.py` | CLI: match a new photo against the gallery, print ranked results |
| `scripts/enrollment_ui.py` | Web UI: ranger labels photos one-at-a-time, saves to gallery |

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

### Gallery format
`data/gallery/gallery.json`:
```json
{"5A": {"left": {"embeddings": [[...768 floats...]], "image_paths": ["..."]}, "right": {...}}}
```
Confidence = `(cosine_similarity + 1) / 2 * 100`
