# Agent Handoff

> Update this file at the end of every session. Assume the next agent has zero context.

---

## START HERE

```bash
cd /Users/ybental/Documents/code/deer-rtg
source .venv/bin/activate
.venv/bin/python -m pytest -q        # must show 31 passed before you touch anything
```

> Always use `.venv/bin/python` — system Python lacks numpy.

**Your task right now:** Manual smoke test of the UI (see verification checklist below). The deployment task is blocked on NGO budget — do not start it yet.

---

## Current State (2026-03-28)

All planned work is complete. Tests: `31 passed`.

### What was built

**UI (INDEX_HTML in `scripts/enrollment_ui.py`):**
- One-at-a-time card flow is the default landing experience (no grid on load)
- Progress header: `X labeled | Photo Y of Z` — labeled count + navigation position
- Deer name picker: dropdown of gallery IDs + quick buttons (up to 8) + `+ New deer` input
- Visual side buttons: `◀ 🦌 Left`, `🦌 ▶ Right`, `? Can't tell`
- `Save & Next` auto-advances; immediate write (no delayed commit)
- Label overwrite via `POST /label-upsert`; label delete via `POST /label-delete`
- "I don't know this deer" option stored as `__UNKNOWN__`
- Close-deer reference cards: 2–4 already-labeled examples shown alongside current photo
- `Hide Photo` removes current image from queue; persists in `localStorage` (`enrollment_ui_hidden_images`)
- `Show Hidden` restores all hidden photos
- Optional **Review labeled photos** grid mode (not the landing view)
- All jargon removed from user-visible text

**Python logic (`scripts/enrollment_ui.py`):**
- Default source auto-selects: `images/` if it contains images, else `data/reid/embeddings.csv`
- `--images-dir` and `--embeddings-csv` flags override auto-select
- `build_items()` joins side hints from `data/reid/index.csv` (`side_pred`)
- Side prefill precedence: (1) prior gallery label, (2) model/index suggestion, (3) unknown
- Cluster name suggestions prefer real deer name when cluster has prior gallery examples
- Similarity-based close matches (top 2–4 deer) when `embeddings.pt` is available

### Verification checklist (run this next)

```bash
.venv/bin/python scripts/enrollment_ui.py --host 127.0.0.1 --port 8765
```

- [ ] One-at-a-time flow on load (no grid on first screen)
- [ ] `Photo Y` changes on Next / Previous
- [ ] `X labeled` increments only on save, not on navigation
- [ ] Side prefill respects prior gallery labels after page reload
- [ ] `Review labeled photos` toggles grid mode
- [ ] `Hide Photo` removes image from queue; survives page reload
- [ ] `Show Hidden` restores hidden photos
- [ ] Name suggestions appear (cluster overlap via embeddings mode)

---

## Task 3 — Deployment (BLOCKED — do not start)

Waiting on NGO budget approval (expected 2–5 days from 2026-03-28). Full plan in `docs/deployment_s3_lambda_plan.md`.

**When budget is confirmed:** AWS S3 + Lambda (Python), ~$1/year. Four scripts to write:

| Script | Purpose |
|--------|---------|
| `scripts/resize_images_for_upload.py` | Resize originals to thumbnails, upload to S3 |
| `scripts/build_manifest.py` | Build image index with side hints, upload to S3 |
| `lambda/handler.py` | Lambda API — reuses `enrollment_ui.py` logic, swaps disk for S3 |
| `scripts/compile_gallery_from_labels.py` | Local M3 only — downloads labels, runs MegaDescriptor, builds `gallery.json` |

**State issue to fix in the Lambda handler:** use per-image label files (`labels/images_FD_IMG_0011.JPG.json`), not a monolithic `labels.json`. Each POST is one atomic S3 PUT — no read-modify-write, no data loss risk. Details in `docs/deployment_s3_lambda_plan.md`.

---

## Key Decisions (permanent)

- **Embedding model:** MegaDescriptor-L-384 — `timm.create_model("hf_hub:BVRA/MegaDescriptor-L-384", pretrained=True, num_classes=0)`. L2-normalize all outputs.
- **Threshold:** 0.57 (preliminary — re-calibrate once gallery has >5 deer). In `gallery_utils.DEFAULT_SIMILARITY_THRESHOLD`.
- **Left ≠ Right** — never compare across sides. Hard constraint in `rank_matches()`.
- **Gallery format:** `data/gallery/gallery.json` → `{"deer_id": {"left": {"embeddings": [...], "image_paths": [...]}, "right": {...}}}`
- **Confidence:** `(cosine_similarity + 1) / 2 * 100`
- **HTTP server:** stdlib `BaseHTTPRequestHandler`, port 8765, no Flask.
- **`identify_deer.py`** accepts pre-cropped flank images only (detect+crop path deferred).
- **Hosted deployment:** AWS S3 + Lambda (Python), labels-only in cloud, no embeddings. Blocked on budget.
