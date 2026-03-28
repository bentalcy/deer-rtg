# Agent Handoff

> Update this file at the end of every session. Be specific — assume the next agent has zero context.

---

## Status

All 4 enrollment/identification scripts are built and tested (31 passing). First real user session found 3 bugs — fix these before the next enrollment attempt.

| Script | Status |
|--------|--------|
| `scripts/gallery_utils.py` | COMPLETE |
| `scripts/enroll_deer.py` | COMPLETE |
| `scripts/identify_deer.py` | COMPLETE |
| `scripts/enrollment_ui.py` | COMPLETE |

---

## Bugs to Fix (priority order)

### Bug 1 — Name suggestions never show

**Root cause:** Default mode scans `images/` for real ranger photos. `data/reid/clusters.csv` contains flank pool filenames (`food1_1_frame000000__t000__d00.jpg`). Real images have names like `IMG_0011.JPG`. Zero basename overlap → no suggestions ever match.

**Fix:** Change default in `main()` from `images_dir=DEFAULT_IMAGES_DIR` to `embeddings_csv=data/reid/embeddings.csv`. Cluster suggestions work correctly in that mode (basenames match). If user explicitly passes `--images-dir`, accept that no suggestions will show.

### Bug 2 — Side always shows "left"

Two root causes:

1. **`embeddings.csv` has no `side_pred` column** — only `instance_id` and `image_path`. The `build_items` code does `row.get("side_pred")` → always None → always "unknown". Side predictions may be in `data/reid/index.csv` (has `side_pred` column — verify). If so, join on `image_path` when building items.

2. **HTML `<select>` has no "unknown" option.** When `side_prefill` is "unknown", browser defaults to first option ("left"). Immediate fix: add `<option value="unknown">unknown</option>` as the first/default option.

Fix HTML first (unblocks users). Then fix the data join.

### Bug 3 — Side ignores previous user markings

If a user already enrolled an image as "right", reloading the UI should show "right" — not re-read from the CSV.

**Fix:** In `build_items`, after loading `side_prefill` from CSV, check gallery. Gallery format is `{deer_id: {left: {image_paths: [...]}, right: {...}}}` — invert to a `path → side` dict and use it to override.

---

## Resume

```bash
cd /Users/ybental/Documents/code/deer-rtg
source .venv/bin/activate
.venv/bin/python -m pytest -q        # expect 31 passed
.venv/bin/python scripts/enrollment_ui.py --host 127.0.0.1 --port 8765
```

> Always use `.venv/bin/python` — system Python lacks numpy.

---

## Key Decisions (permanent)

- **Embedding model:** MegaDescriptor-L-384 — `timm.create_model("hf_hub:BVRA/MegaDescriptor-L-384", pretrained=True, num_classes=0)`. L2-normalize all outputs.
- **Threshold:** 0.57 (preliminary — re-calibrate once gallery has >5 deer). Lives in `gallery_utils.DEFAULT_SIMILARITY_THRESHOLD`.
- **Left ≠ Right** — never compare across sides. Hard constraint in `rank_matches()`.
- **Gallery format:** `data/gallery/gallery.json` → `{"deer_id": {"left": {"embeddings": [...], "image_paths": [...]}, "right": {...}}}`
- **Confidence:** `(cosine_similarity + 1) / 2 * 100`
- **HTTP server:** stdlib `BaseHTTPRequestHandler`, port 8765, no Flask.
- **`identify_deer.py`** accepts pre-cropped flank images only (no raw-photo detect+crop — deferred).
- **Hosted deployment:** Rangers are remote, no M3 access. **Decision pending — do not implement yet.** Four options evaluated in `docs/deployment_options_for_review.md`; awaiting expert input. Once decided, the chosen option will be recorded in `docs/deployment_s3_lambda_plan.md`.
