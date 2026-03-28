# Deployment Plan — AWS S3 + Lambda (Python)

**Decision date:** 2026-03-28
**Status:** Approved, budget pending NGO sign-off (expected 2–5 days). Do not implement until budget is confirmed.

---

## Why this was chosen

Four options were evaluated on raw dollar cost and total practical cost (engineering + maintenance included). Full comparison in `docs/deployment_options_for_review.md`.

- **Raw hosting: ~$1/year.** No-use-no-cost — idle months cost ~$0.07 (S3 storage only).
- **Python throughout.** No rewrite required. Best-documented serverless platform — lowest agent error rate.
- **Zero ongoing maintenance.** No server to patch or monitor.

Hetzner VPS was rejected: €39/year for a workload costing $1/year is not cost-optimized. Oracle Free was rejected: $0 raw cost but Oracle's weaker documentation makes agent mistakes more likely and token cost higher — savings are eaten by that risk.

---

## Architecture

```
Rangers (browser)
    │
    ▼
Lambda Function URL  ←── shared secret header auth
    │
    ├── GET /           → serves index.html (static UI)
    ├── GET /items      → reads labels.json + image manifest from S3, returns JSON
    ├── GET /image?path → returns presigned S3 URL for the thumbnail
    ├── POST /enroll    → reads labels.json, updates one entry, writes back to S3
    └── POST /bulk-enroll → reads labels.json, updates N entries, writes back to S3

S3 Bucket (versioning enabled)
    ├── images/          ← thumbnails only (300KB each, ~200MB total)
    ├── manifest.json    ← list of all image paths + side_pred hints
    └── labels.json      ← current label state (image_path → deer_id + side)

M3 Mac (local only, never in Lambda)
    ├── images/          ← originals (3–8MB each)
    ├── data/gallery/gallery.json  ← compiled from labels, contains embeddings
    └── scripts/compile_gallery_from_labels.py  ← downloads labels, runs MegaDescriptor, builds gallery.json
```

**Hard constraint:** Lambda never imports torch, timm, or numpy. No ML inference in the cloud path — ever.

---

## Data formats

### Per-image label files (hosted in S3 under `labels/` prefix)

One S3 object per labeled image. Key: image path with `/` replaced by `_`, e.g.:
- `images/FD/IMG_0011.JPG` → `labels/images_FD_IMG_0011.JPG.json`

```json
{"deer_id": "5A", "side": "left", "updated_at": "2026-03-28T12:00:00Z"}
```

Each POST writes exactly one file (atomic S3 PUT). `GET /items` lists `labels/` prefix and reads all files. No read-modify-write on a shared file — see state issue note in AWS services section.

### manifest.json (hosted in S3, built locally and uploaded once)
```json
[
  {"image_path": "images/FD/IMG_0011.JPG", "side_hint": "left"},
  {"image_path": "images/FD/IMG_0015.JPG", "side_hint": "unknown"}
]
```
`side_hint` comes from `data/reid/index.csv` (`side_pred` column) matched by filename. Falls back to `"unknown"`.

---

## AWS services needed

| Service | Purpose | Cost |
|---------|---------|------|
| S3 (one bucket) | Images, labels, manifest | ~$0.005/month storage |
| Lambda (one function) | API + HTML serving | ~$0.00 at this scale |
| Lambda Function URL | HTTPS endpoint, no API Gateway needed | Free |
| S3 versioning | Rollback on bad label writes | Free (slightly more storage) |

No API Gateway. No CloudFront. No DynamoDB. No VPC.

**Lambda config:** Python 3.12 runtime, 256MB RAM, 30s timeout, **reserved concurrency = 1** (prevents concurrent label writes — safe alternative to distributed locking).

### ⚠ Known state issue — fix before implementing

The plan stores all labels in a single `labels.json` file. Every POST does: GET file → update in memory → PUT file back. This is a read-modify-write pattern with one known failure mode: **if Lambda is killed or times out between the GET and PUT, the in-progress write is lost silently.** S3 versioning lets you recover manually but does not prevent data loss.

**Fix: use per-image label files instead of a monolithic JSON.**

Store each label as its own S3 object:
```
labels/images_FD_IMG_0011.JPG.json   ← one file per image
labels/images_FD_IMG_0015.JPG.json
...
```

Each file contains a single label entry:
```json
{"deer_id": "5A", "side": "left", "updated_at": "2026-03-28T12:00:00Z"}
```

Each POST writes exactly one file — a single S3 PUT, which is atomic. No read-modify-write. A failed write affects only that one image, not the entire label store. Recovery is trivial.

`GET /items` aggregates by listing `labels/` prefix and reading all files. At 660 images this is fast (~0.5s). If it becomes slow at larger scale, add a background job that consolidates into a summary JSON — but do not prematurely optimise.

**Update the key naming:** use the image path with `/` replaced by `_` as the S3 key (e.g. `images/FD/IMG_0011.JPG` → `labels/images_FD_IMG_0011.JPG.json`). Keep a consistent, reversible mapping.

This fix applies to both the cloud Lambda implementation and any local-disk equivalent. Do not implement the monolithic `labels.json` approach.

---

## Code changes required

### 1. New script: `scripts/resize_images_for_upload.py` (local, run once)

Resize all images in `images/` to max 1200px long edge, save to `images_web/`. Upload `images_web/` to S3 `images/` prefix. Originals stay in `images/` on M3 for embedding generation.

```
python scripts/resize_images_for_upload.py --input images/ --output images_web/
aws s3 sync images_web/ s3://BUCKET_NAME/images/
```

### 2. New script: `scripts/build_manifest.py` (local, run once then update when images change)

Reads `images_web/` directory, joins with `data/reid/index.csv` on filename to get `side_pred` hints, writes `manifest.json`, uploads to S3.

```
python scripts/build_manifest.py --images-dir images_web/ --out manifest.json
aws s3 cp manifest.json s3://BUCKET_NAME/manifest.json
```

### 3. New Lambda handler: `lambda/handler.py`

Pure Python, no ML dependencies. Replaces the HTTP server logic from `enrollment_ui.py`. Handles all 5 endpoints listed in the architecture above. Uses `boto3` (pre-installed in Lambda runtime) for S3 reads/writes.

Auth: checks `X-Enroll-Secret` header against `ENROLL_SECRET` env var on all POST endpoints. GET endpoints for images and items are unauthenticated (images are served via short-lived presigned URLs, not directly exposed).

Read the existing `enrollment_ui.py` handler logic as the reference — the business logic (`build_items`, `handle_enroll_payload`, `handle_bulk_enroll_payload`) can be reused nearly verbatim. Only the storage layer changes (S3 instead of local disk).

Reuse the pure functions from `scripts/enrollment_ui.py`:
- `build_items()` — adapt to read from manifest.json + labels.json instead of embeddings.csv
- `handle_enroll_payload()` — same logic, swap `save_gallery()` for S3 PUT
- `handle_bulk_enroll_payload()` — same logic

### 4. New script: `scripts/compile_gallery_from_labels.py` (local M3 only, never deployed)

Downloads `labels.json` from S3, loads each labeled image from local `images/`, runs MegaDescriptor embedding, writes `data/gallery/gallery.json`. This is the bridge between the hosted label store and the local identification tool.

```
python scripts/compile_gallery_from_labels.py \
  --bucket BUCKET_NAME \
  --labels-key labels/labels.json \
  --images-dir images/ \
  --out data/gallery/gallery.json
```

---

## Deployment steps

### Phase 1 — working hosted labeling

1. Create S3 bucket, enable versioning, note bucket name
2. Run `resize_images_for_upload.py`, sync thumbnails to S3
3. Run `build_manifest.py`, upload manifest.json to S3
4. Initialize `labels.json` (empty: `{"version": 1, "items": {}}`) and upload to S3
5. Write `lambda/handler.py` — test locally with mocked S3 using `moto`
6. Package and deploy Lambda (`zip` + `aws lambda create-function`)
7. Create Lambda Function URL, note the HTTPS endpoint
8. Set `ENROLL_SECRET` env var on Lambda
9. Test: open Function URL in browser, enroll one image, verify labels.json in S3 updated

### Phase 2 — local compile workflow

1. Write `compile_gallery_from_labels.py`
2. Run it, verify `data/gallery/gallery.json` is produced correctly
3. Test `identify_deer.py` still works with the compiled gallery

### Phase 3 — hardening (optional, do after Phase 1 works)

1. Set S3 bucket policy to block public access (images served only via presigned URLs)
2. Add basic rate limiting in Lambda (reject if >100 requests/minute — protects against accidental loops)
3. Add CloudWatch alarm on Lambda error rate (free tier covers this)

---

## What to tell the implementing agent

- Read `scripts/enrollment_ui.py` in full before writing `lambda/handler.py` — the business logic is already correct, only the storage layer changes
- Read `scripts/gallery_utils.py` for the labels schema and `load_gallery`/`save_gallery` patterns
- Run `pytest -q` before and after any changes to enrollment_ui.py
- Do not add Flask, FastAPI, or any web framework to the Lambda — plain Python + boto3 only
- Lambda Function URLs handle HTTPS automatically — no need for API Gateway or ACM certificates
- The `already_enrolled` flag in the UI should be driven by `labels.json` items, not `gallery.json` (gallery.json is local only)
