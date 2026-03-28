# Deployment Plan - S3 + Lambda, Single User, Local Inference

## ⚠ Decision Pending (2026-03-28) — Do not implement anything yet

A full options review is in `docs/deployment_options_for_review.md`. The user is consulting an external expert. Wait for the decision to be recorded here before writing any deployment code.

Options under consideration: AWS S3+Lambda (Python), Hetzner VPS, Oracle Always Free, Cloudflare Workers. Each has meaningful trade-offs around cost model, Python vs JS, and agent reliability. See the review doc for the full comparison.

---

## ⚠ Original S3 + Lambda plan is also superseded

**The S3 + Lambda plan below is the original design. It has been reviewed and replaced. Do not implement it.**

### What changed and why

Cloud hosting is confirmed needed: rangers label remotely on their own devices with no access to the M3 Mac.

The S3 + Lambda plan was evaluated against three alternatives (reviewed with Claude Opus):

| Option | Problem |
|--------|---------|
| AWS S3 + Lambda (Python) | Image egress ~$0.09/GB — exceeds VPS cost once rangers browse images |
| Cloudflare Pages + R2 + Workers | All backend must be rewritten in JavaScript. Non-starter for a Python codebase. KV eventual consistency requires Durable Objects (+$5/month) to fix. Less ecosystem support = more agent errors = higher token cost. |
| Backblaze B2 + Lambda | Two vendors for marginal savings; more setup complexity |
| **Hetzner VPS (€3.29/month)** | **✓ Recommended** |

### Chosen direction: Hetzner VPS

**Why:**
- The existing Python stdlib server (`enrollment_ui.py`) deploys nearly as-is — no architecture rewrite, no JS, no split storage
- AI coding agents are most reliable targeting "a Linux box running Python" — best-documented deployment target, fewest agent mistakes
- At ~660 images and one active user, a small ARM VPS handles this trivially
- Total cost: ~€3.29/month (~$3.60), predictable, no egress surprises

**Stack:**
- Hetzner CAX11 (2 vCPU ARM, 4GB RAM, 40GB disk) — €3.29/month
- nginx reverse proxy + Let's Encrypt (HTTPS)
- `enrollment_ui.py` running as a systemd service
- Images and `labels.json` on local disk
- Daily rsync backup to the M3 Mac via cron

**Risk and mitigations:**
- Server can go down → enable unattended-upgrades; set up UptimeRobot free tier health check
- Disk data loss → daily cron rsync to M3 Mac
- No version history on labels → add `git init` to the labels directory, auto-commit on each write

### Image thumbnails (required before deploy)

Trail camera photos are 3–8MB each. Serving full resolution in a browser grid will be painfully slow over the internet. Before deploying, add a local pre-processing step:

```bash
# resize to max 1200px long edge, ~200–400KB each
python scripts/resize_images_for_upload.py --input images/ --output images_web/
```

The UI serves `images_web/` (display copies). Originals stay in `images/` on the M3 for embedding generation.

### Required code changes to `enrollment_ui.py`

1. Add `--images-dir` arg support (already done — points at `images_web/` on the server)
2. Add a shared-secret auth header check in the Handler (one env var `ENROLL_SECRET`)
3. Switch `labels.json` to an append-safe format (per-image files under `data/labels/` to avoid full-file corruption, or add git auto-commit after each write)
4. Serve over a Unix socket so nginx proxies cleanly

### What to implement (phases)

**Phase 1 — working hosted labeling:**
1. Provision Hetzner CAX11, install nginx + Python 3.11, configure systemd service
2. Run `resize_images_for_upload.py` locally, rsync `images_web/` to server
3. Add shared-secret auth to `enrollment_ui.py`
4. Deploy, test with one ranger

**Phase 2 — hardening:**
1. Add git auto-commit to labels after each write
2. Set up daily rsync cron back to M3 Mac
3. UptimeRobot health check

---

## Original S3 + Lambda Plan (kept for reference only)

## 1) Summary Of The Discussion And Relevant Context

### Decision made
We are not hosting model inference in the cloud.

The hosted system should do only:
- show images in a browser UI
- let the user assign a deer ID like `5A`
- save that label centrally
- keep costs extremely low

Model-related work should stay local on the M3 machine:
- embedding generation
- gallery compilation
- identification runs

### Why this changes the deployment shape
The current app is built as a local Python HTTP server that both:
- serves the UI
- reads local files
- writes `gallery.json`
- computes embeddings during enrollment

That is not a good fit for ultra-low-cost hosting because:
- it is stateful
- it assumes local disk
- it mixes UI, storage, and inference in one process

### Relevant current code
- `scripts/enrollment_ui.py`
  - stdlib `BaseHTTPRequestHandler`
  - serves HTML and API from one long-running Python process
  - reads images from local disk
  - reads `data/reid/embeddings.csv`
  - writes enrollment state through local functions
- `scripts/enroll_deer.py`
  - `enroll_image(...)` currently computes MegaDescriptor embeddings and updates `gallery.json`
- `scripts/gallery_utils.py`
  - `load_gallery()` / `save_gallery()` assume local file paths
  - gallery schema currently stores embeddings and image paths together

### Deployment choice for next agents
Chosen direction:
- AWS S3 for hosted image storage and JSON files
- AWS Lambda for the small write/read API
- single user
- no DynamoDB
- local inference only
- coarse serialization is acceptable
- S3 versioning should be enabled

### Important architectural consequence
Because inference stays local, the hosted backend should not update embedding-bearing `gallery.json` directly during labeling.

Instead, separate the data into:

1. Hosted labels store
   - user assignments only
   - image key/path -> deer ID + side + timestamps + notes if needed

2. Local compiled gallery
   - built later on the M3 machine
   - contains embeddings and image paths
   - used by `identify_deer.py`

This keeps cloud hosting cheap and keeps Torch/timm off Lambda.

## 2) What Should Change In The Code

### Current problem to remove
We need to move away from a single stateful Python web server in `scripts/enrollment_ui.py`.

The hosted version should become:
- a static frontend
- a small stateless JSON API
- S3-backed storage
- no model loading in the web path

### Recommended target architecture

#### Frontend
Host a static UI in S3:
- HTML
- CSS
- JS

The browser should call Lambda endpoints like:
- `GET /items`
- `POST /enroll`
- `POST /bulk-enroll`

#### Backend
Use Lambda for API behavior only:
- read image index / labels JSON from S3
- generate presigned image URLs if images stay private
- validate label payloads
- write updated labels JSON back to S3

#### Storage
Use one S3 bucket or two buckets.

Simplest split:
- `images/` - raw images for review
- `manifests/` - image index CSV/JSON
- `labels/labels.json` - current hosted label state
- `labels/history/` - optional snapshots
- `compiled/gallery.json` - local machine may upload compiled gallery snapshots if useful

Enable:
- bucket versioning
- server-side encryption
- least-privilege IAM for Lambda

### Recommended code refactor

#### A. Split labeling state from embedding gallery
Current issue:
- `scripts/enroll_deer.py` writes embeddings into `gallery.json` immediately

Needed change:
- create a new hosted label format that stores only labeling decisions

Suggested JSON shape:

```json
{
  "version": 1,
  "updated_at": "2026-03-28T12:00:00Z",
  "items": {
    "images/FD/foo.jpg": {
      "deer_id": "5A",
      "side": "left",
      "updated_at": "2026-03-28T12:00:00Z"
    }
  }
}
```

This becomes the cloud source of truth for the web UI.

#### B. Keep `gallery.json` as a local compiled artifact
Current issue:
- `gallery.json` is both label store and embedding store

Needed change:
- make `gallery.json` local/offline only
- generate it from:
  - hosted labels JSON
  - S3 image set
  - local MegaDescriptor embedding pass on the M3 machine

This means:
- `identify_deer.py` still uses compiled gallery data locally
- hosted UI does not need Torch, timm, or image embedding code

#### C. Replace `BaseHTTPRequestHandler` app with a static frontend
Current issue:
- `scripts/enrollment_ui.py` is a long-running Python web server

Needed change:
- extract the HTML/JS into static assets
- remove server-specific behavior from the frontend
- frontend should fetch JSON from Lambda instead of same-process handler methods

Practical refactor:
- move current inline `INDEX_HTML` into static files, for example:
  - `web/index.html`
  - `web/app.js`
  - `web/styles.css`

#### D. Introduce pure service functions for hosted labeling
Refactor toward pure functions that can run in Lambda.

Suggested service functions:
- `list_items(index_data, labels_data) -> list[dict]`
- `apply_enrollment(labels_data, payload) -> labels_data`
- `apply_bulk_enrollment(labels_data, payload) -> labels_data`

These should:
- not read local disk directly
- not depend on HTTP classes
- not import Torch/timm
- not assume `Path(...)` points to local files

#### E. Add an S3 storage layer
Current issue:
- `load_gallery()` / `save_gallery()` only handle local filesystem paths

Needed change:
- add storage helpers for JSON in S3, for example:
  - `load_labels_s3(bucket, key)`
  - `save_labels_s3(bucket, key, data)`
  - `load_manifest_s3(bucket, key)`

This should be separate from domain logic.

#### F. Add a local sync/compile script
Add a local-only script that runs on the M3 machine.

Suggested responsibilities:
- download `labels/labels.json` from S3
- read images from S3 or local synced folder
- compute embeddings for labeled images
- produce compiled `data/gallery/gallery.json`
- optionally upload compiled snapshot to S3 for backup

Suggested file:
- `scripts/compile_gallery_from_labels.py`

#### G. Single-user write serialization
Since the deployment is intentionally single-user, the cheapest safe approach is:
- use one write Lambda
- set reserved concurrency to 1
- all write operations go through that Lambda
- enable S3 versioning

This avoids adding DynamoDB-based locking.

Optional extra safety:
- reject overlapping writes if a short-lived lock object exists
- but reserved concurrency 1 is probably enough here

#### H. Optional simple auth
For a single-user NGO setup, a full auth system may be overkill initially.

Minimum acceptable first version:
- Lambda expects a shared secret header
- static frontend prompts once and stores it in memory or local storage
- images remain private in S3; Lambda returns presigned URLs

If image sensitivity is high, upgrade later to a stronger auth layer.

### Suggested hosted request flow

#### GET /items
Lambda:
1. reads image manifest from S3
2. reads `labels/labels.json` from S3
3. returns:
   - image key
   - current deer ID if labeled
   - current side if labeled
   - already_enrolled boolean
   - optional presigned image URL

#### POST /enroll
Lambda:
1. validates payload
2. reads current labels JSON from S3
3. updates one image record
4. writes updated JSON back to S3
5. returns success payload

#### POST /bulk-enroll
Lambda:
1. validates rows
2. reads current labels JSON
3. applies multiple updates
4. writes JSON back once
5. returns counts

### Suggested migration path

#### Phase 1 - cheapest useful hosted labeling
- upload images to S3
- create image manifest in S3
- create static frontend
- create Lambda API for `GET /items`, `POST /enroll`, `POST /bulk-enroll`
- store labels in `labels/labels.json`

#### Phase 2 - local compile workflow
- add local script to download labels and build compiled `gallery.json`
- keep `identify_deer.py` fully local

#### Phase 3 - optional hardening
- auth improvements
- audit trail / snapshots
- better labeling metadata
- domain name / TLS polish

## 3) Rough Cost Estimate - 1 User, Very Low Utilization, 1 Month

### Assumptions
- one user only
- very low usage
- labeling only, no cloud inference
- current image set roughly hundreds of images, not tens of thousands
- no CloudFront, no EC2, no DynamoDB
- no custom domain cost included
- static site hosted in S3
- images stored in S3
- Lambda only handles small JSON requests

### Cost drivers
Main cost buckets:
- S3 storage
- S3 requests
- data transfer out for viewed images
- Lambda invocations

### Ballpark monthly estimate

#### S3 storage
For a few GB of images plus tiny JSON/HTML:
- likely about $0.05 to $0.25

#### S3 requests
For low browsing and a small number of writes:
- likely well under $0.10

#### Lambda
For a small API with very few requests:
- likely $0.00 to $0.05
- often effectively zero at this scale

#### Data transfer out
This is the most variable part.
For one user browsing a modest number of images:
- likely $0.10 to $1.50

### Expected total

#### Realistic low-usage estimate
- about $0.25 to $2.00 for the month

#### Conservative safe estimate
- under $3.00 for the month

### Things not included
- custom domain registration
- Route 53 hosted zone
- CloudFront
- stronger auth products
- any accidental heavy download session of full-resolution images

## Final Recommendation

For this project and budget, the best next step is:
- keep inference local on the M3
- host only labeling data and images
- use S3 + Lambda
- store labels only in the hosted JSON
- compile embeddings locally into `gallery.json`
- enforce single-writer behavior with Lambda reserved concurrency = 1
- enable S3 versioning for rollback and recovery

This is the cheapest reasonable hosted setup that still gives a real shared labeling workflow.

## Notes For The Next Agent

Do not deploy the current `scripts/enrollment_ui.py` as-is.

The main required change is conceptual:
- the hosted app should save label assignments
- the local M3 workflow should build embedding gallery artifacts

Do not put MegaDescriptor inference into Lambda for this phase.
That would increase complexity, package size, cold start risk, and cost for no real benefit in the single-user setup.
