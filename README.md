# Deer Re-Identification Pipeline (Hugging Face + macOS M-series)

Goal: identify individual deer by their flank spot patterns. Rangers label trail camera photos in a browser UI; `identify_deer.py` matches new photos against the gallery using MegaDescriptor-L-384 cosine similarity. Runs on macOS M-series (36 GB RAM).

## Project Status

- **Current goal**: Validate and iterate the ranger-focused labeling UI after the one-at-a-time redesign and bug fixes.
- **Resume from**: `AGENT_HANDOFF.md` — read this first, it has exact next steps.
- **Full context**: `docs/PROJECT_GUIDE.md` — written for AI agents, has full spec and history.

### Current state
- Enrollment stack is in place: `scripts/gallery_utils.py`, `scripts/enroll_deer.py`, `scripts/identify_deer.py`, `scripts/enrollment_ui.py`.
- UI now defaults to one-at-a-time labeling (not a grid landing page).
- Progress bar, bottom-anchored actions, visual side selector, auto-advance, and toast-based undo are implemented.
- Deer naming now uses picker options from gallery IDs with `+ New deer` fallback.
- Default UI data source now uses `data/reid/embeddings.csv` (instead of `images/`) to restore cluster-name overlap.
- Side prefill now joins `data/reid/index.csv` (`side_pred`) and is overridden by previously saved gallery labels.
- Python API contracts remain unchanged: `GET /items`, `POST /enroll`, `POST /bulk-enroll`.
- Test baseline is green: `31 passed`.

### TODO
- [ ] Run manual browser smoke test of the redesigned UI flow on desktop + iPhone viewport.
- [ ] Verify undo behavior under quick repeated saves and reload conditions.
- [ ] Capture any wording/layout adjustments from ranger feedback before deployment work starts.

### How to resume
- `uv venv .venv && source .venv/bin/activate`
- `.venv/bin/python -m pytest -q`
- `.venv/bin/python scripts/enrollment_ui.py --host 127.0.0.1 --port 8765`
- Next file to inspect for follow-up tweaks: `scripts/enrollment_ui.py`

### Notes / decisions
- Deployment work remains blocked on NGO budget approval; do not start Task 3 yet.
- Undo is implemented as a 5-second delayed commit window (toast cancel) to avoid backend API changes.

### Why we changed direction
The original clustering → ArcFace plan produced no usable identification tool after extensive work. The 253 images come from only 6 video encounters (near-duplicate frames), clustering latched onto wrong features (collars, injury stripes), and no ground-truth deer labels were ever created. The enrollment-based approach works today with existing embeddings and ~30 minutes of human labeling. See `docs/PROJECT_GUIDE.md` for full history.

