# Deer Re-ID Bootstrapping + Incremental Identification (Expert Review)

## Goal

- Today: ~1000 unlabeled full-frame images, ~50 individual deer.
- Deliverable (milestone 1): for each image, output the set of deer IDs that appear (presence per frame).
- Next: for each new batch (100-500 images, mostly the same deer), automatically assign known IDs and flag unknown/ambiguous cases for quick review.
- Later upgrade: per-instance IDs within a frame (which deer is which) if needed.

## Data & Constraints

- Full frames; sometimes multiple deer per image.
- Mixed capture conditions: fixed trail-cams plus multiple cameras/locations; backgrounds vary.
- Metadata available per image (camera/time/etc.); useful for evaluation hygiene and bias diagnosis.
- Identity definition: single deer ID across left/right sides; viewpoint can be modeled as nuisance.

## Core Approach

- Do not attempt "recognize deer" directly from full frames.
- Pipeline: detect -> crop -> quality-gate -> embed -> cluster (bootstrap) -> tiny human cleanup -> train metric re-ID -> deploy on new batches with open-set gating.

## Plan (Phased)

1) Dataset inventory + evaluation splits (metadata-first)

- Build an index: image -> metadata.
- Define splits to prevent leakage and expose camera/background shortcuts (report metrics per camera/time slice).

2) Instance isolation (required because multiple deer can appear)

- Run deer detection on each frame; each detection becomes an instance crop linked back to its source frame.
- Optional if temporal adjacency exists: tracklets to reduce duplicates and enable multi-shot aggregation.

3) Crop tightening + quality gating ("usable flank")

- Tighten crops to minimize background signal; consider segmentation refinement if background bias persists.
- Train/use a binary gate to keep only identity-informative flank/side crops for embedding/clustering.

4) Embedding extraction (self-supervised similarity vectors)

- Compute an embedding per kept crop (prefer DINOv2/CLIP over vanilla ResNet as the main path; ResNet is acceptable only as a baseline sanity check).
- Normalize embeddings for cosine similarity.
- If tracklets exist: optionally aggregate embeddings per tracklet for stability.

5) Bootstrapping via clustering (~50 groups) + confidence

- Cluster embeddings to form proto-identities:
  - KMeans(k~=50) if we want a full partition and accept forced assignments.
  - HDBSCAN if we want explicit outliers/noise instead of forcing every crop into a cluster.
- Compute cluster representatives (medoids) and confidence (distance-to-center / density score / top-2 margin).

6) Tiny human cleanup loop to create deer IDs

- Review only:
  - cluster medoids (quick sanity),
  - low-confidence items (boundary cases),
  - outliers/noise.
- Actions: merge clusters, split clusters, reassign instances.
- Output: mapping instance_crop -> deer_id (clean enough to train a model).

7) Milestone 1 output: per-image deer presence

- For each frame: union predicted deer_ids across its detected instances.
- Output includes confidence and a review flag, but avoids within-frame assignment complexity.

8) Train metric re-ID model for ongoing batches + open-set gating

- Train a metric-learning re-ID model (ArcFace/CosFace-style) on cleaned labels.
- Deploy on new batches:
  - retrieve top-k known IDs,
  - auto-assign when confident,
  - route ambiguous cases to review,
  - handle unknown/new IDs via an open-set threshold or novelty detection.
- Periodically incorporate reviewed labels and recalibrate thresholds.

## Success Metrics

- Bootstrap: cluster purity on a small verified subset + human minutes to finalize IDs.
- New batches: top-1/top-5 accuracy on held-out labeled data + false-known rate at a fixed review budget; breakdown by camera/time.

## Open Questions for Expert Review

- Embeddings: DINOv2 vs CLIP vs wildlife-specific re-ID backbones for deer flank patterns under multi-camera domain shift - what is most reliable in practice?
- Clustering: KMeans(k=50) vs HDBSCAN - how to choose and how to set confidence/review thresholds to minimize human time without accumulating identity drift?
- Tracklets: how much does multi-shot aggregation improve clustering/retrieval here, and what aggregation works best (mean, attention pooling, voting)?
- Open-set gating: simple cosine thresholding vs calibrated novelty detection - recommended approach for "mostly same deer" new batches while still catching true new individuals?
- Background/camera bias: best practical mitigation (crop tightening, segmentation, camera-conditioned normalization, domain adaptation, stratified evaluation)?
- Metadata usage: which constraints are safe and useful (time/location priors) without cheating evaluation or baking in brittle heuristics?
- Viewpoint/side: best way to handle left/right and pose variation while keeping a single deer ID target (auxiliary tasks, sampling strategies, separate galleries, etc.)?
