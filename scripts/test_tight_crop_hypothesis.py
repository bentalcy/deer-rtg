"""
Test whether tighter vertical crops (removing collar/leg regions) improve
DINOv2 discrimination on the 20 user-reviewed pairs.

Tight crop: remove top 25% (neck/collar) + bottom 20% (legs/injury marks)
keeping the middle torso band where the spot pattern lives.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
DECISIONS_FILE = REPO_ROOT / "data/reid/pair_review_decisions.json"

CROP_TOP    = 0.25   # remove top 25%  (collar/neck)
CROP_BOTTOM = 0.20   # remove bottom 20% (legs/injury)

DINOV2_MODEL = "dinov2_vitb14"
IMG_SIZE = 518
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def load_model() -> torch.nn.Module:
    model = torch.hub.load("facebookresearch/dinov2", DINOV2_MODEL, verbose=False)
    model.eval()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return model.to(device)


def tight_crop(img: Image.Image) -> Image.Image:
    w, h = img.size
    top    = int(h * CROP_TOP)
    bottom = int(h * (1.0 - CROP_BOTTOM))
    return img.crop((0, top, w, bottom))


def embed(model: torch.nn.Module, path: str) -> np.ndarray:
    device = next(model.parameters()).device
    transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])
    img = Image.open(REPO_ROOT / path).convert("RGB")
    img = tight_crop(img)
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(x)          # CLS token
    feat = feat.cpu().numpy().astype(np.float32)
    feat = feat / np.linalg.norm(feat, keepdims=True)
    return feat.squeeze()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def main() -> None:
    if not DECISIONS_FILE.exists():
        print(f"No decisions file found at {DECISIONS_FILE}")
        return

    with open(DECISIONS_FILE) as f:
        data = json.load(f)

    decisions = data["decisions"]
    pairs     = data["pairs"]

    # Only test pairs the user actually judged (not unclear)
    judged = [p for p in pairs if decisions.get(p["id"]) in ("same", "different")]
    print(f"Loading DINOv2 ({DINOV2_MODEL})...")
    model = load_model()

    print(f"\nTight crop: remove top {int(CROP_TOP*100)}% + bottom {int(CROP_BOTTOM*100)}%")
    print(f"Testing {len(judged)} judged pairs\n")
    print(f"{'ID':<14} {'original':>10} {'tight':>10} {'delta':>8}  user_label   model_was   now")
    print("-" * 75)

    improvements = 0
    regressions  = 0
    unchanged    = 0

    for p in judged:
        orig_sim  = p["sim"]
        user_says = decisions[p["id"]]

        emb_a = embed(model, p["a"])
        emb_b = embed(model, p["b"])
        tight_sim = cosine(emb_a, emb_b)
        delta = tight_sim - orig_sim

        # Was the original correct?
        orig_correct = (p["type"] == "high_sim" and user_says == "same") or \
                       (p["type"] == "low_sim"  and user_says == "different")

        # Is the tight crop correct?
        # For same-deer pairs: higher sim = better
        # For different-deer pairs: lower sim = better
        if user_says == "same":
            tight_correct = tight_sim >= 0.85
        else:
            tight_correct = tight_sim < 0.85

        if orig_correct and not tight_correct:
            regressions += 1
            flag = "WORSE ⬇"
        elif not orig_correct and tight_correct:
            improvements += 1
            flag = "FIXED ✓"
        elif orig_correct and tight_correct:
            unchanged += 1
            flag = "still ok"
        else:
            unchanged += 1
            flag = "still wrong"

        orig_was = "correct" if orig_correct else "WRONG"
        print(f"{p['id']:<14} {orig_sim:>10.4f} {tight_sim:>10.4f} {delta:>+8.4f}  "
              f"{user_says:<12} {orig_was:<10} {flag}")

    print("-" * 75)
    print(f"\nResults vs original crops:")
    print(f"  Fixed (model was wrong, now right) : {improvements}")
    print(f"  Regressed (was right, now wrong)   : {regressions}")
    print(f"  Unchanged                           : {unchanged}")

    if improvements > regressions:
        print("\n→ Tight crops HELP. Proceed with tight-crop approach.")
    elif regressions > improvements:
        print("\n→ Tight crops HURT. Collar/leg removal is not the issue, or crops too tight.")
        print("  Consider MegaDescriptor or different crop ratios.")
    else:
        print("\n→ No net change. The shortcut features may be elsewhere in the crop.")
        print("  Consider MegaDescriptor.")


if __name__ == "__main__":
    main()
