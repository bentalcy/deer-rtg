"""
Test MegaDescriptor-L-384 vs DINOv2 on the 15 user-judged pairs.

MegaDescriptor is a SwinV2 backbone trained on wildlife re-ID datasets —
designed to attend to fine-grained coat patterns, not salient accessories.

Tests both full crop and tight crop (top 25% / bottom 20% removed).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import timm
import torch
import torchvision.transforms as T
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
DECISIONS_FILE = REPO_ROOT / "data/reid/pair_review_decisions.json"

MEGA_MODEL = "hf_hub:BVRA/MegaDescriptor-L-384"
MEGA_SIZE  = 384

CROP_TOP    = 0.25
CROP_BOTTOM = 0.20

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def load_megadescriptor() -> tuple[torch.nn.Module, T.Compose]:
    print(f"Loading MegaDescriptor-L-384 (downloading if needed)...")
    model = timm.create_model(MEGA_MODEL, pretrained=True, num_classes=0)
    model = model.to(DEVICE).eval()
    cfg = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**cfg, is_training=False)
    print(f"  Input size: {cfg['input_size']}  |  Device: {DEVICE}")
    return model, transform


def tight_crop(img: Image.Image) -> Image.Image:
    w, h = img.size
    return img.crop((0, int(h * CROP_TOP), w, int(h * (1.0 - CROP_BOTTOM))))


def embed(model: torch.nn.Module, transform: T.Compose,
          path: str, use_tight: bool = False) -> np.ndarray:
    img = Image.open(REPO_ROOT / path).convert("RGB")
    if use_tight:
        img = tight_crop(img)
    x = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat = model(x).cpu().numpy().astype(np.float32)
    feat = feat / np.linalg.norm(feat, keepdims=True)
    return feat.squeeze()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def evaluate(pairs: list[dict], decisions: dict,
             model: torch.nn.Module, transform: T.Compose,
             use_tight: bool) -> list[dict]:
    judged = [p for p in pairs if decisions.get(p["id"]) in ("same", "different")]
    results = []
    for p in judged:
        emb_a = embed(model, transform, p["a"], use_tight)
        emb_b = embed(model, transform, p["b"], use_tight)
        sim = cosine(emb_a, emb_b)
        user_says = decisions[p["id"]]
        correct_at_85 = (user_says == "same" and sim >= 0.85) or \
                        (user_says == "different" and sim < 0.85)
        results.append({**p, "mega_sim": sim, "user": user_says, "correct": correct_at_85})
    return results


def print_results(results: list[dict], label: str, dino_pairs: list[dict]) -> int:
    dino_by_id = {p["id"]: p["sim"] for p in dino_pairs}
    print(f"\n{'='*72}")
    print(f"  {label}")
    print(f"{'='*72}")
    print(f"{'ID':<14} {'DINOv2':>9} {'MegaDesc':>9} {'delta':>8}  user         verdict")
    print(f"{'-'*72}")
    correct = 0
    for r in results:
        orig = dino_by_id.get(r["id"], 0)
        delta = r["mega_sim"] - orig
        ok = "✓" if r["correct"] else "✗ WRONG"
        if r["correct"]:
            correct += 1
        print(f"{r['id']:<14} {orig:>9.4f} {r['mega_sim']:>9.4f} {delta:>+8.4f}  "
              f"{r['user']:<12} {ok}")
    print(f"{'-'*72}")
    pct = 100 * correct / len(results) if results else 0
    print(f"  Correct: {correct}/{len(results)} ({pct:.0f}%)")
    return correct


def main() -> None:
    if not DECISIONS_FILE.exists():
        print(f"No decisions file at {DECISIONS_FILE}")
        return

    with open(DECISIONS_FILE) as f:
        data = json.load(f)

    decisions = data["decisions"]
    pairs     = data["pairs"]
    judged    = [p for p in pairs if decisions.get(p["id"]) in ("same", "different")]

    model, transform = load_megadescriptor()

    # --- Full crop ---
    print("\nEmbedding with full crop...")
    results_full = evaluate(pairs, decisions, model, transform, use_tight=False)
    correct_full = print_results(results_full, "MegaDescriptor — FULL crop", judged)

    # --- Tight crop ---
    print("\nEmbedding with tight crop (top 25% + bottom 20% removed)...")
    results_tight = evaluate(pairs, decisions, model, transform, use_tight=True)
    correct_tight = print_results(results_tight, "MegaDescriptor — TIGHT crop", judged)

    # --- Summary ---
    print(f"\n{'='*72}")
    print(f"  SUMMARY vs DINOv2 (full crop, 13/15 = 87%)")
    print(f"{'='*72}")
    print(f"  MegaDescriptor full crop  : {correct_full}/{len(judged)}"
          f" ({100*correct_full/len(judged):.0f}%)")
    print(f"  MegaDescriptor tight crop : {correct_tight}/{len(judged)}"
          f" ({100*correct_tight/len(judged):.0f}%)")

    best = max(correct_full, correct_tight)
    if best > 13:
        print(f"\n→ MegaDescriptor IMPROVES on DINOv2. Recommend switching.")
    elif best == 13:
        print(f"\n→ MegaDescriptor matches DINOv2 but doesn't improve. "
              f"Check which pairs differ — may still be preferable for other reasons.")
    else:
        print(f"\n→ MegaDescriptor is WORSE than DINOv2. Escalate — neither model is sufficient.")


if __name__ == "__main__":
    main()
