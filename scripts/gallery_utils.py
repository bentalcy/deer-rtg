from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import timm
import torch
import torchvision.transforms as T
from PIL import Image


DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MEGA_MODEL = "hf_hub:BVRA/MegaDescriptor-L-384"
DEFAULT_SIMILARITY_THRESHOLD = 0.57


def load_gallery(
    path: Path = Path("data/gallery/gallery.json"),
) -> dict[str, dict[str, dict[str, list[Any]]]]:
    if not path.exists():
        return {}

    with path.open() as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        return {}

    gallery: dict[str, dict[str, dict[str, list[Any]]]] = {}
    for deer_id, deer_data in raw.items():
        if not isinstance(deer_data, dict):
            continue
        left = deer_data.get("left", {})
        right = deer_data.get("right", {})
        gallery[str(deer_id)] = {
            "left": {
                "embeddings": list(left.get("embeddings", []))
                if isinstance(left, dict)
                else [],
                "image_paths": list(left.get("image_paths", []))
                if isinstance(left, dict)
                else [],
            },
            "right": {
                "embeddings": list(right.get("embeddings", []))
                if isinstance(right, dict)
                else [],
                "image_paths": list(right.get("image_paths", []))
                if isinstance(right, dict)
                else [],
            },
        }
    return gallery


def save_gallery(
    gallery: dict[str, dict[str, dict[str, list[Any]]]],
    path: Path = Path("data/gallery/gallery.json"),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(gallery, f, indent=2)


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(x))
    if norm == 0.0:
        raise ValueError("Cannot normalize zero vector")
    return x / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    va = np.asarray(a, dtype=np.float32).reshape(-1)
    vb = np.asarray(b, dtype=np.float32).reshape(-1)
    if va.shape != vb.shape:
        raise ValueError(
            f"Shape mismatch for cosine_similarity: {va.shape} vs {vb.shape}"
        )
    return float(np.dot(va, vb))


def compute_prototype(embeddings: list[list[float]]) -> np.ndarray:
    if not embeddings:
        raise ValueError("Cannot compute prototype from empty embeddings list")
    arr = np.asarray(embeddings, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Embeddings must be a 2D array, got shape {arr.shape}")
    return _l2_normalize(arr.mean(axis=0))


def rank_matches(
    query_emb: np.ndarray,
    gallery: dict[str, dict[str, dict[str, list[Any]]]],
    query_side: str,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    if not gallery:
        return [{"deer_id": "UNKNOWN", "side": "", "confidence": 0.0}]

    query_side_norm = query_side.strip().lower()
    if query_side_norm in {"left", "right"}:
        sides = [query_side_norm]
    elif query_side_norm in {"unknown", ""}:
        sides = ["left", "right"]
    else:
        raise ValueError("query_side must be one of: left, right, unknown")

    q = _l2_normalize(np.asarray(query_emb, dtype=np.float32).reshape(-1))
    rows: list[dict[str, Any]] = []

    for deer_id, deer_data in gallery.items():
        for side in sides:
            side_data = deer_data.get(side, {})
            emb_list = (
                side_data.get("embeddings", []) if isinstance(side_data, dict) else []
            )
            if not emb_list:
                continue

            proto = compute_prototype(emb_list)
            sim = cosine_similarity(q, proto)
            if sim < DEFAULT_SIMILARITY_THRESHOLD:
                continue
            confidence = ((sim + 1.0) / 2.0) * 100.0
            rows.append(
                {
                    "deer_id": deer_id,
                    "side": side,
                    "confidence": float(np.clip(confidence, 0.0, 100.0)),
                }
            )

    if not rows:
        return [{"deer_id": "UNKNOWN", "side": "", "confidence": 0.0}]

    rows.sort(key=lambda x: x["confidence"], reverse=True)
    return rows[: max(top_k, 0)]


def load_megadescriptor() -> tuple[torch.nn.Module, T.Compose]:
    model = timm.create_model(MEGA_MODEL, pretrained=True, num_classes=0)
    model = model.to(DEVICE).eval()
    cfg = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**cfg, is_training=False)
    return model, transform


def embed_single(
    model: torch.nn.Module, transform: T.Compose, path: Path
) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat = model(x).cpu().numpy().astype(np.float32)
    feat = feat / np.linalg.norm(feat, keepdims=True)
    return feat.squeeze()
