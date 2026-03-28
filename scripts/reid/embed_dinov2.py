import csv
from pathlib import Path
from typing import cast

import torch
import numpy as np
from PIL import Image


DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def list_images(root: Path) -> list[Path]:
    images = [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    return sorted(images)


def load_index(index_csv: Path) -> list[dict[str, str]]:
    with index_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        rows: list[dict[str, str]] = []
        for row in reader:
            instance_id = (row.get("instance_id") or "").strip()
            image_path = (row.get("image_path") or "").strip()
            if not instance_id or not image_path:
                continue
            rows.append(
                {
                    "instance_id": instance_id,
                    "image_path": image_path,
                    "side_pred": (row.get("side_pred") or "").strip(),
                    "side_pred_prob": (row.get("side_pred_prob") or "").strip(),
                    "recognizability_pred": (row.get("recognizability_pred") or "").strip(),
                    "recognizability_pred_prob": (row.get("recognizability_pred_prob") or "").strip(),
                }
            )
        return rows


def load_dinov2(model_name: str = "dinov2_vitb14"):
    model = cast(torch.nn.Module, torch.hub.load("facebookresearch/dinov2", model_name))
    model.to(DEVICE)
    model.eval()
    return model


def load_image(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    img = img.resize((518, 518))
    arr = np.array(img).astype("float32") / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return (tensor - MEAN) / STD


def embed_images(model: torch.nn.Module, images: list[Path], batch_size: int) -> torch.Tensor:
    feats = []
    for i in range(0, len(images), batch_size):
        batch_paths = images[i : i + batch_size]
        batch = torch.stack([load_image(p) for p in batch_paths]).to(DEVICE)
        with torch.no_grad():
            out = model(batch)
            if isinstance(out, (tuple, list)):
                out = out[0]
            out = out / out.norm(dim=-1, keepdim=True)
        feats.append(out.cpu())
    return torch.cat(feats, dim=0)


def write_metadata(
    instance_ids: list[str],
    image_paths: list[str],
    side_preds: list[str],
    side_pred_probs: list[str],
    recognizability_preds: list[str],
    recognizability_pred_probs: list[str],
    out_csv: Path,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "instance_id",
                "image_path",
                "side_pred",
                "side_pred_prob",
                "recognizability_pred",
                "recognizability_pred_prob",
            ],
        )
        writer.writeheader()
        for instance_id, p, side_pred, side_pred_prob, rec_pred, rec_prob in zip(
            instance_ids,
            image_paths,
            side_preds,
            side_pred_probs,
            recognizability_preds,
            recognizability_pred_probs,
        ):
            writer.writerow(
                {
                    "instance_id": instance_id,
                    "image_path": p,
                    "side_pred": side_pred,
                    "side_pred_prob": side_pred_prob,
                    "recognizability_pred": rec_pred,
                    "recognizability_pred_prob": rec_prob,
                }
            )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index-csv",
        default="data/reid/index.csv",
        help="CSV produced by scripts/reid/build_index.py (preferred input).",
    )
    parser.add_argument(
        "--images-root",
        default="data/reid/flank_pool",
        help="Fallback image root if index-csv is missing/empty.",
    )
    parser.add_argument("--out-embeddings", default="data/reid/embeddings.pt")
    parser.add_argument("--out-metadata", default="data/reid/embeddings.csv")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model", default="dinov2_vitb14")
    args = parser.parse_args()

    index_rows: list[dict[str, str]] = []
    index_path = Path(args.index_csv)
    if index_path.exists():
        index_rows = load_index(index_path)

    image_path_strings: list[str]
    side_preds: list[str]
    side_pred_probs: list[str]
    recognizability_preds: list[str]
    recognizability_pred_probs: list[str]
    if index_rows:
        instance_ids = [r["instance_id"] for r in index_rows]
        image_path_strings = [r["image_path"] for r in index_rows]
        side_preds = [r.get("side_pred", "") for r in index_rows]
        side_pred_probs = [r.get("side_pred_prob", "") for r in index_rows]
        recognizability_preds = [r.get("recognizability_pred", "") for r in index_rows]
        recognizability_pred_probs = [r.get("recognizability_pred_prob", "") for r in index_rows]
        images = [Path(p) for p in image_path_strings]
    else:
        instance_ids = []
        images = list_images(Path(args.images_root))
        image_path_strings = [str(p) for p in images]
        side_preds = ["" for _ in images]
        side_pred_probs = ["" for _ in images]
        recognizability_preds = ["" for _ in images]
        recognizability_pred_probs = ["" for _ in images]

    if not images:
        print("No images found for embeddings")
        return

    # Resolve relative paths against repo root for loading.
    repo_root = Path(__file__).resolve().parents[2]
    resolved_images: list[Path] = []
    for p in images:
        if p.is_absolute():
            resolved_images.append(p)
        else:
            resolved_images.append(repo_root / p)
    images = resolved_images

    if not instance_ids:
        instance_ids = ["" for _ in images]

    model = load_dinov2(args.model)
    feats = embed_images(model, images, args.batch_size)
    out_path = Path(args.out_embeddings)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(feats, out_path)

    # Keep this metadata file backward compatible: downstream can still read image_path.
    write_metadata(
        instance_ids,
        image_path_strings,
        side_preds,
        side_pred_probs,
        recognizability_preds,
        recognizability_pred_probs,
        Path(args.out_metadata),
    )
    print("Saved embeddings to", out_path)


if __name__ == "__main__":
    main()
