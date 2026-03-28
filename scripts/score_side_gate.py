import argparse
import csv
from pathlib import Path

import torch
from typing import cast
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def list_images(pool_dir: Path) -> list[Path]:
    images: list[Path] = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        images.extend(pool_dir.rglob(ext))
    return sorted(images)


def load_model(model_path: Path):
    ckpt = torch.load(model_path, map_location="cpu")
    model_name = ckpt["model_name"]
    label_to_idx = ckpt["label_to_idx"]
    idx_to_label = {v: k for k, v in label_to_idx.items()}

    clip_model = cast(CLIPModel, CLIPModel.from_pretrained(model_name))
    clip_model.to(DEVICE)  # type: ignore[call-arg]
    processor = cast(CLIPProcessor, CLIPProcessor.from_pretrained(model_name))
    head = torch.nn.Linear(clip_model.config.projection_dim, len(label_to_idx))
    head.load_state_dict(ckpt["state_dict"])
    head = head.to(DEVICE)
    head.eval()
    clip_model.eval()
    return clip_model, processor, head, idx_to_label


def score_images(
    images: list[Path],
    clip_model: CLIPModel,
    processor: CLIPProcessor,
    head: torch.nn.Module,
    idx_to_label: dict[int, str],
    batch_size: int,
) -> list[dict]:
    rows = []
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        pil_images = [Image.open(p).convert("RGB") for p in batch]
        inputs = processor(images=pil_images, return_tensors="pt")
        pixel_values = cast(torch.FloatTensor, inputs["pixel_values"]).to(DEVICE)
        with torch.no_grad():
            feats = clip_model.get_image_features(pixel_values=pixel_values)  # type: ignore[arg-type]
            feats = feats / feats.norm(dim=-1, keepdim=True)
            logits = head(feats)
            probs = torch.softmax(logits, dim=-1).cpu()

        for path, prob_vec in zip(batch, probs):
            prob_list = prob_vec.tolist()
            pred_idx = int(torch.argmax(prob_vec).item())
            pred_label = idx_to_label[pred_idx]
            max_prob = float(max(prob_list))
            rows.append(
                {
                    "path": str(path),
                    "pred": pred_label,
                    "pred_prob": max_prob,
                    **{f"p_{idx_to_label[j]}": prob_list[j] for j in range(len(prob_list))},
                }
            )
    return rows


def write_scores_csv(rows: list[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "pred", "pred_prob"])
            writer.writeheader()
        return
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main(pool_dir: Path, model_path: Path, out_csv: Path, batch_size: int) -> None:
    images = list_images(pool_dir)
    if not images:
        print("No images found in pool")
        return
    clip_model, processor, head, idx_to_label = load_model(model_path)
    rows = score_images(images, clip_model, processor, head, idx_to_label, batch_size)
    write_scores_csv(rows, out_csv)
    print("Wrote side scores to", out_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-dir", default="data/pool/crops")
    parser.add_argument("--model-path", default="data/models/side_gate/side_gate.pt")
    parser.add_argument("--out-csv", default="data/active_learning/side_scores.csv")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    main(
        pool_dir=Path(args.pool_dir),
        model_path=Path(args.model_path),
        out_csv=Path(args.out_csv),
        batch_size=args.batch_size,
    )
