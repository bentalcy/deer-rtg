import json
import random
import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from typing import cast
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import CLIPModel, CLIPProcessor


MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


@dataclass
class LabeledImage:
    path: Path
    label: int


def list_labeled_images(base_dir: Path, label_id: int, exclude_fragment: str | None) -> list[LabeledImage]:
    if not base_dir.exists():
        return []
    images: list[LabeledImage] = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for p in base_dir.rglob(ext):
            if exclude_fragment and exclude_fragment in str(p):
                continue
            images.append(LabeledImage(path=p, label=label_id))
    return images


def train_val_split(items: list[LabeledImage], val_ratio: float, seed: int) -> tuple[list[LabeledImage], list[LabeledImage]]:
    by_label: dict[int, list[LabeledImage]] = {}
    for item in items:
        by_label.setdefault(item.label, []).append(item)

    train: list[LabeledImage] = []
    val: list[LabeledImage] = []
    rng = random.Random(seed)
    for _, group in by_label.items():
        rng.shuffle(group)
        split = max(1, int(len(group) * (1 - val_ratio))) if len(group) > 1 else 1
        train.extend(group[:split])
        val.extend(group[split:])
    return train, val


def embed_images(model: CLIPModel, processor: CLIPProcessor, items: list[LabeledImage], batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    feats = []
    labels = []
    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        images = [Image.open(item.path).convert("RGB") for item in batch]
        inputs = processor(images=images, return_tensors="pt")
        pixel_values = cast(torch.FloatTensor, inputs["pixel_values"]).to(DEVICE)
        with torch.no_grad():
            image_features = model.get_image_features(pixel_values=pixel_values)  # type: ignore[arg-type]
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        feats.append(image_features.cpu())
        labels.extend([item.label for item in batch])
    feats_tensor = torch.cat(feats, dim=0)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return feats_tensor, labels_tensor


def evaluate(head: nn.Module, feats: torch.Tensor, labels: torch.Tensor, idx_to_label: list[str]) -> dict:
    head.eval()
    with torch.no_grad():
        logits = head(feats)
        preds = logits.argmax(dim=1)
    total = labels.numel()
    acc = (preds == labels).sum().item() / total if total else 0.0
    metrics = {"accuracy": acc, "per_class": {}}
    for idx, name in enumerate(idx_to_label):
        tp = ((preds == idx) & (labels == idx)).sum().item()
        pred_pos = (preds == idx).sum().item()
        actual_pos = (labels == idx).sum().item()
        precision = tp / pred_pos if pred_pos else 0.0
        recall = tp / actual_pos if actual_pos else 0.0
        metrics["per_class"][name] = {"precision": precision, "recall": recall, "count": actual_pos}
    return metrics


def main(
    labeled_dir: Path,
    out_dir: Path,
    labels: list[str],
    val_ratio: float,
    seed: int,
    batch_size: int,
    epochs: int,
    lr: float,
    exclude_fragment: str | None,
) -> None:
    if len(labels) < 2:
        raise ValueError("need at least two labels")

    label_to_idx = {name: i for i, name in enumerate(labels)}
    items: list[LabeledImage] = []
    for label in labels:
        items += list_labeled_images(labeled_dir / label, label_to_idx[label], exclude_fragment)

    if not items:
        print("No labeled side images found.")
        return

    train_items, val_items = train_val_split(items, val_ratio=val_ratio, seed=seed)

    model = cast(CLIPModel, CLIPModel.from_pretrained(MODEL_NAME))
    model.to(DEVICE)  # type: ignore[call-arg]
    processor = cast(CLIPProcessor, CLIPProcessor.from_pretrained(MODEL_NAME))
    model.eval()

    train_feats, train_labels = embed_images(model, processor, train_items, batch_size)
    val_feats, val_labels = embed_images(model, processor, val_items, batch_size)

    head = nn.Linear(train_feats.shape[1], len(labels)).to(DEVICE)
    class_counts = torch.bincount(train_labels, minlength=len(labels)).float()
    class_weights = class_counts.sum() / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.mean()
    class_weights = class_weights.to(DEVICE)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    train_feats = train_feats.to(DEVICE)
    train_labels = train_labels.to(DEVICE)
    val_feats = val_feats.to(DEVICE)
    val_labels = val_labels.to(DEVICE)

    loader = DataLoader(TensorDataset(train_feats, train_labels), batch_size=min(batch_size, len(train_feats)), shuffle=True)

    for epoch in range(epochs):
        head.train()
        total_loss = 0.0
        for feats_batch, labels_batch in loader:
            optimizer.zero_grad()
            logits = head(feats_batch)
            loss = loss_fn(logits, labels_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch + 1 == epochs:
            metrics = evaluate(head, val_feats, val_labels, labels)
            print(f"epoch={epoch+1} loss={total_loss:.4f} val_acc={metrics['accuracy']:.3f}")

    metrics = evaluate(head, val_feats, val_labels, labels)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": head.state_dict(),
            "label_to_idx": label_to_idx,
            "model_name": MODEL_NAME,
        },
        out_dir / "side_gate.pt",
    )
    with open(out_dir / "side_gate_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "side_gate_meta.json", "w") as f:
        json.dump(
            {
                "train_count": len(train_items),
                "val_count": len(val_items),
                "labels": labels,
                "model_name": MODEL_NAME,
                "val_ratio": val_ratio,
                "seed": seed,
                "exclude_fragment": exclude_fragment,
                "class_weights": class_weights.detach().cpu().tolist(),
            },
            f,
            indent=2,
        )
    print("Saved model to", out_dir / "side_gate.pt")
    print("Saved metrics to", out_dir / "side_gate_metrics.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled-dir", default="data/labeled_side")
    parser.add_argument("--out-dir", default="data/models/side_gate")
    parser.add_argument("--labels", default="left,right,unknown")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--exclude-fragment", default="/frames/")
    args = parser.parse_args()

    labels = [x.strip() for x in args.labels.split(",") if x.strip()]
    main(
        labeled_dir=Path(args.labeled_dir),
        out_dir=Path(args.out_dir),
        labels=labels,
        val_ratio=args.val_ratio,
        seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        exclude_fragment=args.exclude_fragment,
    )
