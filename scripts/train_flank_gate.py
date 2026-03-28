import json
import random
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


def list_labeled_images(
    base_dir: Path,
    label_name: str,
    label_id: int,
    exclude_fragment: str | None,
) -> list[LabeledImage]:
    if not base_dir.exists():
        return []
    images = []
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
    for label, group in by_label.items():
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


def evaluate(head: nn.Module, feats: torch.Tensor, labels: torch.Tensor, label_names: list[str]) -> dict:
    head.eval()
    with torch.no_grad():
        logits = head(feats)
        preds = logits.argmax(dim=1)

    total = labels.numel()
    correct = (preds == labels).sum().item()
    acc = correct / total if total else 0.0

    metrics = {"accuracy": acc, "per_class": {}}
    for idx, name in enumerate(label_names):
        true_pos = ((preds == idx) & (labels == idx)).sum().item()
        pred_pos = (preds == idx).sum().item()
        actual_pos = (labels == idx).sum().item()
        precision = true_pos / pred_pos if pred_pos else 0.0
        recall = true_pos / actual_pos if actual_pos else 0.0
        metrics["per_class"][name] = {
            "precision": precision,
            "recall": recall,
            "count": actual_pos,
        }

    return metrics


def evaluate_expected_cost(
    head: nn.Module,
    feats: torch.Tensor,
    labels: torch.Tensor,
    label_names: list[str],
    cost_false_keep: float,
    cost_false_reject: float,
    threshold_step: float,
) -> dict:
    if "keep" not in label_names or "reject" not in label_names:
        return {}

    keep_idx = label_names.index("keep")
    reject_idx = label_names.index("reject")
    if threshold_step <= 0:
        threshold_step = 0.01
    thresholds = [round(i * threshold_step, 6) for i in range(int(1 / threshold_step) + 1)]

    head.eval()
    with torch.no_grad():
        logits = head(feats)
        probs = torch.softmax(logits, dim=1)
        keep_probs = probs[:, keep_idx]

    total = labels.numel()
    best = {
        "expected_cost": float("inf"),
        "threshold": None,
        "false_keep_rate": None,
        "false_reject_rate": None,
    }

    for threshold in thresholds:
        preds_keep = keep_probs >= threshold
        fp = ((labels == reject_idx) & preds_keep).sum().item()
        fn = ((labels == keep_idx) & (~preds_keep)).sum().item()
        expected_cost = (fp * cost_false_keep + fn * cost_false_reject) / total if total else 0.0
        if expected_cost < best["expected_cost"]:
            best = {
                "expected_cost": expected_cost,
                "threshold": threshold,
                "false_keep_rate": fp / total if total else 0.0,
                "false_reject_rate": fn / total if total else 0.0,
            }

    return {
        "cost_false_keep": cost_false_keep,
        "cost_false_reject": cost_false_reject,
        "threshold_step": threshold_step,
        "best": best,
    }


def main(
    labeled_dir: Path,
    out_dir: Path,
    val_ratio: float,
    seed: int,
    batch_size: int,
    epochs: int,
    lr: float,
    exclude_fragment: str | None,
    cost_false_keep: float,
    cost_false_reject: float,
    threshold_step: float,
) -> None:
    label_to_idx = {
        "keep": 0,
        "reject": 1,
    }
    idx_to_label = ["keep", "reject"]

    items: list[LabeledImage] = []
    items += list_labeled_images(
        labeled_dir / "keep",
        "keep",
        label_to_idx["keep"],
        exclude_fragment,
    )
    items += list_labeled_images(
        labeled_dir / "reject",
        "reject",
        label_to_idx["reject"],
        exclude_fragment,
    )
    items += list_labeled_images(
        labeled_dir / "reject_back",
        "reject",
        label_to_idx["reject"],
        exclude_fragment,
    )
    items += list_labeled_images(
        labeled_dir / "reject_other",
        "reject",
        label_to_idx["reject"],
        exclude_fragment,
    )

    if not items:
        print("No labeled images found.")
        return

    train_items, val_items = train_val_split(items, val_ratio=val_ratio, seed=seed)

    model = cast(CLIPModel, CLIPModel.from_pretrained(MODEL_NAME))
    model.to(DEVICE)  # type: ignore[call-arg]
    processor = cast(CLIPProcessor, CLIPProcessor.from_pretrained(MODEL_NAME))
    model.eval()

    train_feats, train_labels = embed_images(model, processor, train_items, batch_size)
    val_feats, val_labels = embed_images(model, processor, val_items, batch_size)

    head = nn.Linear(train_feats.shape[1], len(label_to_idx)).to(DEVICE)
    class_counts = torch.bincount(train_labels, minlength=len(label_to_idx)).float()
    class_weights = class_counts.sum() / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.mean()
    class_weights = class_weights.to(DEVICE)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    train_feats = train_feats.to(DEVICE)
    train_labels = train_labels.to(DEVICE)
    val_feats = val_feats.to(DEVICE)
    val_labels = val_labels.to(DEVICE)

    dataset = TensorDataset(train_feats, train_labels)
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)

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
            metrics = evaluate(head, val_feats, val_labels, idx_to_label)
            print(f"epoch={epoch+1} loss={total_loss:.4f} val_acc={metrics['accuracy']:.3f}")

    metrics = evaluate(head, val_feats, val_labels, idx_to_label)
    metrics["expected_cost"] = evaluate_expected_cost(
        head,
        val_feats,
        val_labels,
        idx_to_label,
        cost_false_keep,
        cost_false_reject,
        threshold_step,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": head.state_dict(),
            "label_to_idx": label_to_idx,
            "model_name": MODEL_NAME,
        },
        out_dir / "flank_gate.pt",
    )
    with open(out_dir / "flank_gate_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(out_dir / "flank_gate_meta.json", "w") as f:
        json.dump(
            {
                "train_count": len(train_items),
                "val_count": len(val_items),
                "labels": idx_to_label,
                "model_name": MODEL_NAME,
                "val_ratio": val_ratio,
                "seed": seed,
                "exclude_fragment": exclude_fragment,
                "class_weights": class_weights.detach().cpu().tolist(),
            },
            f,
            indent=2,
        )

    print("Saved model to", out_dir / "flank_gate.pt")
    print("Saved metrics to", out_dir / "flank_gate_metrics.json")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled-dir", default="data/labeled")
    parser.add_argument("--out-dir", default="data/models")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--exclude-fragment", default="/frames/")
    parser.add_argument("--cost-false-keep", type=float, default=5.0)
    parser.add_argument("--cost-false-reject", type=float, default=1.0)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    args = parser.parse_args()

    main(
        labeled_dir=Path(args.labeled_dir),
        out_dir=Path(args.out_dir),
        val_ratio=args.val_ratio,
        seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        exclude_fragment=args.exclude_fragment,
        cost_false_keep=args.cost_false_keep,
        cost_false_reject=args.cost_false_reject,
        threshold_step=args.threshold_step,
    )
