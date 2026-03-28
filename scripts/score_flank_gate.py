import csv
import math
import shutil
from pathlib import Path

import torch
import cv2
from typing import cast
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def list_images(pool_dir: Path) -> list[Path]:
    images = []
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
):
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
            uncertainty = 1.0 - max_prob
            rows.append(
                {
                    "path": str(path),
                    "pred": pred_label,
                    "uncertainty": uncertainty,
                    **{f"p_{idx_to_label[i]}": prob_list[i] for i in range(len(prob_list))},
                }
            )

    return rows


def write_scores_csv(rows: list[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def stem_from_path(path: Path) -> str:
    return path.name.split("_frame", 1)[0]


def load_video_durations(videos_root: Path) -> dict[str, float]:
    durations: dict[str, float] = {}
    if not videos_root.exists():
        return durations

    for video in videos_root.rglob("*"):
        if video.suffix.lower() not in {".mp4", ".mov", ".mkv", ".avi"}:
            continue
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        cap.release()
        duration = frames / fps if fps > 0 else 0.0
        durations[video.stem] = duration

    return durations


def build_duration_caps(
    durations: dict[str, float],
    per_minute: float,
    min_cap: int,
    max_cap: int,
) -> dict[str, int]:
    caps: dict[str, int] = {}
    for stem, seconds in durations.items():
        minutes = seconds / 60.0 if seconds > 0 else 0.0
        cap = math.ceil(minutes * per_minute) if minutes > 0 else min_cap
        caps[stem] = max(min_cap, min(max_cap, cap))
    return caps


def select_next_batch(
    rows: list[dict],
    out_dir: Path,
    max_images: int,
    caps_by_stem: dict[str, int] | None,
) -> list[dict]:
    if max_images <= 0:
        return []
    rows_sorted = sorted(rows, key=lambda r: r["uncertainty"], reverse=True)
    selected = []
    counts: dict[str, int] = {}
    for row in rows_sorted:
        if len(selected) >= max_images:
            break
        stem = stem_from_path(Path(row["path"]))
        if caps_by_stem is not None:
            cap = caps_by_stem.get(stem, None)
            if cap is not None and counts.get(stem, 0) >= cap:
                continue
        selected.append(row)
        counts[stem] = counts.get(stem, 0) + 1
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(selected):
        src = Path(row["path"])
        dst = out_dir / f"{i:04d}__{src.name}"
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
        row["next_path"] = str(dst)
    return selected


def write_next_batch_preds(rows: list[dict], out_csv: Path) -> None:
    if not rows:
        return
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["path", "pred", "uncertainty"] + [k for k in rows[0].keys() if k.startswith("p_")]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "path": row["next_path"],
                    "pred": row["pred"],
                    "uncertainty": row["uncertainty"],
                    **{k: row[k] for k in row if k.startswith("p_")},
                }
            )


def main(
    pool_dir: Path,
    model_path: Path,
    out_csv: Path,
    next_batch_dir: Path | None,
    next_batch_size: int,
    next_batch_preds_csv: Path | None,
    batch_size: int,
    videos_root: Path,
    cap_per_minute: float,
    cap_min: int,
    cap_max: int,
) -> None:
    images = list_images(pool_dir)
    if not images:
        print("No images found in pool")
        return

    clip_model, processor, head, idx_to_label = load_model(model_path)
    rows = score_images(images, clip_model, processor, head, idx_to_label, batch_size)
    write_scores_csv(rows, out_csv)

    if next_batch_dir is not None:
        caps_by_stem = None
        if cap_per_minute > 0:
            durations = load_video_durations(videos_root)
            caps_by_stem = build_duration_caps(durations, cap_per_minute, cap_min, cap_max)
        selected = select_next_batch(rows, next_batch_dir, next_batch_size, caps_by_stem)
        if next_batch_preds_csv is None:
            next_batch_preds_csv = next_batch_dir / "preds.csv"
        write_next_batch_preds(selected, next_batch_preds_csv)
        print(f"Selected {len(selected)} images for next batch")

    print("Wrote scores to", out_csv)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-dir", default="data/pool/crops")
    parser.add_argument("--model-path", default="data/models/flank_gate.pt")
    parser.add_argument("--out-csv", default="data/active_learning/scores.csv")
    parser.add_argument("--next-batch-dir", default="data/active_learning/next_batch")
    parser.add_argument("--next-batch-size", type=int, default=200)
    parser.add_argument("--next-batch-preds-csv", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--videos-root", default="videos")
    parser.add_argument("--cap-per-minute", type=float, default=10.0)
    parser.add_argument("--cap-min", type=int, default=10)
    parser.add_argument("--cap-max", type=int, default=50)
    args = parser.parse_args()

    next_batch_dir = Path(args.next_batch_dir) if args.next_batch_dir else None
    main(
        pool_dir=Path(args.pool_dir),
        model_path=Path(args.model_path),
        out_csv=Path(args.out_csv),
        next_batch_dir=next_batch_dir,
        next_batch_size=args.next_batch_size,
        next_batch_preds_csv=Path(args.next_batch_preds_csv) if args.next_batch_preds_csv else None,
        batch_size=args.batch_size,
        videos_root=Path(args.videos_root),
        cap_per_minute=args.cap_per_minute,
        cap_min=args.cap_min,
        cap_max=args.cap_max,
    )
