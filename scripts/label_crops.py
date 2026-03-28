import csv
import random
import shutil
import time
from pathlib import Path
from typing import Any

import cv2


LABEL_KEEP = "keep"
LABEL_REJECT = "reject"
LABEL_SKIP = "skip"


def list_images(pool_dir: Path, extensions: set[str], require_fragment: str | None) -> list[Path]:
    images = []
    for ext in extensions:
        images.extend(pool_dir.rglob(f"*.{ext}"))
    if require_fragment:
        images = [p for p in images if require_fragment in str(p)]
    return sorted(images)


def load_predictions(preds_csv: Path) -> dict[str, str]:
    if not preds_csv.exists():
        return {}
    predictions: dict[str, str] = {}
    with open(preds_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            path = row.get("path")
            pred = row.get("pred")
            if path and pred:
                if pred in {"reject_back", "reject_other"}:
                    pred = LABEL_REJECT
                predictions[path] = pred
                predictions[Path(path).name] = pred
    return predictions


def ensure_labels_csv(labels_csv: Path) -> None:
    if labels_csv.exists():
        return
    labels_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(labels_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "label", "src_path", "dst_path"])


def append_label(labels_csv: Path, label: str, src_path: Path, dst_path: Path) -> None:
    with open(labels_csv, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([int(time.time()), label, str(src_path), str(dst_path)])


def pop_last_label(labels_csv: Path) -> None:
    if not labels_csv.exists():
        return
    lines = labels_csv.read_text().splitlines()
    if len(lines) <= 1:
        return
    labels_csv.write_text("\n".join(lines[:-1]) + "\n")


def draw_overlay(img, lines: list[str]) -> Any:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2
    padding_top = 8
    padding_bottom = 8
    padding_left = 16
    text_size = cv2.getTextSize("Ag", font, font_scale, thickness)[0]
    line_height = max(18, int(text_size[1] * 1.6))
    band_height = padding_top + padding_bottom + line_height * len(lines)

    band_height = max(1, band_height)
    display = cv2.copyMakeBorder(
        img,
        band_height,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )

    y = padding_top + line_height - 6
    for line in lines:
        cv2.putText(
            display,
            line,
            (padding_left, y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        y += line_height

    return display


def move_to_label(
    img_path: Path,
    pool_dir: Path,
    out_dir: Path,
    label: str,
) -> Path:
    rel = img_path.relative_to(pool_dir)
    dest = out_dir / label / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(img_path), str(dest))
    return dest


def main(
    pool_dir: Path,
    out_dir: Path,
    labels_csv: Path,
    extensions: set[str],
    shuffle: bool,
    seed: int | None,
    max_images: int | None,
    preds_csv: Path | None,
    require_fragment: str | None,
) -> None:
    ensure_labels_csv(labels_csv)
    images = list_images(pool_dir, extensions, require_fragment)
    if not images:
        print(f"No images found under {pool_dir}")
        return

    if shuffle:
        rnd = random.Random(seed)
        rnd.shuffle(images)

    if max_images is not None:
        images = images[:max_images]

    predictions = load_predictions(preds_csv) if preds_csv is not None else {}

    history: list[dict] = []
    idx = 0

    while idx < len(images):
        img_path = images[idx]
        if not img_path.exists():
            idx += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            idx += 1
            continue

        predicted_label = predictions.get(str(img_path))
        if predicted_label is None:
            predicted_label = predictions.get(img_path.name)
        overlay = draw_overlay(
            img,
            [
                f"{idx + 1}/{len(images)}  {img_path.name}",
                "k=KEEP  r=REJECT  b=REJECT  s=SKIP  c=CORRECT",
                "u=UNDO  q=QUIT",
                "KEEP only if flank >=85% and side view",
                "BACK view is always reject",
                f"pred: {predicted_label or 'n/a'}",
            ],
        )

        cv2.imshow("label_crops", overlay)
        key = cv2.waitKey(0) & 0xFF

        if key == ord("q"):
            break
        if key == ord("u"):
            if history:
                last = history.pop()
                dst_path = Path(last["dst_path"])
                src_path = Path(last["src_path"])
                src_path.parent.mkdir(parents=True, exist_ok=True)
                if dst_path.exists():
                    shutil.move(str(dst_path), str(src_path))
                pop_last_label(labels_csv)
                idx = last["index"]
            else:
                idx = max(0, idx - 1)
            continue

        if key == ord("c"):
            if predicted_label in {LABEL_KEEP, LABEL_REJECT}:
                label = predicted_label
            else:
                continue
        elif key == ord("k"):
            label = LABEL_KEEP
        elif key == ord("b"):
            label = LABEL_REJECT
        elif key == ord("r"):
            label = LABEL_REJECT
        elif key == ord("s"):
            label = LABEL_SKIP
        else:
            continue

        dst_path = move_to_label(img_path, pool_dir, out_dir, label)
        append_label(labels_csv, label, img_path, dst_path)
        history.append(
            {
                "index": idx,
                "src_path": str(img_path),
                "dst_path": str(dst_path),
                "label": label,
            }
        )
        idx += 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-dir", required=True)
    parser.add_argument("--out-dir", default="data/labeled")
    parser.add_argument("--labels-csv", default=None)
    parser.add_argument("--extensions", default="jpg,jpeg,png")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--preds-csv", default=None, help="CSV with path,pred columns")
    parser.add_argument("--require-fragment", default=None, help="Only label files whose path contains this fragment")
    args = parser.parse_args()

    pool_dir = Path(args.pool_dir)
    out_dir = Path(args.out_dir)
    labels_csv = Path(args.labels_csv) if args.labels_csv else out_dir / "labels.csv"
    extensions = {e.strip().lower() for e in args.extensions.split(",") if e.strip()}

    main(
        pool_dir=pool_dir,
        out_dir=out_dir,
        labels_csv=labels_csv,
        extensions=extensions,
        shuffle=args.shuffle,
        seed=args.seed,
        max_images=args.max_images,
        preds_csv=Path(args.preds_csv) if args.preds_csv else None,
        require_fragment=args.require_fragment,
    )
