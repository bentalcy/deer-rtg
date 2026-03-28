import csv
import random
import time
from pathlib import Path
from typing import Any

import cv2

LABEL_RECOGNIZABLE = "recognizable"
LABEL_UNRECOGNIZABLE = "unrecognizable"
LABEL_BAD = "bad_image"
LABEL_SKIP = "skip"


def list_images(pool_dir: Path, extensions: set[str]) -> list[Path]:
    images: list[Path] = []
    for ext in extensions:
        images.extend(pool_dir.rglob(f"*.{ext}"))
    return sorted(images)


def ensure_csv(out_csv: Path) -> None:
    if out_csv.exists():
        return
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "label", "path"])


def append_label(out_csv: Path, label: str, path: Path) -> None:
    with out_csv.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([int(time.time()), label, str(path)])


def pop_last_label(out_csv: Path) -> None:
    if not out_csv.exists():
        return
    lines = out_csv.read_text().splitlines()
    if len(lines) <= 1:
        return
    out_csv.write_text("\n".join(lines[:-1]) + "\n")


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


def main(
    images_dir: Path,
    out_csv: Path,
    extensions: set[str],
    shuffle: bool,
    seed: int | None,
    max_images: int | None,
) -> None:
    ensure_csv(out_csv)
    images = list_images(images_dir, extensions)
    if not images:
        print(f"No images found under {images_dir}")
        return

    if shuffle:
        rnd = random.Random(seed)
        rnd.shuffle(images)

    if max_images is not None:
        images = images[:max_images]

    history: list[dict[str, str | int]] = []
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

        overlay = draw_overlay(
            img,
            [
                f"{idx + 1}/{len(images)}  {img_path.name}",
                "y=RECOGNIZABLE  n=UNRECOGNIZABLE  b=BAD  s=SKIP",
                "u=UNDO  q=QUIT",
                "Recognizable = you can confidently tell which deer",
            ],
        )

        cv2.imshow("review_recognizability", overlay)
        key = cv2.waitKey(0) & 0xFF

        if key == ord("q"):
            break
        if key == ord("u"):
            if history:
                last = history.pop()
                pop_last_label(out_csv)
                idx = int(last.get("index", max(0, idx - 1)))
            else:
                idx = max(0, idx - 1)
            continue

        if key == ord("y"):
            label = LABEL_RECOGNIZABLE
        elif key == ord("n"):
            label = LABEL_UNRECOGNIZABLE
        elif key == ord("b"):
            label = LABEL_BAD
        elif key == ord("s"):
            label = LABEL_SKIP
        else:
            continue

        append_label(out_csv, label, img_path)
        history.append(
            {
                "index": idx,
                "path": str(img_path),
                "label": label,
            }
        )
        idx += 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", default="data/verify_filtered/next_batch")
    parser.add_argument("--out-csv", default="data/verify_filtered/recognizability.csv")
    parser.add_argument("--extensions", default="jpg,jpeg,png")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    out_csv = Path(args.out_csv)
    extensions = {e.strip().lower() for e in args.extensions.split(",") if e.strip()}

    main(
        images_dir=images_dir,
        out_csv=out_csv,
        extensions=extensions,
        shuffle=args.shuffle,
        seed=args.seed,
        max_images=args.max_images,
    )
