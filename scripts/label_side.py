import argparse
import csv
import random
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


KEY_TO_LABEL = {
    "d": "left",
    "j": "right",
    "l": "left",
    "r": "right",
    "u": "unknown",
    "e": "error",
}


def list_images(pool_dir: Path) -> list[Path]:
    files: list[Path] = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        files.extend(pool_dir.rglob(ext))
    return sorted(files)


def ensure_csv(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "label", "src_path", "dst_path"])
        writer.writeheader()


def load_labeled_src_paths(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out: set[str] = set()
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            src = (row.get("src_path") or "").strip()
            if src:
                out.add(src)
    return out


def append_csv(path: Path, label: str, src_path: Path, dst_path: Path) -> None:
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "label", "src_path", "dst_path"])
        writer.writerow(
            {
                "timestamp": int(__import__("time").time()),
                "label": label,
                "src_path": str(src_path),
                "dst_path": str(dst_path),
            }
        )


def pop_last_csv_row(path: Path) -> None:
    if not path.exists():
        return
    rows = list(csv.DictReader(path.open(newline="")))
    if not rows:
        return
    rows = rows[:-1]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "label", "src_path", "dst_path"])
        writer.writeheader()
        writer.writerows(rows)


def label_one(img_path: Path, idx: int, total: int) -> str | None:
    img = Image.open(img_path).convert("RGB")
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(img)
    ax.axis("off")
    fig.suptitle(
        (
            f"{idx}/{total}  {img_path.name}\n"
            "Keys: [d]=left  [j]=right  [u]=unknown  [e]=error  [z]=undo  [q]=quit"
        ),
        fontsize=10,
    )
    picked: dict[str, str | None] = {"label": None}

    def on_key(event):
        key = (event.key or "").lower()
        if key in KEY_TO_LABEL:
            picked["label"] = KEY_TO_LABEL[key]
            plt.close(fig)
            return
        if key in {"z", "backspace"}:
            picked["label"] = "__undo__"
            plt.close(fig)
            return
        if key == "q":
            picked["label"] = None
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.tight_layout()
    plt.show()
    return picked["label"]


def main(pool_dir: Path, out_dir: Path, labels_csv: Path, max_images: int, seed: int) -> None:
    images = list_images(pool_dir)
    if not images:
        print("No images found in", pool_dir)
        return

    rng = random.Random(seed)
    rng.shuffle(images)

    ensure_csv(labels_csv)
    already_labeled = load_labeled_src_paths(labels_csv)
    images = [p for p in images if str(p) not in already_labeled]
    if max_images > 0:
        images = images[:max_images]
    for label in ("left", "right", "unknown", "error"):
        (out_dir / label).mkdir(parents=True, exist_ok=True)

    if not images:
        print("No unlabeled images left for this pool/seed/csv.")
        print("Labels CSV:", labels_csv)
        return

    done = 0
    i = 0
    history: list[dict[str, Path]] = []
    while i < len(images):
        img_path = images[i]
        label = label_one(img_path, i + 1, len(images))
        if label is None:
            print("Stopped by user")
            break
        if label == "__undo__":
            if not history:
                print("Nothing to undo.")
                continue
            last = history.pop()
            dst = last["dst"]
            if dst.exists():
                dst.unlink()
            pop_last_csv_row(labels_csv)
            done = max(0, done - 1)
            i = max(0, i - 1)
            print(f"Undid last label. Progress: {done}/{len(images)}")
            continue
        dst = out_dir / label / img_path.name
        if dst.exists():
            stem, suf = dst.stem, dst.suffix
            k = 1
            while (out_dir / label / f"{stem}__dup{k}{suf}").exists():
                k += 1
            dst = out_dir / label / f"{stem}__dup{k}{suf}"
        shutil.copy2(img_path, dst)
        append_csv(labels_csv, label, img_path, dst)
        history.append({"src": img_path, "dst": dst})
        done += 1
        i += 1

    print(f"Labeled {done} images.")
    print("Wrote labels to", labels_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-dir", default="data/pool/unified_images_train_crops")
    parser.add_argument("--out-dir", default="data/labeled_side")
    parser.add_argument("--labels-csv", default="data/labeled_side/labels.csv")
    parser.add_argument("--max-images", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(
        pool_dir=Path(args.pool_dir),
        out_dir=Path(args.out_dir),
        labels_csv=Path(args.labels_csv),
        max_images=args.max_images,
        seed=args.seed,
    )
