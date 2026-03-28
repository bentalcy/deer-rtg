import csv
import random
import shutil
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def list_track_images(track_dir: Path) -> list[Path]:
    images = [p for p in track_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    return sorted(images)


def sample_evenly(images: list[Path], count: int) -> list[Path]:
    if count <= 0:
        return []
    if len(images) <= count:
        return images
    step = (len(images) - 1) / (count - 1)
    indices = [round(i * step) for i in range(count)]
    return [images[i] for i in indices]


def build_pool(
    tracklets_root: Path,
    out_dir: Path,
    samples_per_track: int,
    cap_per_track: int,
    seed: int,
) -> Path:
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.csv"
    rows: list[dict] = []

    track_dirs = sorted([p for p in tracklets_root.iterdir() if p.is_dir()])
    for track_dir in track_dirs:
        images = list_track_images(track_dir)
        if not images:
            continue

        if cap_per_track > 0:
            images = rng.sample(images, min(len(images), cap_per_track))
            images = sorted(images)

        sampled = sample_evenly(images, samples_per_track)
        if not sampled:
            continue

        dst_track_dir = out_dir / track_dir.name
        dst_track_dir.mkdir(parents=True, exist_ok=True)
        for src in sampled:
            dst = dst_track_dir / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
            rows.append(
                {
                    "track_id": track_dir.name,
                    "src_path": str(src),
                    "dst_path": str(dst),
                }
            )

    with index_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["track_id", "src_path", "dst_path"])
        writer.writeheader()
        writer.writerows(rows)

    return index_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tracklets-root", default="runs/poc1_food1_1/04_tracklets")
    parser.add_argument("--out-dir", default="data/reid/flank_pool")
    parser.add_argument("--samples-per-track", type=int, default=5)
    parser.add_argument("--cap-per-track", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    index_path = build_pool(
        tracklets_root=Path(args.tracklets_root),
        out_dir=Path(args.out_dir),
        samples_per_track=args.samples_per_track,
        cap_per_track=args.cap_per_track,
        seed=args.seed,
    )
    print("Wrote pool index to", index_path)


if __name__ == "__main__":
    main()
