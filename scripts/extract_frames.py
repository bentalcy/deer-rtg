import cv2
from pathlib import Path
from tqdm import tqdm

VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi")


def extract_frames_from_video(video_path, out_dir, stride=30):
    """Extract every `stride`-th frame from a single video."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return 0

    frame_idx = 0
    saved = 0
    stem = Path(video_path).stem  # filename without extension

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % stride == 0:
            out_name = f"{stem}_frame{frame_idx:06d}.jpg"
            out_path = out_dir / out_name
            cv2.imwrite(str(out_path), frame)
            saved += 1

        frame_idx += 1

    cap.release()
    return saved


def main(videos_root, out_root, stride=30):
    videos_root = Path(videos_root)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    video_files = [
        p for p in videos_root.rglob("*")
        if p.suffix.lower() in VIDEO_EXTS
    ]

    print(f"Found {len(video_files)} videos.")

    for vid in tqdm(video_files):
        rel = vid.relative_to(videos_root)
        vid_out_dir = out_root / rel.parent
        vid_out_dir.mkdir(parents=True, exist_ok=True)
        n = extract_frames_from_video(vid, vid_out_dir, stride=stride)
        print(f"{vid}: saved {n} frames")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--videos-root", default="./videos")
    parser.add_argument("--out-root", default="./tmp_sample_frames")
    parser.add_argument("--stride", type=int, default=30)
    args = parser.parse_args()

    main(args.videos_root, args.out_root, args.stride)
