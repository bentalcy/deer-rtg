import cv2
from pathlib import Path
from tqdm import tqdm

VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi")


def extract_frames_from_video(
    video_path,
    out_dir,
    stride=30,
    max_frames=None,
    actions=None,
    every_seconds=None,
):
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
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    stride_frames = stride
    if every_seconds is not None and fps > 0:
        stride_frames = max(1, round(fps * every_seconds))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % stride_frames == 0:
            out_name = f"{stem}_frame{frame_idx:06d}.jpg"
            out_path = out_dir / out_name
            cv2.imwrite(str(out_path), frame)
            saved += 1
            if actions is not None:
                time_sec = frame_idx / fps if fps > 0 else None
                actions.append(
                    {
                        "orig_name": Path(video_path).name,
                        "frame": frame_idx,
                        "time_sec": time_sec,
                        "source_video": str(video_path),
                        "new_name": str(out_path),
                    }
                )
            if max_frames is not None and saved >= max_frames:
                break

        frame_idx += 1

    cap.release()
    return saved


def main(
    videos_root,
    out_root,
    stride=30,
    max_frames=None,
    actions_path=None,
    video_path=None,
    every_seconds=None,
):
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if video_path is not None:
        video_files = [Path(video_path)]
        if not video_files[0].exists():
            print(f"Video not found: {video_files[0]}")
            return
    else:
        videos_root = Path(videos_root)
        video_files = [
            p for p in videos_root.rglob("*")
            if p.suffix.lower() in VIDEO_EXTS
        ]

    print(f"Found {len(video_files)} videos.")

    actions = []
    for vid in tqdm(video_files):
        if video_path is not None:
            vid_out_dir = out_root
        else:
            rel = vid.relative_to(videos_root)
            vid_out_dir = out_root / rel.parent
            vid_out_dir.mkdir(parents=True, exist_ok=True)
        n = extract_frames_from_video(
            vid,
            vid_out_dir,
            stride=stride,
            max_frames=max_frames,
            actions=actions,
            every_seconds=every_seconds,
        )
        print(f"{vid}: saved {n} frames")

    if actions_path is not None:
        actions_path = Path(actions_path)
        actions_path.parent.mkdir(parents=True, exist_ok=True)
        import json

        with open(actions_path, "w") as f:
            json.dump(actions, f, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--videos-root", default="./videos")
    parser.add_argument("--out-root", default="./tmp_sample_frames")
    parser.add_argument("--stride", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--actions-path", default=None)
    parser.add_argument("--video-path", default=None)
    parser.add_argument("--every-seconds", type=float, default=None)
    args = parser.parse_args()

    max_frames = args.max_frames if args.max_frames and args.max_frames > 0 else None
    main(
        args.videos_root,
        args.out_root,
        args.stride,
        max_frames=max_frames,
        actions_path=args.actions_path,
        video_path=args.video_path,
        every_seconds=args.every_seconds,
    )
