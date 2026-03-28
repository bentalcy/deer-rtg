import argparse
import json
from pathlib import Path
from typing import cast

import cv2


def to_abs(repo_root: Path, image_path: str) -> Path:
    p = Path(image_path)
    return p if p.is_absolute() else repo_root / p


def padded_output_path(crop_path: Path, suffix: str) -> Path:
    return crop_path.with_name(f"{crop_path.stem}{suffix}{crop_path.suffix}")


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def parse_bbox(value: object) -> tuple[int, int, int, int] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    out: list[int] = []
    for item in value:
        if isinstance(item, (int, float)):
            out.append(int(item))
        else:
            return None
    return out[0], out[1], out[2], out[3]


def main() -> int:
    parser = argparse.ArgumentParser(description="Create padded display crops from tracklet actions")
    _ = parser.add_argument(
        "--actions-globs",
        default="data/pool/**/actions.json,runs/**/04_tracklets/actions.json",
    )
    _ = parser.add_argument("--pad-frac", type=float, default=0.15)
    _ = parser.add_argument("--min-pad", type=int, default=0)
    _ = parser.add_argument("--suffix", default="__pad")
    _ = parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    a = cast(dict[str, object], vars(args))
    patterns = [p.strip() for p in str(a.get("actions_globs", "")).split(",") if p.strip()]
    pad_frac_obj = a.get("pad_frac")
    pad_frac = float(pad_frac_obj) if isinstance(pad_frac_obj, (int, float)) else 0.15
    min_pad_obj = a.get("min_pad")
    min_pad = int(min_pad_obj) if isinstance(min_pad_obj, int) else 0
    suffix_obj = a.get("suffix")
    suffix = str(suffix_obj) if isinstance(suffix_obj, str) else "__pad"
    overwrite_obj = a.get("overwrite")
    overwrite = bool(overwrite_obj) if isinstance(overwrite_obj, bool) else False

    total = 0
    written = 0
    skipped = 0
    missing = 0

    for pattern in patterns:
        for actions_path in repo_root.glob(pattern):
            if not actions_path.is_file():
                continue
            try:
                raw = json.loads(actions_path.read_text())
            except Exception:
                continue
            if not isinstance(raw, list):
                continue
            for row in raw:
                if not isinstance(row, dict):
                    continue
                frame_path = str(row.get("frame_path") or "").strip()
                dets = row.get("detections") or []
                if not frame_path or not isinstance(dets, list):
                    continue
                abs_frame = to_abs(repo_root, frame_path)
                if not abs_frame.exists():
                    missing += 1
                    continue
                frame = cv2.imread(str(abs_frame))
                if frame is None:
                    missing += 1
                    continue
                height, width = frame.shape[:2]
                for det in dets:
                    if not isinstance(det, dict):
                        continue
                    crop_new = str(det.get("new_name") or "").strip()
                    bbox = parse_bbox(det.get("bbox"))
                    if not crop_new or bbox is None:
                        continue
                    abs_crop = to_abs(repo_root, crop_new)
                    if abs_crop.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                        continue
                    out_path = padded_output_path(abs_crop, suffix)
                    if out_path.exists() and not overwrite:
                        skipped += 1
                        continue
                    x1, y1, x2, y2 = bbox
                    if x2 <= x1 or y2 <= y1:
                        continue
                    pad_x = max(min_pad, int((x2 - x1) * pad_frac))
                    pad_y = max(min_pad, int((y2 - y1) * pad_frac))
                    px1 = clamp(x1 - pad_x, 0, width)
                    px2 = clamp(x2 + pad_x, 0, width)
                    py1 = clamp(y1 - pad_y, 0, height)
                    py2 = clamp(y2 + pad_y, 0, height)
                    if px2 <= px1 or py2 <= py1:
                        continue
                    crop = frame[py1:py2, px1:px2]
                    if crop.size == 0:
                        continue
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    ok = cv2.imwrite(str(out_path), crop)
                    total += 1
                    if ok:
                        written += 1
                    else:
                        skipped += 1

    print("crops_processed", total)
    print("crops_written", written)
    print("crops_skipped", skipped)
    print("frames_missing", missing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
