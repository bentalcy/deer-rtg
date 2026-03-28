import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

@dataclass(frozen=True)
class FrameMeta:
    orig_name: str | None
    source_video: str | None
    frame_idx: int | None
    time_sec: float | None
    new_name: str

    @property
    def video_id(self) -> str | None:
        if self.source_video is None:
            return None
        return Path(self.source_video).stem


def _sha1_id(value: str, length: int = 16) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def _normalize_path(path_str: str) -> str:
    if not path_str:
        return ""
    p = Path(path_str)
    if not p.is_absolute():
        return path_str

    try:
        resolved = p.resolve()
    except FileNotFoundError:
        resolved = p

    try:
        rel = resolved.relative_to(REPO_ROOT)
    except ValueError:
        return str(resolved)
    return str(rel)


def _infer_run_id(path: Path) -> str | None:
    parts = list(path.parts)
    if "runs" not in parts:
        return None
    i = parts.index("runs")
    if i + 1 >= len(parts):
        return None
    return parts[i + 1]


def _load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def load_frames_meta(frames_actions: Path) -> dict[str, FrameMeta]:
    raw = _load_json(frames_actions)
    by_frame_path: dict[str, FrameMeta] = {}
    for row in raw:
        frame_path = _normalize_path(str(row.get("new_name") or ""))
        if not frame_path:
            continue
        source_video = _normalize_path(str(row.get("source_video") or ""))
        by_frame_path[frame_path] = FrameMeta(
            orig_name=row.get("orig_name"),
            source_video=source_video or None,
            frame_idx=row.get("frame"),
            time_sec=row.get("time_sec"),
            new_name=frame_path,
        )
    return by_frame_path


def iter_tracklet_instances(tracklets_actions: Path):
    raw = _load_json(tracklets_actions)
    for frame_action in raw:
        frame_path = _normalize_path(str(frame_action.get("frame_path") or ""))
        dets = frame_action.get("detections") or []
        for det_i, det in enumerate(dets):
            crop_path = _normalize_path(str(det.get("new_name") or ""))
            if not crop_path:
                continue
            yield {
                "frame_path": frame_path,
                "det_i": det_i,
                "track_id": det.get("track_id"),
                "bbox": det.get("bbox"),
                "det_conf": det.get("conf"),
                "cls": det.get("cls"),
                "cls_name": det.get("cls_name"),
                "image_path": crop_path,
                "flank_keep_prob": det.get("flank_keep_prob"),
                "side_pred": det.get("side_pred"),
                "side_pred_prob": det.get("side_pred_prob"),
                "recognizability_pred": det.get("recognizability_pred"),
                "recognizability_pred_prob": det.get("recognizability_pred_prob"),
            }


def build_index_for_run(
    run_id: str,
    frames_meta: dict[str, FrameMeta],
    tracklets_actions: Path,
    encounter_gap_sec: float,
) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    missing_frame_meta = 0

    for inst in iter_tracklet_instances(tracklets_actions):
        frame_path = inst["frame_path"]
        meta = frames_meta.get(frame_path)
        if meta is None:
            missing_frame_meta += 1

        crop_path = inst["image_path"]
        instance_id = _sha1_id(f"{run_id}|{crop_path}")

        time_sec = meta.time_sec if meta is not None else None
        encounter_id = None
        if time_sec is not None and encounter_gap_sec > 0:
            bucket = int(time_sec // encounter_gap_sec)
            vid = meta.video_id if meta is not None else None
            if vid is not None:
                encounter_id = f"{vid}__b{bucket:06d}"

        rows.append(
            {
                "run_id": run_id,
                "instance_id": instance_id,
                "image_path": crop_path,
                "frame_path": frame_path,
                "video_id": meta.video_id if meta is not None else "",
                "orig_name": meta.orig_name if meta is not None and meta.orig_name is not None else "",
                "source_video": meta.source_video if meta is not None and meta.source_video is not None else "",
                "frame_idx": meta.frame_idx if meta is not None and meta.frame_idx is not None else "",
                "time_sec": time_sec if time_sec is not None else "",
                "encounter_id": encounter_id or "",
                "track_id": inst.get("track_id") if inst.get("track_id") is not None else "",
                "det_i": inst.get("det_i"),
                "bbox": json.dumps(inst.get("bbox"), separators=(",", ":")),
                "det_conf": inst.get("det_conf") if inst.get("det_conf") is not None else "",
                "cls": inst.get("cls") if inst.get("cls") is not None else "",
                "cls_name": inst.get("cls_name") if inst.get("cls_name") is not None else "",
                "flank_keep_prob": inst.get("flank_keep_prob") if inst.get("flank_keep_prob") is not None else "",
                "side_pred": inst.get("side_pred") if inst.get("side_pred") is not None else "",
                "side_pred_prob": inst.get("side_pred_prob") if inst.get("side_pred_prob") is not None else "",
                "recognizability_pred": inst.get("recognizability_pred") if inst.get("recognizability_pred") is not None else "",
                "recognizability_pred_prob": inst.get("recognizability_pred_prob") if inst.get("recognizability_pred_prob") is not None else "",
            }
        )

    stats = {
        "run_id": run_id,
        "instances": len(rows),
        "missing_frame_meta": missing_frame_meta,
    }
    return rows, stats


def write_index_csv(rows: list[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "instance_id",
        "image_path",
        "frame_path",
        "video_id",
        "orig_name",
        "source_video",
        "frame_idx",
        "time_sec",
        "encounter_id",
        "track_id",
        "det_i",
        "bbox",
        "det_conf",
        "cls",
        "cls_name",
        "flank_keep_prob",
        "side_pred",
        "side_pred_prob",
        "recognizability_pred",
        "recognizability_pred_prob",
    ]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def validate_rows(rows: list[dict]) -> None:
    seen: set[str] = set()
    dupes = 0
    for row in rows:
        iid = str(row.get("instance_id") or "")
        if not iid:
            raise ValueError("missing instance_id")
        if iid in seen:
            dupes += 1
        seen.add(iid)

    if dupes:
        raise ValueError(f"instance_id not unique (dupes={dupes})")


def discover_actions(runs_root: Path) -> list[tuple[str, Path, Path]]:
    triples: list[tuple[str, Path, Path]] = []
    for tracklets_actions in sorted(runs_root.rglob("04_tracklets/actions.json")):
        run_id = _infer_run_id(tracklets_actions)
        if run_id is None:
            continue
        frames_actions = tracklets_actions.parent.parent / "02_frames" / "actions.json"
        if not frames_actions.exists():
            continue
        triples.append((run_id, frames_actions, tracklets_actions))
    return triples


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--out-csv", default="data/reid/index.csv")
    parser.add_argument(
        "--encounter-gap-sec",
        type=float,
        default=300.0,
        help="Seconds used to bucket time_sec into encounter_id (<=0 disables).",
    )
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    actions = discover_actions(runs_root)
    if not actions:
        print("No actions.json found under", runs_root)
        return

    all_rows: list[dict] = []
    all_stats: list[dict] = []
    for run_id, frames_actions, tracklets_actions in actions:
        frames_meta = load_frames_meta(frames_actions)
        rows, stats = build_index_for_run(
            run_id=run_id,
            frames_meta=frames_meta,
            tracklets_actions=tracklets_actions,
            encounter_gap_sec=args.encounter_gap_sec,
        )
        all_rows.extend(rows)
        all_stats.append(stats)

    validate_rows(all_rows)
    out_csv = Path(args.out_csv)
    write_index_csv(all_rows, out_csv)

    total = len(all_rows)
    missing = sum(s["missing_frame_meta"] for s in all_stats)
    print("Wrote", total, "instances to", out_csv)
    print("Missing frame meta:", missing)
    for s in all_stats:
        print("-", s)


if __name__ == "__main__":
    main()
