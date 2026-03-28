import argparse
import csv
import json
import re
from pathlib import Path
from PIL import ExifTags

from PIL import Image
import matplotlib.pyplot as plt


KEY_TO_DECISION = {
    "s": "same",
    "d": "different",
    "e": "error_image",
    "m": "side_mismatch",
}


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def normalize_pair_row(row: dict[str, str]) -> dict[str, str]:
    cluster_id = (row.get("cluster_id") or row.get("cluster_a") or "").strip()
    prototype = (row.get("prototype_path") or row.get("image_path_a") or "").strip()
    question = (row.get("question_path") or row.get("image_path_b") or "").strip()
    if not cluster_id or not prototype or not question:
        raise ValueError(f"Invalid pair row: {row}")
    return {
        "cluster_id": cluster_id,
        "deer_id_likely": (row.get("deer_id_likely") or f"CLUSTER_{cluster_id}").strip(),
        "prototype_path": prototype,
        "prototype_path_2": (row.get("prototype_path_2") or "").strip(),
        "question_path": question,
    }


def load_existing_decisions(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    rows = load_rows(path)
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        cid = (row.get("cluster_id") or "").strip()
        if not cid:
            continue
        out[cid] = row
    return out


def save_decisions(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["cluster_id", "deer_id_likely", "prototype_path", "prototype_path_2", "question_path", "decision"],
        )
        writer.writeheader()
        writer.writerows(rows)


def to_abs(repo_root: Path, image_path: str) -> Path:
    p = Path(image_path)
    if p.is_absolute():
        return p
    return repo_root / p


def normalize_crop_basename(name: str) -> str:
    return re.sub(r"__dup\d+(?=(_det\d+\.[A-Za-z0-9]+)$)", "", name)


def build_source_lookup(repo_root: Path, actions_globs: list[str]) -> dict[str, Path]:
    lookup: dict[str, Path] = {}
    for pattern in actions_globs:
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
                frame_path = str(row.get("frame_path") or "").strip()
                dets = row.get("detections") or []
                if not frame_path or not isinstance(dets, list):
                    continue
                abs_frame = to_abs(repo_root, frame_path)
                for det in dets:
                    crop_new = str((det or {}).get("new_name") or "").strip()
                    if not crop_new:
                        continue
                    base = Path(crop_new).name
                    if base and base not in lookup:
                        lookup[base] = abs_frame
    return lookup


def _dms_to_deg(value):
    try:
        d = float(value[0][0]) / float(value[0][1])
        m = float(value[1][0]) / float(value[1][1])
        s = float(value[2][0]) / float(value[2][1])
        return d + (m / 60.0) + (s / 3600.0)
    except Exception:
        return None


def extract_image_meta(path: Path, source_frame_path: Path | None = None) -> dict[str, str]:
    out: dict[str, str] = {}
    meta_path = source_frame_path if source_frame_path is not None and source_frame_path.exists() else path
    try:
        img = Image.open(meta_path)
        exif = img.getexif()
        if not exif:
            raise ValueError("no exif")
        date_time = exif.get(306) or exif.get(36867)
        if date_time:
            out["datetime"] = str(date_time)

        gps_info = exif.get(34853)
        if gps_info:
            gps = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps_info.items()}
            lat = _dms_to_deg(gps.get("GPSLatitude")) if gps.get("GPSLatitude") else None
            lon = _dms_to_deg(gps.get("GPSLongitude")) if gps.get("GPSLongitude") else None
            if lat is not None and gps.get("GPSLatitudeRef") == "S":
                lat = -lat
            if lon is not None and gps.get("GPSLongitudeRef") == "W":
                lon = -lon
            if lat is not None and lon is not None:
                out["location"] = f"{lat:.6f}, {lon:.6f}"
    except Exception:
        pass

    if "datetime" not in out:
        m = re.search(r"(\d{4}-\d{2}-\d{2})[ _](\d{2}-\d{2}-\d{2})", path.name)
        if m:
            out["datetime"] = f"{m.group(1)} {m.group(2).replace('-', ':', 2).replace('-', ':', 1)}"

    if "datetime" not in out:
        out["datetime"] = "n/a"

    if "location" not in out:
        out["location"] = "n/a"

    if source_frame_path is not None:
        out["source_frame"] = str(source_frame_path)
    return out


def panel_title(base: str, path: Path, source_lookup: dict[str, Path]) -> str:
    key = path.name
    source = source_lookup.get(key)
    if source is None:
        source = source_lookup.get(normalize_crop_basename(key))
    meta = extract_image_meta(path, source_frame_path=source)
    lines = [base, path.name]
    lines.append(f"time: {meta['datetime']}")
    lines.append(f"location: {meta['location']}")
    if source is not None:
        lines.append(f"source: {source.name}")
    return "\n".join(lines)


def review_pair(
    repo_root: Path,
    source_lookup: dict[str, Path],
    pair: dict[str, str],
    current_index: int,
    total: int,
) -> str | None:
    left_path = to_abs(repo_root, pair["prototype_path"])
    left_path_2 = to_abs(repo_root, pair.get("prototype_path_2", "")) if pair.get("prototype_path_2") else None
    right_path = to_abs(repo_root, pair["question_path"])

    if not left_path.exists() or not right_path.exists():
        print(f"Skipping cluster {pair['cluster_id']}: missing image file")
        return "error_image"

    deer_id = pair.get("deer_id_likely", f"CLUSTER_{pair['cluster_id']}")
    left_img = Image.open(left_path).convert("RGB")
    left_img_2 = Image.open(left_path_2).convert("RGB") if left_path_2 is not None and left_path_2.exists() else None
    right_img = Image.open(right_path).convert("RGB")

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle(
        (
            f"Cluster {pair['cluster_id']} | Likely Deer ID: {deer_id}  ({current_index}/{total})\n"
            "Keys: [s]=same  [d]=different  [e]=error image  [m]=side mismatch  [q]=quit"
        ),
        fontsize=11,
    )
    axes[0].imshow(left_img)
    axes[0].set_title(panel_title("Prototype A (in cluster)", left_path, source_lookup), fontsize=9)
    axes[0].axis("off")

    if left_img_2 is not None and left_path_2 is not None:
        axes[1].imshow(left_img_2)
        axes[1].set_title(panel_title("Prototype B (in cluster)", left_path_2, source_lookup), fontsize=9)
    else:
        axes[1].text(0.5, 0.5, "No second prototype", ha="center", va="center")
        axes[1].set_title("Prototype B (if available)", fontsize=9)
    axes[1].axis("off")

    axes[2].imshow(right_img)
    axes[2].set_title(panel_title("Question (uncertain)", right_path, source_lookup), fontsize=9)
    axes[2].axis("off")

    picked: dict[str, str | None] = {"decision": None}

    def on_key(event):
        key = (event.key or "").lower()
        if key in KEY_TO_DECISION:
            picked["decision"] = KEY_TO_DECISION[key]
            plt.close(fig)
            return
        if key == "q":
            picked["decision"] = None
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.tight_layout()
    plt.show()
    return picked["decision"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pairs-csv",
        default="data/reid/unified_round5/pair_review_k10.csv",
    )
    parser.add_argument(
        "--decisions-csv",
        default="data/reid/unified_round5/pair_review_decisions.csv",
    )
    parser.add_argument(
        "--min-remaining-for-ui",
        type=int,
        default=10,
        help="If remaining pairs are below this, advise manual chat method and exit.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run UI even when remaining pairs are below min-remaining-for-ui.",
    )
    parser.add_argument(
        "--actions-globs",
        default="data/pool/**/actions.json,runs/**/04_tracklets/actions.json",
        help="Comma-separated glob patterns (repo-relative) for actions.json to recover source frame metadata.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    actions_globs = [x.strip() for x in args.actions_globs.split(",") if x.strip()]
    source_lookup = build_source_lookup(repo_root, actions_globs)
    pairs_csv = Path(args.pairs_csv)
    decisions_csv = Path(args.decisions_csv)

    pairs = [normalize_pair_row(r) for r in load_rows(pairs_csv)]
    existing = load_existing_decisions(decisions_csv)

    remaining = [p for p in pairs if p["cluster_id"] not in existing]
    if not remaining:
        print("No remaining pairs. Review is complete.")
        return

    if len(remaining) < args.min_remaining_for_ui and not args.force:
        print(
            f"Remaining pairs: {len(remaining)} (< {args.min_remaining_for_ui}). "
            "Keep current chat/manual method. Use --force to run UI anyway."
        )
        return

    ordered: list[dict[str, str]] = []
    for pair in pairs:
        cid = pair["cluster_id"]
        if cid in existing:
            ordered.append(
                {
                    "cluster_id": cid,
                    "deer_id_likely": pair.get("deer_id_likely", f"CLUSTER_{cid}"),
                    "prototype_path": pair["prototype_path"],
                    "prototype_path_2": pair.get("prototype_path_2", ""),
                    "question_path": pair["question_path"],
                    "decision": existing[cid].get("decision", ""),
                }
            )
        else:
            ordered.append(
                {
                    "cluster_id": cid,
                    "deer_id_likely": pair.get("deer_id_likely", f"CLUSTER_{cid}"),
                    "prototype_path": pair["prototype_path"],
                    "prototype_path_2": pair.get("prototype_path_2", ""),
                    "question_path": pair["question_path"],
                    "decision": "",
                }
            )

    total_remaining = sum(1 for row in ordered if not row["decision"])
    seen = 0
    for row in ordered:
        if row["decision"]:
            continue
        seen += 1
        decision = review_pair(repo_root, source_lookup, row, seen, total_remaining)
        if decision is None:
            print("Stopped by user.")
            break
        row["decision"] = decision
        save_decisions(decisions_csv, ordered)
        print(f"Saved: cluster {row['cluster_id']} -> {decision}")

    save_decisions(decisions_csv, ordered)
    done = sum(1 for row in ordered if row["decision"])
    print(f"Decisions saved to {decisions_csv}")
    print(f"Progress: {done}/{len(ordered)}")


if __name__ == "__main__":
    main()
