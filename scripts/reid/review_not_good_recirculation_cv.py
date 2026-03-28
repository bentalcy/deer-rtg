import argparse
import csv
import shutil
import time
from pathlib import Path

import cv2
import numpy as np


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_decisions(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            p = (row.get("path") or "").strip()
            d = (row.get("decision") or "").strip()
            if p and d:
                out[p] = d
    return out


def save_decisions(path: Path, rows: list[dict[str, str]], decisions: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["timestamp", "path", "filename", "pred", "pred_prob", "reason", "decision"])
        w.writeheader()
        for row in rows:
            p = row["path"]
            d = decisions.get(p, "")
            if not d:
                continue
            w.writerow(
                {
                    "timestamp": int(time.time()),
                    "path": p,
                    "filename": row.get("filename", Path(p).name),
                    "pred": row.get("pred", ""),
                    "pred_prob": row.get("pred_prob", ""),
                    "reason": row.get("reason", ""),
                    "decision": d,
                }
            )


def export_kept(repo_root: Path, rows: list[dict[str, str]], decisions: dict[str, str], out_dir: Path, out_csv: Path) -> None:
    kept = [r for r in rows if decisions.get(r["path"]) == "keep"]
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[dict[str, str]] = []
    for i, row in enumerate(kept, start=1):
        src = repo_root / row["path"]
        if not src.exists():
            continue
        dst = out_dir / src.name
        if dst.exists():
            dst = out_dir / f"{i:04d}_{src.name}"
        shutil.copy2(src, dst)
        written.append(
            {
                "index": str(i),
                "src_path": row["path"],
                "dst_path": str(dst.relative_to(repo_root)),
                "pred": row.get("pred", ""),
                "pred_prob": row.get("pred_prob", ""),
                "reason": row.get("reason", ""),
            }
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["index", "src_path", "dst_path", "pred", "pred_prob", "reason"])
        w.writeheader()
        w.writerows(written)


def draw_ui(img, row: dict[str, str], idx: int, total: int, kept_count: int, skipped_count: int):
    h, w = 980, 1600
    if img is None:
        canvas = np.full((h, w, 3), 245, dtype="uint8")
    else:
        ih, iw = img.shape[:2]
        scale = min((w - 80) / max(1, iw), (h - 240) / max(1, ih))
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        frame = np.full((h, w, 3), 245, dtype="uint8")
        ox = (w - nw) // 2
        oy = 120
        frame[oy : oy + nh, ox : ox + nw] = resized
        canvas = frame

    cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), (220, 220, 220), 2)
    cv2.putText(canvas, f"Not-good Re-ID Review  {idx}/{total}", (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 40, 40), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"kept={kept_count} skipped={skipped_count}", (20, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"reason={row.get('reason','')} pred={row.get('pred','')} prob={row.get('pred_prob','')}",
        (20, 92),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (60, 60, 60),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(canvas, Path(row["path"]).name[:110], (20, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "Keys: [k]=keep  [space]=next/skip  [b]=back  [u]=undo prev  [q]=save+quit",
        (20, h - 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (30, 30, 120),
        1,
        cv2.LINE_AA,
    )
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", default="data/reid/unified_round5/not_good_for_reid_round2.csv")
    parser.add_argument("--decisions-csv", default="data/reid/unified_round5/not_good_for_reid_round2_decisions.csv")
    parser.add_argument("--out-dir", default="data/active_learning/recirculated_from_not_good_round2")
    parser.add_argument("--out-csv", default="data/active_learning/recirculated_from_not_good_round2.csv")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    rows = load_rows(Path(args.input_csv))
    decisions = load_decisions(Path(args.decisions_csv))

    idx = 0
    history: list[int] = []
    cv2.namedWindow("recirculation_review", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("recirculation_review", 1600, 980)

    while idx < len(rows):
        row = rows[idx]
        p = repo_root / row["path"]
        img = cv2.imread(str(p)) if p.exists() else None
        kept_count = sum(1 for d in decisions.values() if d == "keep")
        skipped_count = sum(1 for d in decisions.values() if d == "skip")
        canvas = draw_ui(img, row, idx + 1, len(rows), kept_count, skipped_count)
        cv2.imshow("recirculation_review", canvas)
        key = cv2.waitKey(0) & 0xFF

        if key == ord("q"):
            break
        if key == ord("k"):
            decisions[row["path"]] = "keep"
            history.append(idx)
            idx += 1
            continue
        if key == ord(" "):
            decisions[row["path"]] = "skip"
            history.append(idx)
            idx += 1
            continue
        if key == ord("b"):
            idx = max(0, idx - 1)
            continue
        if key == ord("u"):
            if history:
                prev = history.pop()
                prev_path = rows[prev]["path"]
                if prev_path in decisions:
                    del decisions[prev_path]
                idx = prev
            continue

    save_decisions(Path(args.decisions_csv), rows, decisions)
    export_kept(repo_root, rows, decisions, repo_root / args.out_dir, repo_root / args.out_csv)
    cv2.destroyAllWindows()
    print("saved_decisions", args.decisions_csv)
    print("saved_recirculation", args.out_csv)


if __name__ == "__main__":
    main()
