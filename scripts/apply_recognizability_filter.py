import argparse
import csv
import json
import shutil
from pathlib import Path


def main(
    scores_csv: Path,
    out_dir: Path,
    keep_label: str,
    min_prob: float,
    mode: str,
    out_manifest: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    rows = list(csv.DictReader(scores_csv.open(newline="")))
    if not rows:
        print("No rows found in", scores_csv)
        return

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    kept = 0
    skipped_missing = 0
    counts: dict[str, int] = {}
    kept_rows: list[dict[str, str]] = []

    for row in rows:
        pred = (row.get("pred") or "").strip()
        prob = float(row.get("pred_prob") or 0.0)
        counts[pred] = counts.get(pred, 0) + 1

        if pred != keep_label or prob < min_prob:
            continue

        src = Path((row.get("path") or "").strip())
        if not src.is_absolute():
            src = repo_root / src
        if not src.exists():
            skipped_missing += 1
            continue

        dst = out_dir / src.name
        if dst.exists():
            stem, suf = dst.stem, dst.suffix
            i = 1
            while (out_dir / f"{stem}__dup{i}{suf}").exists():
                i += 1
            dst = out_dir / f"{stem}__dup{i}{suf}"

        if mode == "copy":
            shutil.copy2(src, dst)
        else:
            dst.symlink_to(src.resolve())

        kept += 1
        kept_rows.append({"src": str(src), "dst": str(dst), "pred": pred, "pred_prob": f"{prob:.6f}"})

    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(
        json.dumps(
            {
                "scores_csv": str(scores_csv),
                "out_dir": str(out_dir),
                "keep_label": keep_label,
                "min_prob": min_prob,
                "mode": mode,
                "total_scored": len(rows),
                "pred_counts": counts,
                "kept": kept,
                "skipped_missing": skipped_missing,
                "kept_rows": kept_rows,
            },
            indent=2,
        )
    )
    print("Kept", kept, "images in", out_dir)
    print("Manifest:", out_manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores-csv", default="data/active_learning/recognizability_scores.csv")
    parser.add_argument("--out-dir", default="data/pool/recognizable_only")
    parser.add_argument("--keep-label", default="recognizable")
    parser.add_argument("--min-prob", type=float, default=0.5)
    parser.add_argument("--mode", choices=["copy", "symlink"], default="symlink")
    parser.add_argument("--out-manifest", default="data/pool/recognizable_only_manifest.json")
    args = parser.parse_args()

    main(
        scores_csv=Path(args.scores_csv),
        out_dir=Path(args.out_dir),
        keep_label=args.keep_label,
        min_prob=args.min_prob,
        mode=args.mode,
        out_manifest=Path(args.out_manifest),
    )
