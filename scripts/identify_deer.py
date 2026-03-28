from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from scripts.gallery_utils import (
        embed_single,
        load_gallery,
        load_megadescriptor,
        rank_matches,
    )
except ModuleNotFoundError:
    from gallery_utils import (
        embed_single,
        load_gallery,
        load_megadescriptor,
        rank_matches,
    )


def identify_image(
    image: Path,
    gallery_path: Path,
    side: str,
    top_k: int,
) -> list[dict[str, Any]]:
    if not image.exists():
        raise FileNotFoundError(f"Image not found: {image}")

    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    model, transform = load_megadescriptor()
    query_emb = embed_single(model, transform, image)
    gallery = load_gallery(gallery_path)
    return rank_matches(query_emb, gallery, query_side=side, top_k=top_k)


def format_ranked_results(matches: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for idx, row in enumerate(matches, start=1):
        deer_id = str(row.get("deer_id", "UNKNOWN"))
        side = str(row.get("side", "")).strip()
        confidence = float(row.get("confidence", 0.0))
        if deer_id == "UNKNOWN":
            lines.append(f"Rank {idx}: UNKNOWN")
            continue
        side_part = f" ({side})" if side else ""
        lines.append(
            f"Rank {idx}: {deer_id}{side_part} confidence: {round(confidence):.0f}%"
        )
    return lines


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--side", default="unknown", choices=["left", "right", "unknown"]
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--gallery", default="data/gallery/gallery.json")
    args = parser.parse_args()

    matches = identify_image(
        image=Path(args.image),
        gallery_path=Path(args.gallery),
        side=args.side,
        top_k=args.top_k,
    )
    for line in format_ranked_results(matches):
        print(line)


if __name__ == "__main__":
    main()
