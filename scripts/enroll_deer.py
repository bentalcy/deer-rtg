from __future__ import annotations

from pathlib import Path

try:
    from scripts.gallery_utils import (
        embed_single,
        load_gallery,
        load_megadescriptor,
        save_gallery,
    )
except ModuleNotFoundError:
    from gallery_utils import (
        embed_single,
        load_gallery,
        load_megadescriptor,
        save_gallery,
    )


def normalize_side(side: str) -> str:
    normalized = side.strip().lower()
    if normalized not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'")
    return normalized


def enroll_image(
    image: Path,
    deer_id: str,
    side: str,
    gallery_path: Path,
) -> dict[str, object]:
    if not image.exists():
        raise FileNotFoundError(f"Image not found: {image}")

    deer_id_clean = deer_id.strip()
    if not deer_id_clean:
        raise ValueError("deer_id cannot be empty")

    side_clean = normalize_side(side)

    gallery = load_gallery(gallery_path)
    deer_entry = gallery.setdefault(
        deer_id_clean,
        {
            "left": {"embeddings": [], "image_paths": []},
            "right": {"embeddings": [], "image_paths": []},
        },
    )
    side_entry = deer_entry.setdefault(
        side_clean, {"embeddings": [], "image_paths": []}
    )

    image_path = str(image)
    existing_paths = side_entry.setdefault("image_paths", [])
    if image_path in existing_paths:
        return {
            "deer_id": deer_id_clean,
            "side": side_clean,
            "image": image_path,
            "count_for_side": len(existing_paths),
            "gallery_path": str(gallery_path),
        }

    model, transform = load_megadescriptor()
    embedding = embed_single(model, transform, image)

    side_entry.setdefault("embeddings", []).append(embedding.tolist())
    existing_paths.append(image_path)

    save_gallery(gallery, gallery_path)

    return {
        "deer_id": deer_id_clean,
        "side": side_clean,
        "image": str(image),
        "count_for_side": len(side_entry["embeddings"]),
        "gallery_path": str(gallery_path),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--deer-id", required=True)
    parser.add_argument("--side", required=True, choices=["left", "right"])
    parser.add_argument("--gallery", default="data/gallery/gallery.json")
    args = parser.parse_args()

    result = enroll_image(
        image=Path(args.image),
        deer_id=args.deer_id,
        side=args.side,
        gallery_path=Path(args.gallery),
    )

    print(
        "Enrolled {deer_id} ({side}) from {image} into {gallery_path}. Total for side: {count_for_side}".format(
            **result
        )
    )


if __name__ == "__main__":
    main()
