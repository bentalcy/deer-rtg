from __future__ import annotations

import csv
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

try:
    from scripts.enroll_deer import enroll_image
    from scripts.gallery_utils import load_gallery
except ModuleNotFoundError:
    from enroll_deer import enroll_image
    from gallery_utils import load_gallery


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGES_DIR = ROOT / "images"
DEFAULT_EMBEDDINGS_CSV = ROOT / "data/reid/embeddings.csv"
DEFAULT_EMBEDDINGS_PT = ROOT / "data/reid/embeddings.pt"
DEFAULT_INDEX_CSV = ROOT / "data/reid/index.csv"
DEFAULT_CLUSTERS_CSV = ROOT / "data/reid/clusters.csv"
DEFAULT_GALLERY_JSON = ROOT / "data/gallery/gallery.json"
UNKNOWN_DEER_ID = "__UNKNOWN__"


def _resolve_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def _load_gallery_raw(gallery_path: Path) -> dict[str, Any]:
    if not gallery_path.exists():
        return {}
    try:
        with gallery_path.open() as f:
            raw = json.load(f)
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    return raw


def _save_gallery_raw(gallery: dict[str, Any], gallery_path: Path) -> None:
    gallery_path.parent.mkdir(parents=True, exist_ok=True)
    with gallery_path.open("w") as f:
        json.dump(gallery, f, indent=2)


def _load_cluster_rows(clusters_csv: Path) -> list[tuple[str, str]]:
    if not clusters_csv.exists():
        return []
    rows: list[tuple[str, str]] = []
    with clusters_csv.open(newline="") as f:
        for raw in csv.DictReader(f):
            image_path = str(raw.get("image_path") or "").strip()
            cluster_id = str(raw.get("cluster_id") or "").strip()
            if image_path and cluster_id and cluster_id != "-1":
                rows.append((Path(image_path).name, cluster_id))
    return rows


def _build_cluster_suggestions(
    cluster_rows: list[tuple[str, str]],
    deer_by_basename: dict[str, str],
) -> tuple[dict[str, str], dict[str, str]]:
    fallback_by_name: dict[str, str] = {}
    names_by_cluster: dict[str, dict[str, int]] = {}

    for basename, cluster_id in cluster_rows:
        fallback_by_name[basename] = f"cluster-{cluster_id}"
        deer_id = deer_by_basename.get(basename)
        if not deer_id:
            continue
        cluster_map = names_by_cluster.setdefault(cluster_id, {})
        cluster_map[deer_id] = cluster_map.get(deer_id, 0) + 1

    cluster_winner: dict[str, str] = {}
    for cluster_id, counts in names_by_cluster.items():
        if not counts:
            continue
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0].lower()))
        if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
            continue
        cluster_winner[cluster_id] = ranked[0][0]

    preferred_by_name: dict[str, str] = {}
    for basename, cluster_id in cluster_rows:
        winner = cluster_winner.get(cluster_id)
        if winner:
            preferred_by_name[basename] = winner

    return preferred_by_name, fallback_by_name


def _normalize_side(value: str) -> str:
    side = value.strip().lower()
    if side in {"left", "right"}:
        return side
    return "unknown"


def _load_gallery_state(
    gallery_path: Path,
) -> tuple[
    set[str],
    dict[str, str],
    dict[str, str],
    list[str],
    dict[str, str],
    dict[str, str],
    dict[str, list[str]],
    dict[str, dict[str, list[list[float]]]],
]:
    gallery = _load_gallery_raw(gallery_path)
    enrolled_paths: set[str] = set()
    path_to_side: dict[str, str] = {}
    path_to_deer: dict[str, str] = {}
    basename_to_side: dict[str, str] = {}
    basename_to_deer: dict[str, str] = {}
    deer_to_images: dict[str, list[str]] = {}

    for deer_id, deer_data in gallery.items():
        if not isinstance(deer_data, dict):
            continue
        deer_id_str = str(deer_id)
        deer_to_images.setdefault(deer_id_str, [])
        for side in ("left", "right", "unknown"):
            side_data = deer_data.get(side, {})
            if not isinstance(side_data, dict):
                continue
            for path in side_data.get("image_paths", []):
                image_path = str(path)
                enrolled_paths.add(image_path)
                path_to_side[image_path] = side
                path_to_deer[image_path] = deer_id_str
                base = Path(image_path).name
                basename_to_side.setdefault(base, side)
                basename_to_deer.setdefault(base, deer_id_str)
                if image_path not in deer_to_images[deer_id_str]:
                    deer_to_images[deer_id_str].append(image_path)

    deer_ids = sorted(
        (str(deer_id) for deer_id in gallery.keys() if str(deer_id) != UNKNOWN_DEER_ID),
        key=str.lower,
    )
    return (
        enrolled_paths,
        path_to_side,
        path_to_deer,
        deer_ids,
        basename_to_side,
        basename_to_deer,
        deer_to_images,
        gallery,
    )


def _load_side_predictions(
    index_csv: Path | None,
) -> tuple[dict[str, str], dict[str, str]]:
    if index_csv is None or not index_csv.exists():
        return {}, {}

    side_by_path: dict[str, str] = {}
    side_by_basename: dict[str, str] = {}
    with index_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            image_path = str(row.get("image_path") or "").strip()
            if not image_path:
                continue
            side = _normalize_side(str(row.get("side_pred") or ""))
            if side == "unknown":
                continue
            side_by_path[image_path] = side
            side_by_basename.setdefault(Path(image_path).name, side)
    return side_by_path, side_by_basename


_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def _has_image_files(images_dir: Path | None) -> bool:
    if images_dir is None or not images_dir.exists():
        return False
    for p in images_dir.rglob("*"):
        if p.suffix.lower() in _IMAGE_EXTS:
            return True
    return False


def _resolve_side(
    image_path: str,
    side_from_gallery: dict[str, str],
    side_from_gallery_basename: dict[str, str],
    side_from_index: dict[str, str],
    side_from_index_basename: dict[str, str],
    model_side: str = "unknown",
) -> str:
    basename = Path(image_path).name
    if image_path in side_from_gallery:
        return side_from_gallery[image_path]
    if basename in side_from_gallery_basename:
        return side_from_gallery_basename[basename]
    if model_side in {"left", "right"}:
        return model_side
    if image_path in side_from_index:
        return side_from_index[image_path]
    if basename in side_from_index_basename:
        return side_from_index_basename[basename]
    return "unknown"


def _resolve_deer_prefill(
    image_path: str,
    deer_from_gallery: dict[str, str],
    deer_from_gallery_basename: dict[str, str],
) -> str:
    if image_path in deer_from_gallery:
        return deer_from_gallery[image_path]
    return deer_from_gallery_basename.get(Path(image_path).name, "")


def _is_already_enrolled(
    image_path: str,
    enrolled: set[str],
    deer_from_gallery_basename: dict[str, str],
) -> bool:
    if image_path in enrolled:
        return True
    return Path(image_path).name in deer_from_gallery_basename


def _load_query_embeddings(
    embeddings_csv: Path | None,
    embeddings_pt: Path | None,
) -> dict[str, Any]:
    if (
        embeddings_csv is None
        or embeddings_pt is None
        or not embeddings_csv.exists()
        or not embeddings_pt.exists()
    ):
        return {}

    image_paths: list[str] = []
    with embeddings_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            image_path = str(row.get("image_path") or "").strip()
            if image_path:
                image_paths.append(image_path)

    if not image_paths:
        return {}

    try:
        import numpy as np
        import torch

        raw = torch.load(embeddings_pt, map_location="cpu")
        if isinstance(raw, torch.Tensor):
            arr = raw.detach().cpu().numpy()
        else:
            arr = np.asarray(raw)
        if arr.ndim != 2 or arr.shape[0] != len(image_paths):
            return {}
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        arr = arr / norms
        return {image_paths[i]: arr[i] for i in range(len(image_paths))}
    except Exception:
        return {}


def _build_gallery_match_rows(
    gallery: dict[str, dict[str, dict[str, list[object]]]],
) -> list[dict[str, object]]:
    try:
        import numpy as np
    except Exception:
        return []

    rows: list[dict[str, object]] = []
    for deer_id, deer_data in gallery.items():
        if str(deer_id) == UNKNOWN_DEER_ID:
            continue
        if not isinstance(deer_data, dict):
            continue
        for side in ("left", "right"):
            side_data = deer_data.get(side, {})
            if not isinstance(side_data, dict):
                continue
            emb_list = side_data.get("embeddings", [])
            if not emb_list:
                continue
            arr = np.asarray(emb_list, dtype="float32")
            if arr.ndim != 2:
                continue
            proto = arr.mean(axis=0)
            norm = float(np.linalg.norm(proto))
            if norm == 0.0:
                continue
            proto = proto / norm
            refs = [str(p) for p in side_data.get("image_paths", [])]
            rows.append(
                {
                    "deer_id": str(deer_id),
                    "side": side,
                    "prototype": proto,
                    "reference_images": refs,
                }
            )
    return rows


def _build_likely_matches(
    image_path: str,
    side_prefill: str,
    query_embeddings: dict[str, Any],
    gallery_match_rows: list[dict[str, object]],
    deer_to_images: dict[str, list[str]],
    preferred_cluster_name: str,
) -> tuple[list[dict[str, object]], str]:
    try:
        import numpy as np
    except Exception:
        return [], ""

    emb = query_embeddings.get(image_path)
    if emb is None:
        if preferred_cluster_name and preferred_cluster_name in deer_to_images:
            refs: list[str] = []
            seen_bases: set[str] = set()
            for raw_ref in deer_to_images.get(preferred_cluster_name, []):
                ref = str(raw_ref)
                base = Path(ref).name
                if base in seen_bases:
                    continue
                seen_bases.add(base)
                refs.append(ref)
                if len(refs) >= 4:
                    break
            fallback = [
                {
                    "deer_id": preferred_cluster_name,
                    "side": "",
                    "score_pct": None,
                    "image_path": ref,
                }
                for ref in refs
            ]
            return (
                fallback,
                f"This photo is in a cluster that was previously labeled as {preferred_cluster_name}.",
            )

        fallback_rows: list[dict[str, object]] = []
        first_deer = next(iter(sorted(deer_to_images.keys(), key=str.lower)), "")
        if first_deer:
            refs: list[str] = []
            seen_bases: set[str] = set()
            for raw_ref in deer_to_images.get(first_deer, []):
                ref = str(raw_ref)
                base = Path(ref).name
                if base in seen_bases:
                    continue
                seen_bases.add(base)
                refs.append(ref)
                if len(refs) >= 4:
                    break
            fallback_rows = [
                {
                    "deer_id": first_deer,
                    "side": "",
                    "score_pct": None,
                    "image_path": str(ref),
                }
                for ref in refs
            ]
        if fallback_rows:
            return (
                fallback_rows,
                "Do we think this is the same as one of the previously labeled deer?",
            )
        return [], ""

    scored: list[dict[str, object]] = []
    for row in gallery_match_rows:
        row_side = str(row.get("side") or "")
        if side_prefill in {"left", "right"} and row_side not in {side_prefill, ""}:
            continue
        prototype = row.get("prototype")
        if prototype is None:
            continue
        score = float(np.dot(emb, prototype))
        scored.append(
            {
                "deer_id": str(row.get("deer_id") or ""),
                "side": row_side,
                "score": score,
                "reference_images": row.get("reference_images") or [],
            }
        )

    if not scored:
        return [], ""

    scored.sort(key=lambda r: float(r["score"]), reverse=True)
    best_per_deer: dict[str, dict[str, object]] = {}
    for row in scored:
        deer_id = str(row["deer_id"])
        if not deer_id:
            continue
        if deer_id not in best_per_deer:
            best_per_deer[deer_id] = row

    top_deer_rows = sorted(
        best_per_deer.values(), key=lambda r: float(r["score"]), reverse=True
    )
    if not top_deer_rows:
        return [], ""

    top_deer = top_deer_rows[0]
    top_refs: list[str] = []
    seen_bases: set[str] = set()
    for raw_ref in top_deer.get("reference_images") or []:
        ref = str(raw_ref)
        if not ref:
            continue
        base = Path(ref).name
        if base in seen_bases:
            continue
        seen_bases.add(base)
        top_refs.append(ref)
    # Prefer 2-4 examples from the most likely already-labeled deer.
    if len(top_refs) < 2:
        for candidate in top_deer_rows[1:]:
            for p in candidate.get("reference_images") or []:
                p_str = str(p)
                base = Path(p_str).name
                if p_str and base not in seen_bases:
                    seen_bases.add(base)
                    top_refs.append(p_str)
                if len(top_refs) >= 4:
                    break
            if len(top_refs) >= 4:
                break
    top_refs = top_refs[:4]

    matches: list[dict[str, object]] = []
    top_score_pct = round((float(top_deer["score"]) + 1.0) / 2.0 * 100.0, 1)
    for ref in top_refs:
        matches.append(
            {
                "deer_id": str(top_deer["deer_id"]),
                "side": str(top_deer.get("side") or ""),
                "score_pct": top_score_pct,
                "image_path": str(ref),
            }
        )

    hint = ""
    if matches:
        top_match = matches[0]
        top_score = float(top_match.get("score_pct") or 0.0)
        if top_score >= 80.0:
            hint = (
                f"Do we think this is the same as previous deer {top_match['deer_id']}? "
                f"Strong match ({top_score:.0f}%)."
            )
        elif top_score >= 70.0:
            hint = (
                f"Do we think this is the same as previous deer {top_match['deer_id']}? "
                f"Possible match ({top_score:.0f}%)."
            )
        else:
            hint = "Not seeing a strong match with previously labeled deer."

    return matches, hint


def build_items(
    gallery_path: Path,
    embeddings_csv: Path | None = None,
    images_dir: Path | None = None,
    clusters_csv: Path | None = None,
    index_csv: Path | None = DEFAULT_INDEX_CSV,
    embeddings_pt: Path | None = None,
) -> list[dict[str, Any]]:
    (
        enrolled,
        side_from_gallery,
        deer_from_gallery,
        deer_ids,
        side_from_gallery_basename,
        deer_from_gallery_basename,
        deer_to_images,
        gallery,
    ) = _load_gallery_state(gallery_path)
    deer_by_basename = {
        Path(path).name: deer for path, deer in deer_from_gallery.items()
    }
    cluster_rows = _load_cluster_rows(clusters_csv) if clusters_csv else []
    preferred_cluster_names, cluster_fallback = _build_cluster_suggestions(
        cluster_rows, deer_by_basename
    )
    side_from_index, side_from_index_basename = _load_side_predictions(index_csv)
    query_embeddings = _load_query_embeddings(embeddings_csv, embeddings_pt)
    gallery_match_rows = _build_gallery_match_rows(gallery)
    items: list[dict[str, Any]] = []

    if images_dir is not None and images_dir.exists():
        for p in sorted(images_dir.rglob("*")):
            if p.suffix.lower() not in _IMAGE_EXTS:
                continue
            image_path = str(p)
            basename = p.name
            name_suggestion = preferred_cluster_names.get(
                basename, cluster_fallback.get(basename, "")
            )
            side_prefill = _resolve_side(
                image_path=image_path,
                side_from_gallery=side_from_gallery,
                side_from_gallery_basename=side_from_gallery_basename,
                side_from_index=side_from_index,
                side_from_index_basename=side_from_index_basename,
            )
            likely_matches, same_hint = _build_likely_matches(
                image_path=image_path,
                side_prefill=side_prefill,
                query_embeddings=query_embeddings,
                gallery_match_rows=gallery_match_rows,
                deer_to_images=deer_to_images,
                preferred_cluster_name=preferred_cluster_names.get(basename, ""),
            )
            items.append(
                {
                    "image_path": image_path,
                    "side_prefill": side_prefill,
                    "already_enrolled": _is_already_enrolled(
                        image_path, enrolled, deer_from_gallery_basename
                    ),
                    "name_suggestion": (
                        name_suggestion
                        or (
                            str(likely_matches[0].get("deer_id") or "")
                            if likely_matches
                            else ""
                        )
                    ),
                    "deer_prefill": _resolve_deer_prefill(
                        image_path, deer_from_gallery, deer_from_gallery_basename
                    ),
                    "deer_options": deer_ids,
                    "same_deer_hint": same_hint,
                    "likely_matches": likely_matches,
                }
            )
    elif embeddings_csv is not None and embeddings_csv.exists():
        with embeddings_csv.open(newline="") as f:
            for row in csv.DictReader(f):
                image_path = str(row.get("image_path") or "").strip()
                if not image_path:
                    continue
                basename = Path(image_path).name
                model_side = _normalize_side(str(row.get("side_pred") or ""))
                side_prefill = _resolve_side(
                    image_path=image_path,
                    side_from_gallery=side_from_gallery,
                    side_from_gallery_basename=side_from_gallery_basename,
                    side_from_index=side_from_index,
                    side_from_index_basename=side_from_index_basename,
                    model_side=model_side,
                )
                name_suggestion = preferred_cluster_names.get(
                    basename, cluster_fallback.get(basename, "")
                )
                likely_matches, same_hint = _build_likely_matches(
                    image_path=image_path,
                    side_prefill=side_prefill,
                    query_embeddings=query_embeddings,
                    gallery_match_rows=gallery_match_rows,
                    deer_to_images=deer_to_images,
                    preferred_cluster_name=preferred_cluster_names.get(basename, ""),
                )
                items.append(
                    {
                        "image_path": image_path,
                        "side_prefill": side_prefill,
                        "already_enrolled": _is_already_enrolled(
                            image_path, enrolled, deer_from_gallery_basename
                        ),
                        "name_suggestion": (
                            name_suggestion
                            or (
                                str(likely_matches[0].get("deer_id") or "")
                                if likely_matches
                                else ""
                            )
                        ),
                        "deer_prefill": _resolve_deer_prefill(
                            image_path, deer_from_gallery, deer_from_gallery_basename
                        ),
                        "deer_options": deer_ids,
                        "same_deer_hint": same_hint,
                        "likely_matches": likely_matches,
                    }
                )
    return items


def handle_enroll_payload(
    payload: dict[str, Any],
    gallery_path: Path,
    enroll_fn: Callable[[Path, str, str, Path], dict[str, object]] = enroll_image,
) -> dict[str, object]:
    image_path = str(payload.get("image_path") or "").strip()
    deer_id = str(payload.get("deer_id") or "").strip()
    side = str(payload.get("side") or "").strip().lower()

    if not image_path or not deer_id or side not in {"left", "right"}:
        raise ValueError(
            "payload must include image_path, deer_id, and side (left/right)"
        )

    return enroll_fn(Path(image_path), deer_id, side, gallery_path)


def handle_bulk_enroll_payload(
    payload: dict[str, Any],
    gallery_path: Path,
    enroll_fn: Callable[[Path, str, str, Path], dict[str, object]] = enroll_image,
) -> dict[str, int]:
    rows = payload.get("items", [])
    if not isinstance(rows, list):
        raise ValueError("items must be a list")

    enrolled = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        image_path = str(row.get("image_path") or "").strip()
        deer_id = str(row.get("deer_id") or "").strip()
        side = str(row.get("side") or "").strip().lower()
        if not image_path or not deer_id or side not in {"left", "right"}:
            continue
        enroll_fn(Path(image_path), deer_id, side, gallery_path)
        enrolled += 1

    return {"requested": len(rows), "enrolled": enrolled}


def _remove_image_from_gallery(image_path: str, gallery_path: Path) -> bool:
    gallery = _load_gallery_raw(gallery_path)
    removed = False

    for deer_data in gallery.values():
        if not isinstance(deer_data, dict):
            continue
        for side in ("left", "right", "unknown"):
            side_data = deer_data.get(side, {})
            if not isinstance(side_data, dict):
                continue
            paths = side_data.get("image_paths", [])
            embeddings = side_data.get("embeddings", [])
            if not isinstance(paths, list) or not isinstance(embeddings, list):
                continue
            keep_paths: list[str] = []
            keep_embeddings: list[object] = []
            for idx, path in enumerate(paths):
                path_str = str(path)
                if path_str == image_path:
                    removed = True
                    continue
                keep_paths.append(path_str)
                if idx < len(embeddings):
                    keep_embeddings.append(embeddings[idx])
            if removed:
                side_data["image_paths"] = keep_paths
                side_data["embeddings"] = keep_embeddings

    if removed:
        _save_gallery_raw(gallery, gallery_path)
    return removed


def handle_label_upsert_payload(
    payload: dict[str, Any],
    gallery_path: Path,
    enroll_fn: Callable[[Path, str, str, Path], dict[str, object]] = enroll_image,
) -> dict[str, object]:
    image_path = str(payload.get("image_path") or "").strip()
    deer_id = str(payload.get("deer_id") or "").strip()
    side = str(payload.get("side") or "").strip().lower()

    if not image_path or not deer_id or side not in {"left", "right", "unknown"}:
        raise ValueError(
            "payload must include image_path, deer_id, and side (left/right/unknown)"
        )

    _remove_image_from_gallery(image_path, gallery_path)
    if side == "unknown":
        gallery = _load_gallery_raw(gallery_path)
        deer_entry = gallery.setdefault(
            deer_id,
            {
                "left": {"embeddings": [], "image_paths": []},
                "right": {"embeddings": [], "image_paths": []},
                "unknown": {"embeddings": [], "image_paths": []},
            },
        )
        unknown_entry = deer_entry.setdefault(
            "unknown", {"embeddings": [], "image_paths": []}
        )
        paths = unknown_entry.setdefault("image_paths", [])
        if image_path not in paths:
            paths.append(image_path)
        unknown_entry.setdefault("embeddings", [])
        _save_gallery_raw(gallery, gallery_path)
        return {
            "deer_id": deer_id,
            "side": "unknown",
            "image": image_path,
            "count_for_side": len(paths),
            "gallery_path": str(gallery_path),
        }
    preserved_unknown = {}
    gallery_before_left_right = _load_gallery_raw(gallery_path)
    for deer_key, deer_data in gallery_before_left_right.items():
        if not isinstance(deer_data, dict):
            continue
        unknown_data = deer_data.get("unknown", {})
        if not isinstance(unknown_data, dict):
            continue
        unknown_paths = [str(p) for p in unknown_data.get("image_paths", []) if str(p)]
        if unknown_paths:
            preserved_unknown[str(deer_key)] = {
                "image_paths": unknown_paths,
                "embeddings": list(unknown_data.get("embeddings", [])),
            }

    result = enroll_fn(Path(image_path), deer_id, side, gallery_path)

    if preserved_unknown:
        gallery_after = _load_gallery_raw(gallery_path)
        for deer_key, unknown_data in preserved_unknown.items():
            deer_entry = gallery_after.setdefault(
                deer_key,
                {
                    "left": {"embeddings": [], "image_paths": []},
                    "right": {"embeddings": [], "image_paths": []},
                },
            )
            if isinstance(deer_entry, dict):
                deer_entry["unknown"] = unknown_data
        _save_gallery_raw(gallery_after, gallery_path)

    return result


def handle_label_delete_payload(
    payload: dict[str, Any],
    gallery_path: Path,
) -> dict[str, object]:
    image_path = str(payload.get("image_path") or "").strip()
    if not image_path:
        raise ValueError("payload must include image_path")
    removed = _remove_image_from_gallery(image_path, gallery_path)
    return {"image_path": image_path, "deleted": removed}


INDEX_HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Label Deer Photos</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    :root {
      --bg: #f2efe9;
      --panel: #ffffff;
      --line: #d8d1c4;
      --text: #1f1d18;
      --muted: #6e685c;
      --brand: #1f6c45;
      --brand-soft: #e4f2ea;
      --warn: #c38111;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Avenir Next", Avenir, "Segoe UI", sans-serif;
      background: radial-gradient(circle at 20% -20%, #fef7eb, var(--bg) 55%);
      color: var(--text);
      min-height: 100vh;
      padding-bottom: 220px;
    }
    .progress-wrap {
      position: sticky;
      top: 0;
      z-index: 20;
      padding: 12px 14px;
      background: #f7f3ec;
      border-bottom: 1px solid var(--line);
      backdrop-filter: blur(4px);
    }
    .progress-head {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: center;
      font-size: 14px;
      margin-bottom: 8px;
    }
    .progress-main {
      font-weight: 700;
      font-size: 15px;
    }
    .deer-top-label {
      margin-top: 6px;
      font-size: 14px;
      font-weight: 700;
      color: #2f2a22;
    }
    .progress-track {
      height: 12px;
      border-radius: 999px;
      border: 1px solid #cec6b7;
      background: #ede6d8;
      overflow: hidden;
    }
    .progress-fill {
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, #2d8b57, #1f6c45);
      transition: width 180ms ease;
    }
    .top-actions {
      margin-top: 10px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }
    .start-at {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 13px;
      color: var(--muted);
    }
    .start-at input {
      width: 76px;
      min-height: 36px;
      border-radius: 10px;
      border: 1px solid #c7c0b3;
      padding: 4px 8px;
      font-size: 14px;
    }
    button {
      min-height: 48px;
      border-radius: 12px;
      border: 1px solid #c7c0b3;
      background: var(--panel);
      color: var(--text);
      cursor: pointer;
      padding: 10px 14px;
      font-size: 16px;
      font-weight: 600;
    }
    button.primary {
      background: var(--brand);
      color: #fff;
      border-color: var(--brand);
    }
    button:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }
    .content {
      max-width: 880px;
      margin: 0 auto;
      padding: 16px;
    }
    .title { margin: 0 0 8px; font-size: 24px; }
    .instruction {
      margin: 0 0 14px;
      color: var(--muted);
      font-size: 16px;
    }
    .photo-card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 10px;
      box-shadow: 0 8px 30px rgba(31, 29, 24, 0.08);
    }
    .photo-card img {
      width: 100%;
      max-height: 62vh;
      object-fit: contain;
      border-radius: 12px;
      background: #ebe4d6;
    }
    .labeled-chip {
      margin-top: 10px;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      border: 1px solid #9bc9ad;
      background: #ebf8ef;
      color: #205838;
      font-size: 14px;
      border-radius: 999px;
      padding: 4px 10px;
      font-weight: 600;
    }
    .name-area {
      margin-top: 14px;
      display: grid;
      gap: 8px;
    }
    .deer-quick {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .deer-quick button {
      min-height: 40px;
      font-size: 14px;
      padding: 8px 10px;
    }
    .suggestion {
      margin-top: 12px;
      border: 1px solid #cfe1d5;
      background: #f2faf5;
      border-radius: 12px;
      padding: 10px;
    }
    .suggestion strong {
      display: block;
      margin-bottom: 4px;
      font-size: 14px;
      color: #184f34;
    }
    .suggestion .suggestion-name {
      font-size: 16px;
      font-weight: 700;
      color: #184f34;
      cursor: pointer;
      text-decoration: underline;
      text-decoration-thickness: 1px;
      text-underline-offset: 2px;
    }
    .same-check {
      margin-top: 12px;
      border: 1px solid #d7d0c2;
      background: #faf8f3;
      border-radius: 12px;
      padding: 10px;
    }
    .same-check .question {
      font-size: 14px;
      font-weight: 700;
      margin-bottom: 8px;
      color: #2b271f;
    }
    .close-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }
    .close-card {
      border: 1px solid #ddd4c5;
      border-radius: 10px;
      background: #fff;
      overflow: hidden;
      cursor: pointer;
    }
    .close-card img {
      width: 100%;
      height: 84px;
      object-fit: cover;
      border-radius: 0;
      background: #e8e0d0;
    }
    .close-card .meta {
      padding: 6px 8px;
      font-size: 12px;
      line-height: 1.2;
      color: #2e2a22;
    }
    .name-area select,
    .name-area input[type=text] {
      width: 100%;
      min-height: 48px;
      border-radius: 12px;
      border: 1px solid #cbc4b6;
      font-size: 17px;
      padding: 10px 12px;
      background: #fff;
    }
    .muted { color: var(--muted); font-size: 14px; }
    .empty {
      text-align: center;
      color: var(--muted);
      margin-top: 40px;
      font-size: 18px;
    }
    .dock {
      position: fixed;
      left: 0;
      right: 0;
      bottom: 0;
      z-index: 30;
      background: rgba(247, 243, 236, 0.98);
      border-top: 1px solid var(--line);
      padding: 12px;
    }
    .dock-inner {
      max-width: 900px;
      margin: 0 auto;
      display: grid;
      gap: 10px;
    }
    .side-row {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
    }
    .side-btn {
      font-size: 15px;
      line-height: 1.2;
      font-weight: 700;
      white-space: normal;
    }
    .side-btn.active {
      background: var(--brand-soft);
      border-color: var(--brand);
      color: #184f34;
    }
    .side-btn.unknown.active {
      background: #fff4dc;
      border-color: var(--warn);
      color: #6a4b08;
    }
    .save-row { display: grid; }
    .nav-row {
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 8px;
    }
    .danger {
      border-color: #cb7a7a;
      color: #8b1f1f;
      background: #fff4f4;
    }
    .warn {
      border-color: #c9a96a;
      color: #6e4d12;
      background: #fff7e7;
    }
    .review-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
      gap: 10px;
      margin-top: 10px;
    }
    .review-card {
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 8px;
    }
    .review-card img {
      width: 100%;
      height: 130px;
      object-fit: cover;
      border-radius: 8px;
      background: #ebe4d6;
    }
    .review-meta {
      margin-top: 6px;
      font-size: 13px;
      color: #3f3a31;
    }
    .toast {
      position: fixed;
      left: 12px;
      right: 12px;
      bottom: 180px;
      z-index: 40;
      background: #1f1d18;
      color: #fff;
      border-radius: 12px;
      padding: 10px 12px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
    }
    .toast button {
      min-height: 40px;
      padding: 8px 12px;
      background: transparent;
      border-color: #fff;
      color: #fff;
      font-size: 14px;
    }
    @media (max-width: 640px) {
      .title { font-size: 22px; }
      .content { padding: 12px; }
      .photo-card img { max-height: 48vh; }
      .side-btn { font-size: 14px; padding: 8px 6px; }
    }
  </style>
</head>
<body>
  <div class="progress-wrap">
    <div class="progress-head">
      <strong id="progressText" class="progress-main">0 labeled</strong>
      <span class="muted" id="queueText"></span>
    </div>
    <div id="deerTopLabel" class="deer-top-label">Deer: not selected</div>
    <div class="progress-track"><div class="progress-fill" id="progressFill"></div></div>
    <div class="top-actions">
      <div class="start-at">
        <span>Start at photo #</span>
        <input id="startIndexInput" type="number" min="1" step="1" />
        <button type="button" onclick="jumpToPhotoNumber()">Jump</button>
      </div>
      <button id="reviewToggle" onclick="toggleReviewMode()">Review labeled photos</button>
    </div>
  </div>

  <div class="content">
    <h1 class="title">Label Deer Photos</h1>
    <div class="muted" id="buildStamp">UI build: 2026-03-28m</div>
    <p class="instruction"><em>Do you recognise this deer? Pick its name and which side you can see, then tap Save.</em></p>
    <div id="main"></div>
  </div>

  <div class="dock" id="dock">
    <div class="dock-inner">
      <div class="side-row">
        <button class="side-btn" id="side-left" onclick="setSide('left')">◀ 🦌<br/>Left side visible</button>
        <button class="side-btn" id="side-right" onclick="setSide('right')">🦌 ▶<br/>Right side visible</button>
        <button class="side-btn unknown" id="side-unknown" onclick="setSide('unknown')">?<br/>Can't tell</button>
      </div>
      <div class="save-row">
        <button class="primary" onclick="saveAndNext()">Save &amp; Next</button>
      </div>
      <div class="nav-row">
        <button onclick="goPrev()">Previous</button>
        <button onclick="goNext()">Next</button>
        <button class="warn" onclick="hideCurrentImage()">Hide Photo</button>
      </div>
      <div class="nav-row">
        <button onclick="showAllHidden()">Show Hidden</button>
        <button class="danger" onclick="deleteLabelForCurrent()">Delete Label</button>
        <span></span>
      </div>
    </div>
  </div>

  <div class="toast" id="toast" style="display:none;">
    <span id="toastText"></span>
    <button id="undoBtn" style="display:none;" onclick="undoLastAction()">Undo</button>
  </div>

<script>
let items = [];
let currentIndex = 0;
let reviewMode = false;
let selectedSide = 'unknown';
let activeToastActionId = null;
const UNKNOWN_DEER_ID = '__UNKNOWN__';
const pendingActions = new Map();
const labelsByImage = new Map();
const RESUME_INDEX_KEY = 'enrollment_ui_current_index';
const HIDDEN_IMAGES_KEY = 'enrollment_ui_hidden_images';
let hiddenImagePaths = new Set();

function escapeHtml(value) {
  return String(value || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/\"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function escapeJsSingle(value) {
  return String(value || '')
    .split('\\\\').join('\\\\\\\\')
    .split("'").join("\\\\'");
}

function deerDisplayName(deerId) {
  return deerId === UNKNOWN_DEER_ID ? 'Unknown deer' : deerId;
}

window.addEventListener('error', function(event) {
  const message = event && event.message ? event.message : 'Unknown script error';
  const target = document.getElementById('main');
  if (target) {
    target.innerHTML = `<div class="empty">UI script error: ${escapeHtml(message)}</div>`;
  }
});

window.addEventListener('unhandledrejection', function(event) {
  const reason = event && event.reason ? String(event.reason) : 'Unknown async error';
  const target = document.getElementById('main');
  if (target) {
    target.innerHTML = `<div class="empty">UI async error: ${escapeHtml(reason)}</div>`;
  }
});

async function loadItems() {
  if (typeof fetch !== 'function') {
    document.getElementById('main').innerHTML = '<div class="empty">This browser is too old for this page. Please use a newer Safari/Chrome.</div>';
    return;
  }
  const resp = await fetch('/items');
  items = await resp.json();
  loadHiddenImages();
  currentIndex = getResumeIndex(getVisibleItems().length);
  reviewMode = false;

  for (const item of items) {
    if (item.already_enrolled && item.deer_prefill) {
      labelsByImage.set(item.image_path, {
        deer_id: item.deer_prefill,
        side: item.side_prefill,
        state: 'committed',
      });
    }
  }

  render();
}

function loadHiddenImages() {
  hiddenImagePaths = new Set();
  try {
    const raw = window.localStorage.getItem(HIDDEN_IMAGES_KEY);
    if (!raw) {
      return;
    }
    const arr = JSON.parse(raw);
    if (!Array.isArray(arr)) {
      return;
    }
    for (const path of arr) {
      if (typeof path === 'string' && path) {
        hiddenImagePaths.add(path);
      }
    }
  } catch (e) {
    hiddenImagePaths = new Set();
  }
}

function persistHiddenImages() {
  try {
    const arr = Array.from(hiddenImagePaths.values());
    window.localStorage.setItem(HIDDEN_IMAGES_KEY, JSON.stringify(arr));
  } catch (e) {
    // ignore storage failures
  }
}

function getVisibleItems() {
  return items.filter((item) => !hiddenImagePaths.has(item.image_path));
}

function getResumeIndex(total) {
  if (!total) {
    return 0;
  }
  let idx = 0;
  try {
    const saved = window.localStorage.getItem(RESUME_INDEX_KEY);
    if (saved) {
      const parsed = parseInt(saved, 10);
      if (!Number.isNaN(parsed)) {
        idx = parsed;
      }
    }
  } catch (e) {
    idx = 0;
  }
  if (idx < 0) {
    idx = 0;
  }
  if (idx >= total) {
    idx = total - 1;
  }
  return idx;
}

function persistCurrentIndex() {
  try {
    window.localStorage.setItem(RESUME_INDEX_KEY, String(currentIndex));
  } catch (e) {
    // ignore storage failures
  }
}

function jumpToPhotoNumber() {
  const visible = getVisibleItems();
  if (!visible.length) {
    return;
  }
  const input = document.getElementById('startIndexInput');
  if (!input) {
    return;
  }
  const raw = parseInt(input.value || '1', 10);
  if (Number.isNaN(raw)) {
    return;
  }
  const clamped = Math.min(visible.length, Math.max(1, raw));
  currentIndex = clamped - 1;
  render();
}

function hideCurrentImage() {
  const item = currentItem();
  if (!item) {
    return;
  }
  hiddenImagePaths.add(item.image_path);
  persistHiddenImages();
  showToast('Photo hidden from queue.', { timeoutMs: 1500 });
  const visibleCount = getVisibleItems().length;
  if (currentIndex >= visibleCount) {
    currentIndex = Math.max(0, visibleCount - 1);
  }
  render();
}

function showAllHidden() {
  hiddenImagePaths = new Set();
  persistHiddenImages();
  showToast('All hidden photos are visible again.', { timeoutMs: 1600 });
  render();
}

function getLabeledCount() {
  const visible = getVisibleItems();
  let count = 0;
  for (const item of visible) {
    if (labelsByImage.has(item.image_path) || item.already_enrolled) {
      count += 1;
    }
  }
  return count;
}

function updateProgress() {
  const visible = getVisibleItems();
  const total = visible.length;
  const labeled = getLabeledCount();
  const pct = total === 0 ? 0 : Math.round((labeled / total) * 100);
  const currentPhoto = total === 0 ? 0 : (currentIndex + 1);
  document.getElementById('progressText').textContent = `${labeled} labeled  |  Photo ${currentPhoto} of ${total}`;
  const startInput = document.getElementById('startIndexInput');
  if (startInput) {
    startInput.value = String(currentPhoto || 1);
    startInput.max = String(Math.max(total, 1));
  }
  document.getElementById('progressFill').style.width = `${pct}%`;

  let pendingCount = 0;
  for (const action of pendingActions.values()) {
    if (!action.cancelled && !action.committed) {
      pendingCount += 1;
    }
  }
  const hiddenCount = items.length - visible.length;
  let queueText = pendingCount > 0 ? `${pendingCount} waiting to save...` : '';
  if (hiddenCount > 0) {
    queueText = queueText ? `${queueText} | ${hiddenCount} hidden` : `${hiddenCount} hidden`;
  }
  document.getElementById('queueText').textContent = queueText;
}

function updateTopDeerLabel(item) {
  const label = document.getElementById('deerTopLabel');
  if (!label) {
    return;
  }
  if (!item) {
    label.textContent = 'Deer: not selected';
    return;
  }
  const saved = labelsByImage.get(item.image_path);
  const deer =
    (saved && saved.deer_id) ||
    item.deer_prefill ||
    item.name_suggestion ||
    '';
  label.textContent = deer ? `Deer: ${deerDisplayName(deer)}` : 'Deer: not selected';
}

function chooseDeerOption(deerId) {
  setNameFromSuggestion(deerId);
}

function currentItem() {
  const visible = getVisibleItems();
  if (!visible.length) {
    return null;
  }
  if (currentIndex < 0) {
    currentIndex = 0;
  }
  if (currentIndex >= visible.length) {
    currentIndex = visible.length - 1;
  }
  return visible[currentIndex];
}

function goPrev() {
  const visible = getVisibleItems();
  if (!visible.length) {
    return;
  }
  currentIndex = Math.max(0, currentIndex - 1);
  render();
}

function goNext() {
  const visible = getVisibleItems();
  if (!visible.length) {
    return;
  }
  currentIndex = Math.min(visible.length - 1, currentIndex + 1);
  render();
}

function setSide(side) {
  selectedSide = side;
  for (const value of ['left', 'right', 'unknown']) {
    const button = document.getElementById(`side-${value}`);
    if (button) {
      button.classList.toggle('active', value === side);
    }
  }
}

function readSelectedName() {
  const picker = document.getElementById('deer-picker');
  const newInput = document.getElementById('new-deer-id');
  if (!picker) {
    return '';
  }
  if (picker.value === '__unknown_deer__') {
    return UNKNOWN_DEER_ID;
  }
  if (picker.value === '__new__') {
    return (newInput && newInput.value ? newInput.value : '').trim();
  }
  return picker.value.trim();
}

function onNamePickerChange() {
  const picker = document.getElementById('deer-picker');
  const newWrap = document.getElementById('new-deer-wrap');
  if (!picker || !newWrap) {
    return;
  }
  newWrap.style.display = picker.value === '__new__' ? 'block' : 'none';
}

function setNameFromSuggestion(name) {
  const deerId = (name || '').trim();
  if (!deerId) {
    return;
  }
  const picker = document.getElementById('deer-picker');
  const newWrap = document.getElementById('new-deer-wrap');
  const newInput = document.getElementById('new-deer-id');
  if (!picker) {
    return;
  }
  const existing = Array.from(picker.options || []).find((opt) => opt.value === deerId);
  if (existing) {
    picker.value = deerId;
    if (newWrap) {
      newWrap.style.display = 'none';
    }
  } else {
    picker.value = '__new__';
    if (newWrap) {
      newWrap.style.display = 'block';
    }
    if (newInput) {
      newInput.value = deerId;
      newInput.focus();
    }
  }
}

function showToast(message, options = {}) {
  const toast = document.getElementById('toast');
  const text = document.getElementById('toastText');
  const undoButton = document.getElementById('undoBtn');
  text.textContent = message;
  toast.style.display = 'flex';

  if (options.undoActionId) {
    activeToastActionId = options.undoActionId;
    undoButton.style.display = 'inline-flex';
  } else {
    activeToastActionId = null;
    undoButton.style.display = 'none';
  }

  if (window._toastTimer) {
    clearTimeout(window._toastTimer);
  }
  window._toastTimer = setTimeout(() => {
    toast.style.display = 'none';
    activeToastActionId = null;
  }, options.timeoutMs || 5000);
}

function undoLastAction() {
  showToast('Undo is not available after save.', { timeoutMs: 1800 });
}

async function commitAction(actionId) {
  const action = pendingActions.get(actionId);
  if (!action || action.cancelled || action.committed) {
    return false;
  }

  const payload = {
    image_path: action.image_path,
    deer_id: action.deer_id,
    side: action.side,
  };
  const resp = await fetch('/label-upsert', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload),
  });

  if (!resp.ok) {
    let errorText = '';
    try {
      errorText = await resp.text();
    } catch (e) {
      errorText = '';
    }
    action.cancelled = true;
    labelsByImage.delete(action.image_path);
    pendingActions.set(actionId, action);
    const msg = errorText ? `Save failed: ${errorText}` : 'Save failed. Please try again.';
    showToast(msg, { timeoutMs: 2600 });
    render();
    return false;
  }

  action.committed = true;
  pendingActions.set(actionId, action);

  const item = items.find((row) => row.image_path === action.image_path);
  if (item) {
    item.already_enrolled = true;
    item.side_prefill = action.side;
    item.deer_prefill = action.deer_id;
    if (action.deer_id !== UNKNOWN_DEER_ID && !item.deer_options.includes(action.deer_id)) {
      item.deer_options.push(action.deer_id);
      item.deer_options.sort((a, b) => a.localeCompare(b));
    }
  }

  labelsByImage.set(action.image_path, {
    deer_id: action.deer_id,
    side: action.side,
    state: 'committed',
  });
  render();
  return true;
}

async function deleteLabelForCurrent() {
  const item = currentItem();
  if (!item) {
    return;
  }
  const payload = { image_path: item.image_path };
  const resp = await fetch('/label-delete', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload),
  });
  if (!resp.ok) {
    showToast('Delete failed. Please try again.', { timeoutMs: 2200 });
    return;
  }
  item.already_enrolled = false;
  item.deer_prefill = '';
  item.side_prefill = 'unknown';
  labelsByImage.delete(item.image_path);
  showToast('Label deleted.', { timeoutMs: 1600 });
  render();
}

async function saveAndNext() {
  const item = currentItem();
  if (!item) {
    return;
  }

  const deerId = readSelectedName();
  if (!deerId) {
    showToast('Pick a deer name before saving.', { timeoutMs: 2200 });
    return;
  }

  const actionId = `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  const action = {
    id: actionId,
    image_path: item.image_path,
    deer_id: deerId,
    side: selectedSide,
    committed: false,
    cancelled: false,
  };
  pendingActions.set(actionId, action);
  const saved = await commitAction(actionId);
  if (!saved) {
    return;
  }
  showToast(`Saved: ${deerDisplayName(deerId)} (${selectedSide})`, { timeoutMs: 1800 });
  const visible = getVisibleItems();
  currentIndex = Math.min(currentIndex + 1, Math.max(visible.length - 1, 0));
  render();
}

function buildNamePicker(item) {
  const deerOptions = Array.isArray(item.deer_options) ? item.deer_options : [];
  const selectedLabel = labelsByImage.get(item.image_path) || null;
  const selectedByPath = (selectedLabel && selectedLabel.deer_id) || item.deer_prefill || '';
  const fallbackSuggestion = item.name_suggestion || '';
  const initialName = selectedByPath || fallbackSuggestion;
  const isUnknownInitial = initialName === UNKNOWN_DEER_ID;
  const hasInitialInOptions = deerOptions.includes(initialName);

  let pickerHtml = '<select id="deer-picker" onchange="onNamePickerChange()">';
  if (deerOptions.length > 0) {
    pickerHtml += `<option value="" ${initialName ? '' : 'selected'}>Pick a deer name</option>`;
    pickerHtml += `<option value="__unknown_deer__" ${isUnknownInitial ? 'selected' : ''}>I don't know this deer</option>`;
    for (const deerId of deerOptions) {
      const selected = deerId === initialName ? 'selected' : '';
      pickerHtml += `<option value="${escapeHtml(deerId)}" ${selected}>${escapeHtml(deerId)}</option>`;
    }
    pickerHtml += `<option value="__new__" ${hasInitialInOptions || isUnknownInitial ? '' : 'selected'}>+ New deer</option>`;
  } else {
    pickerHtml += `<option value="__unknown_deer__" ${isUnknownInitial ? 'selected' : ''}>I don't know this deer</option>`;
    pickerHtml += `<option value="__new__" ${isUnknownInitial ? '' : 'selected'}>+ New deer</option>`;
  }
  pickerHtml += '</select>';

  const shouldShowNewInput = (deerOptions.length === 0 || !hasInitialInOptions) && !isUnknownInitial;
  const newInputValue = hasInitialInOptions || isUnknownInitial ? '' : initialName;

  const quickButtons = deerOptions.slice(0, 8).map((deerId) =>
    `<button type="button" onclick="chooseDeerOption('${escapeJsSingle(deerId)}')">${escapeHtml(deerId)}</button>`
  ).join('');

  return `
    <div class="name-area">
      ${pickerHtml}
      ${quickButtons ? `<div class="deer-quick">${quickButtons}</div>` : ''}
      <div id="new-deer-wrap" style="display:${shouldShowNewInput ? 'block' : 'none'};">
        <input id="new-deer-id" type="text" value="${escapeHtml(newInputValue)}" placeholder="Type a new deer name" />
      </div>
      ${deerOptions.length === 0 ? '<div class="muted">This looks like your first deer - give it a name to get started.</div>' : ''}
    </div>`;
}

function buildSuggestionAndReferences(item) {
  const suggestion = (item.name_suggestion || '').trim();
  const likely = Array.isArray(item.likely_matches) ? item.likely_matches.slice(0, 4) : [];
  let html = '';

  if (suggestion) {
    const suggestionText = suggestion.startsWith('cluster-')
      ? `Possible cluster: ${escapeHtml(suggestion)}`
      : `Suggested name from previous cluster labels:`;
    const clickable = suggestion.startsWith('cluster-')
      ? `<span>${escapeHtml(suggestion)}</span>`
      : `<span class="suggestion-name" onclick="setNameFromSuggestion('${escapeJsSingle(suggestion)}')">${escapeHtml(suggestion)}</span>`;
    html += `
      <div class="suggestion">
        <strong>${suggestionText}</strong>
        ${clickable}
      </div>`;
  }

  const hint = (item.same_deer_hint || '').trim();
  if (!hint && likely.length === 0) {
    return html;
  }

  const cards = likely.map((match) => {
    const deerId = escapeHtml(match.deer_id || 'Unknown');
    const deerValue = escapeJsSingle(match.deer_id || 'Unknown');
    const side = escapeHtml(match.side || '');
    const scoreText = typeof match.score_pct === 'number' ? `${match.score_pct.toFixed(0)}%` : 'No score';
    const scoreLine = typeof match.score_pct === 'number' ? `Match: ${scoreText}` : scoreText;
    const path = match.image_path ? `/image?path=${encodeURIComponent(match.image_path)}` : '';
    const imageHtml = path ? `<img src="${path}" alt="Reference deer ${deerId}" />` : '';
    return `
      <div class="close-card" onclick="setNameFromSuggestion('${deerValue}')">
        ${imageHtml}
        <div class="meta"><strong>Already labeled: ${deerId}</strong>${side ? ` (${side})` : ''}<br/>${scoreLine}</div>
      </div>`;
  }).join('');

  html += `
    <div class="same-check">
      <div class="question">${hint || 'Do we think this is the same as one of the previously labeled deer?'}</div>
      ${cards ? `<div class="close-grid">${cards}</div>` : ''}
    </div>`;
  return html;
}

function renderMainCard(item) {
  if (!item) {
    document.getElementById('main').innerHTML = '<div class="empty">No photos found.</div>';
    updateTopDeerLabel(null);
    return;
  }

  const label = labelsByImage.get(item.image_path);
  const labeledText = label
    ? `<div class="labeled-chip">✓ Labeled${label.deer_id ? `: ${escapeHtml(deerDisplayName(label.deer_id))} (${escapeHtml(label.side)})` : ''}</div>`
    : '';

  document.getElementById('main').innerHTML = `
    <div class="photo-card">
      <img src="/image?path=${encodeURIComponent(item.image_path)}" alt="Deer photo" />
      ${labeledText}
      ${buildSuggestionAndReferences(item)}
      ${buildNamePicker(item)}
    </div>`;

  const side = (label && label.side) || item.side_prefill || 'unknown';
  setSide(side);
  updateTopDeerLabel(item);
}

function renderReview() {
  updateTopDeerLabel(null);
  const labeledItems = getVisibleItems().filter((item) => labelsByImage.has(item.image_path));
  if (!labeledItems.length) {
    document.getElementById('main').innerHTML = '<div class="empty">No labeled photos yet.</div>';
    return;
  }

  const cards = labeledItems.map((item) => {
    const label = labelsByImage.get(item.image_path) || {};
    const deer = escapeHtml(deerDisplayName(label.deer_id || item.deer_prefill || ''));
    const side = escapeHtml(label.side || item.side_prefill || '');
    return `
      <div class="review-card">
        <img src="/image?path=${encodeURIComponent(item.image_path)}" alt="Labeled deer" />
        <div class="review-meta">✓ Labeled ${deer ? `- ${deer}` : ''} ${side ? `(${side})` : ''}</div>
      </div>`;
  }).join('');

  document.getElementById('main').innerHTML = `<div class="review-grid">${cards}</div>`;
}

function toggleReviewMode() {
  reviewMode = !reviewMode;
  render();
}

function render() {
  persistCurrentIndex();
  updateProgress();

  const toggleButton = document.getElementById('reviewToggle');
  const dock = document.getElementById('dock');
  toggleButton.textContent = reviewMode ? 'Back to labeling' : 'Review labeled photos';
  dock.style.display = reviewMode ? 'none' : 'block';

  if (reviewMode) {
    renderReview();
    return;
  }
  const visible = getVisibleItems();
  if (!visible.length) {
    document.getElementById('main').innerHTML = '<div class="empty">No photos left in queue. Use "Show Hidden" to restore hidden photos.</div>';
    updateTopDeerLabel(null);
    return;
  }
  renderMainCard(currentItem());
}

loadItems().catch((err) => {
  const msg = (err && err.message) ? err.message : 'Unknown error';
  document.getElementById('main').innerHTML = `<div class="empty">Could not load photos: ${escapeHtml(msg)}</div>`;
});
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    images_dir: Path | None = DEFAULT_IMAGES_DIR
    embeddings_csv: Path | None = None
    embeddings_pt: Path | None = DEFAULT_EMBEDDINGS_PT
    index_csv: Path | None = DEFAULT_INDEX_CSV
    clusters_csv: Path = DEFAULT_CLUSTERS_CSV
    gallery_path: Path = DEFAULT_GALLERY_JSON

    def _send(self, code: int, body: bytes, content_type: str = "text/html") -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send(200, INDEX_HTML.encode("utf-8"))
            return
        if parsed.path == "/favicon.ico":
            self._send(204, b"", "image/x-icon")
            return
        if parsed.path == "/items":
            body = json.dumps(
                build_items(
                    self.gallery_path,
                    embeddings_csv=self.embeddings_csv,
                    images_dir=self.images_dir,
                    clusters_csv=self.clusters_csv,
                    index_csv=self.index_csv,
                    embeddings_pt=self.embeddings_pt,
                )
            ).encode("utf-8")
            self._send(200, body, "application/json")
            return
        if parsed.path == "/image":
            qs = parse_qs(parsed.query)
            raw_path = qs.get("path", [""])[0]
            if not raw_path:
                self._send(404, b"not found")
                return
            img_path = Path(raw_path)
            if not img_path.is_absolute():
                img_path = (ROOT / img_path).resolve()
            if not img_path.exists() or img_path.suffix.lower() not in {
                ".jpg",
                ".jpeg",
                ".png",
            }:
                self._send(404, b"not found")
                return
            content_type = (
                "image/png" if img_path.suffix.lower() == ".png" else "image/jpeg"
            )
            self._send(200, img_path.read_bytes(), content_type)
            return
        self._send(404, b"not found")

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")

        try:
            if self.path == "/enroll":
                result = handle_enroll_payload(payload, self.gallery_path)
                self._send(200, json.dumps(result).encode("utf-8"), "application/json")
                return
            if self.path == "/label-upsert":
                result = handle_label_upsert_payload(payload, self.gallery_path)
                self._send(200, json.dumps(result).encode("utf-8"), "application/json")
                return
            if self.path == "/label-delete":
                result = handle_label_delete_payload(payload, self.gallery_path)
                self._send(200, json.dumps(result).encode("utf-8"), "application/json")
                return
            if self.path == "/bulk-enroll":
                result = handle_bulk_enroll_payload(payload, self.gallery_path)
                self._send(200, json.dumps(result).encode("utf-8"), "application/json")
                return
        except (ValueError, FileNotFoundError) as exc:
            self._send(400, str(exc).encode("utf-8"), "text/plain")
            return

        self._send(404, b"not found")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--images-dir", default=None)
    parser.add_argument("--embeddings-csv", default=None)
    parser.add_argument("--embeddings-pt", default=str(DEFAULT_EMBEDDINGS_PT))
    parser.add_argument("--index-csv", default=str(DEFAULT_INDEX_CSV))
    parser.add_argument("--clusters-csv", default=str(DEFAULT_CLUSTERS_CSV))
    parser.add_argument("--gallery", default=str(DEFAULT_GALLERY_JSON))
    parser.add_argument("--no-open-browser", action="store_true")
    args = parser.parse_args()

    if args.images_dir:
        Handler.images_dir = _resolve_path(args.images_dir)
        Handler.embeddings_csv = None
        Handler.embeddings_pt = None
    elif args.embeddings_csv:
        Handler.images_dir = None
        Handler.embeddings_csv = _resolve_path(args.embeddings_csv)
        Handler.embeddings_pt = _resolve_path(args.embeddings_pt)
    else:
        if _has_image_files(DEFAULT_IMAGES_DIR):
            Handler.images_dir = DEFAULT_IMAGES_DIR
            Handler.embeddings_csv = None
            Handler.embeddings_pt = None
        else:
            Handler.images_dir = None
            Handler.embeddings_csv = DEFAULT_EMBEDDINGS_CSV
            Handler.embeddings_pt = _resolve_path(args.embeddings_pt)

    Handler.index_csv = _resolve_path(args.index_csv)
    Handler.clusters_csv = _resolve_path(args.clusters_csv)
    Handler.gallery_path = _resolve_path(args.gallery)

    server = HTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"Enrollment UI running at {url}")
    if not args.no_open_browser:
        threading.Timer(0.3, lambda: webbrowser.open(url)).start()
    server.serve_forever()


if __name__ == "__main__":
    main()
