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


ROOT = Path(".").resolve()
DEFAULT_IMAGES_DIR = Path("images")
DEFAULT_CLUSTERS_CSV = Path("data/reid/clusters.csv")
DEFAULT_GALLERY_JSON = Path("data/gallery/gallery.json")


def _load_cluster_suggestions(clusters_csv: Path) -> dict[str, str]:
    """Returns filename → 'cluster-N' suggestion, keyed by basename only."""
    if not clusters_csv.exists():
        return {}
    suggestions: dict[str, str] = {}
    with clusters_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            image_path = str(row.get("image_path") or "").strip()
            cluster_id = str(row.get("cluster_id") or "").strip()
            if image_path and cluster_id and cluster_id != "-1":
                suggestions[Path(image_path).name] = f"cluster-{cluster_id}"
    return suggestions


def _normalize_side(value: str) -> str:
    side = value.strip().lower()
    if side in {"left", "right"}:
        return side
    return "unknown"


def _load_enrolled_paths(gallery_path: Path) -> set[str]:
    gallery = load_gallery(gallery_path)
    paths: set[str] = set()
    for deer_data in gallery.values():
        for side in ("left", "right"):
            side_data = deer_data.get(side, {}) if isinstance(deer_data, dict) else {}
            if isinstance(side_data, dict):
                for path in side_data.get("image_paths", []):
                    paths.add(str(path))
    return paths


_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def build_items(
    gallery_path: Path,
    embeddings_csv: Path | None = None,
    images_dir: Path | None = None,
    clusters_csv: Path | None = None,
) -> list[dict[str, Any]]:
    enrolled = _load_enrolled_paths(gallery_path)
    suggestions = _load_cluster_suggestions(clusters_csv) if clusters_csv else {}
    items: list[dict[str, Any]] = []

    if images_dir is not None and images_dir.exists():
        for p in sorted(images_dir.rglob("*")):
            if p.suffix.lower() not in _IMAGE_EXTS:
                continue
            image_path = str(p)
            items.append(
                {
                    "image_path": image_path,
                    "side_prefill": "unknown",
                    "already_enrolled": image_path in enrolled,
                    "name_suggestion": suggestions.get(p.name, ""),
                }
            )
    elif embeddings_csv is not None and embeddings_csv.exists():
        with embeddings_csv.open(newline="") as f:
            for row in csv.DictReader(f):
                image_path = str(row.get("image_path") or "").strip()
                if not image_path:
                    continue
                side_prefill = _normalize_side(str(row.get("side_pred") or ""))
                items.append(
                    {
                        "image_path": image_path,
                        "side_prefill": side_prefill,
                        "already_enrolled": image_path in enrolled,
                        "name_suggestion": suggestions.get(Path(image_path).name, ""),
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


INDEX_HTML = """<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>Deer Enrollment UI</title>
  <style>
    body { font-family: "Avenir Next", Avenir, "Segoe UI", sans-serif; margin: 0; background: #f6f4ef; color: #1c1b19; }
    header { padding: 16px 20px; border-bottom: 1px solid #dfd8cc; background: #efe9de; }
    .wrap { padding: 16px 20px; }
    .controls { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-bottom: 12px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 12px; }
    .card { background: #fff; border: 1px solid #dfd8cc; border-radius: 10px; padding: 8px; }
    .card.enrolled { border: 2px solid #2b7a4b; }
    img { width: 100%; height: 280px; object-fit: cover; border-radius: 8px; background: #ece6da; }
    input[type=text], select { width: 100%; padding: 6px 8px; border: 1px solid #d8d1c5; border-radius: 8px; }
    .row { display: grid; grid-template-columns: 1fr 96px; gap: 8px; margin-top: 8px; }
    button { padding: 6px 10px; border: 1px solid #d0c8bb; border-radius: 8px; background: #fff; cursor: pointer; }
    button.primary { background: #1e5a36; color: #fff; border-color: #1e5a36; }
    .muted { color: #736d62; font-size: 12px; }
  </style>
</head>
<body>
  <header>
    <h2 style=\"margin:0;\">Deer Enrollment UI</h2>
    <div class=\"muted\">Batch-enroll known deer from existing image paths.</div>
  </header>
  <div class=\"wrap\">
    <div class=\"controls\">
      <button onclick=\"selectAll()\">Select all</button>
      <button onclick=\"clearAll()\">Clear</button>
      <button class=\"primary\" onclick=\"bulkEnroll()\">Enroll selected</button>
      <span id=\"status\" class=\"muted\"></span>
    </div>
    <div id=\"grid\" class=\"grid\"></div>
  </div>
<script>
let items = [];

async function loadItems() {
  const resp = await fetch('/items');
  items = await resp.json();
  render();
}

function cardHtml(item, idx) {
  const checked = item.already_enrolled ? '' : 'checked';
  const enrolledClass = item.already_enrolled ? 'enrolled' : '';
  const side = item.side_prefill || 'unknown';
  return `
    <div class="card ${enrolledClass}">
      <label><input type="checkbox" id="cb-${idx}" ${checked}/> select</label>
      <img src="/image?path=${encodeURIComponent(item.image_path)}" />
      <div class="muted">${item.image_path.split('/').pop()}</div>
      <input type="text" id="id-${idx}" value="${item.name_suggestion || ''}" placeholder="deer id (e.g. 5A)" />${item.name_suggestion ? `<div class="muted" style="font-size:11px;margin-top:2px;">suggested from cluster</div>` : ''}
      <div class="row">
        <select id="side-${idx}">
          <option value="left" ${side === 'left' ? 'selected' : ''}>left</option>
          <option value="right" ${side === 'right' ? 'selected' : ''}>right</option>
        </select>
        <button onclick="enrollOne(${idx})">Enroll</button>
      </div>
    </div>`;
}

function render() {
  document.getElementById('grid').innerHTML = items.map(cardHtml).join('');
}

function selectAll() { items.forEach((_, i) => { const cb = document.getElementById(`cb-${i}`); if (cb) cb.checked = true; }); }
function clearAll() { items.forEach((_, i) => { const cb = document.getElementById(`cb-${i}`); if (cb) cb.checked = false; }); }

async function enrollOne(idx) {
  const deerId = (document.getElementById(`id-${idx}`).value || '').trim();
  const side = document.getElementById(`side-${idx}`).value;
  if (!deerId) return;
  const payload = { image_path: items[idx].image_path, deer_id: deerId, side };
  const resp = await fetch('/enroll', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(payload) });
  if (resp.ok) {
    items[idx].already_enrolled = true;
    render();
  }
}

async function bulkEnroll() {
  const selected = [];
  items.forEach((item, i) => {
    const cb = document.getElementById(`cb-${i}`);
    if (!cb || !cb.checked) return;
    const deerId = (document.getElementById(`id-${i}`).value || '').trim();
    const side = document.getElementById(`side-${i}`).value;
    selected.push({ image_path: item.image_path, deer_id: deerId, side });
  });
  const resp = await fetch('/bulk-enroll', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({items: selected}),
  });
  const out = await resp.json();
  document.getElementById('status').textContent = `Requested: ${out.requested}, enrolled: ${out.enrolled}`;
  await loadItems();
}

loadItems();
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    images_dir: Path | None = DEFAULT_IMAGES_DIR
    embeddings_csv: Path | None = None
    clusters_csv: Path = DEFAULT_CLUSTERS_CSV
    gallery_path: Path = DEFAULT_GALLERY_JSON

    def _send(self, code: int, body: bytes, content_type: str = "text/html") -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send(200, INDEX_HTML.encode("utf-8"))
            return
        if parsed.path == "/items":
            body = json.dumps(
                build_items(
                    self.gallery_path,
                    embeddings_csv=self.embeddings_csv,
                    images_dir=self.images_dir,
                    clusters_csv=self.clusters_csv,
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
    parser.add_argument("--clusters-csv", default=str(DEFAULT_CLUSTERS_CSV))
    parser.add_argument("--gallery", default=str(DEFAULT_GALLERY_JSON))
    parser.add_argument("--no-open-browser", action="store_true")
    args = parser.parse_args()

    if args.images_dir:
        Handler.images_dir = Path(args.images_dir)
        Handler.embeddings_csv = None
    elif args.embeddings_csv:
        Handler.images_dir = None
        Handler.embeddings_csv = Path(args.embeddings_csv)
    else:
        Handler.images_dir = DEFAULT_IMAGES_DIR
        Handler.embeddings_csv = None

    Handler.clusters_csv = Path(args.clusters_csv)
    Handler.gallery_path = Path(args.gallery)

    server = HTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"Enrollment UI running at {url}")
    if not args.no_open_browser:
        threading.Timer(0.3, lambda: webbrowser.open(url)).start()
    server.serve_forever()


if __name__ == "__main__":
    main()
