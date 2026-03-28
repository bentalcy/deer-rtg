import argparse
import csv
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import cast
from urllib.parse import parse_qs, urlparse


FIELDNAMES = [
    "cluster_id",
    "deer_id_likely",
    "prototype_path",
    "prototype_path_2",
    "question_path",
    "decision",
]

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
        cluster_id = (row.get("cluster_id") or "").strip()
        if cluster_id:
            out[cluster_id] = row
    return out


def merge_pairs_with_existing(
    pair_rows: list[dict[str, str]],
    existing: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    merged: list[dict[str, str]] = []
    for raw in pair_rows:
        pair = normalize_pair_row(raw)
        cluster_id = pair["cluster_id"]
        prev = existing.get(cluster_id)
        merged.append(
            {
                "cluster_id": cluster_id,
                "deer_id_likely": pair["deer_id_likely"],
                "prototype_path": pair["prototype_path"],
                "prototype_path_2": pair.get("prototype_path_2", ""),
                "question_path": pair["question_path"],
                "decision": (prev.get("decision", "").strip() if prev else ""),
            }
        )
    return merged


def save_decisions(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def build_state(rows: list[dict[str, str]], selected_cluster_id: str | None = None) -> dict[str, object]:
    total = len(rows)
    decided = sum(1 for row in rows if row.get("decision", "").strip())
    remaining = total - decided

    by_cluster = {row["cluster_id"]: idx for idx, row in enumerate(rows)}
    current_idx: int | None = None
    if selected_cluster_id and selected_cluster_id in by_cluster:
        current_idx = by_cluster[selected_cluster_id]
    elif remaining > 0:
        for idx, row in enumerate(rows):
            if not row.get("decision", "").strip():
                current_idx = idx
                break
    elif rows:
        current_idx = len(rows) - 1

    current_row = rows[current_idx] if current_idx is not None else None
    return {
        "summary": {
            "total": total,
            "decided": decided,
            "remaining": remaining,
            "progress": f"{decided}/{total}" if total else "0/0",
        },
        "current_index": current_idx,
        "current": current_row,
        "rows": rows,
    }


def repo_relative_path(repo_root: Path, raw_path: str) -> Path | None:
    candidate = Path(raw_path)
    target = candidate if candidate.is_absolute() else (repo_root / candidate)
    resolved_target = target.resolve()
    resolved_root = repo_root.resolve()
    try:
        _ = resolved_target.relative_to(resolved_root)
    except ValueError:
        return None
    return resolved_target


INDEX_HTML = """<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Pair Review Web UI</title>
  <style>
    :root {
      --bg: #f3f0e8;
      --ink: #1c1c1a;
      --muted: #666357;
      --panel: #ffffff;
      --accent: #2f5b42;
      --accent-2: #8f3d23;
      --border: #e1dccf;
    }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: \"Avenir Next\", Avenir, \"Segoe UI\", sans-serif; color: var(--ink); background: radial-gradient(circle at top right, #e8dfcf, var(--bg)); }
    header { padding: 16px 20px; border-bottom: 1px solid var(--border); background: #f8f4ea; }
    header h1 { margin: 0 0 6px 0; font-size: 22px; }
    header .meta { display: flex; gap: 8px; flex-wrap: wrap; color: var(--muted); }
    .pill { background: #efe8da; border: 1px solid var(--border); border-radius: 999px; padding: 4px 10px; font-size: 12px; }
    .layout { display: grid; grid-template-columns: 320px 1fr; gap: 14px; padding: 14px; }
    .panel { background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 12px; }
    .queue { max-height: calc(100vh - 190px); overflow: auto; }
    .queue-item { width: 100%; text-align: left; border: 1px solid var(--border); border-radius: 10px; background: #faf8f2; margin-bottom: 8px; padding: 8px; cursor: pointer; }
    .queue-item.active { outline: 2px solid var(--accent); }
    .queue-item.done { border-left: 4px solid var(--accent); }
    .queue-item.pending { border-left: 4px solid var(--accent-2); }
    .actions { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }
    button { border: 1px solid var(--border); border-radius: 10px; padding: 8px 12px; cursor: pointer; background: #fff; }
    button.primary { background: var(--accent); color: #fff; border-color: var(--accent); }
    button.secondary { background: var(--accent-2); color: #fff; border-color: var(--accent-2); }
    .grid { display: grid; grid-template-columns: repeat(3, minmax(180px, 1fr)); gap: 10px; }
    .card { border: 1px solid var(--border); border-radius: 10px; padding: 8px; background: #fcfbf7; }
    .card h3 { margin: 0 0 6px 0; font-size: 14px; color: var(--muted); }
    img { width: 100%; height: 280px; object-fit: cover; border-radius: 8px; background: #ede6d7; }
    .path { font-size: 12px; color: var(--muted); margin-top: 6px; word-break: break-all; }
    .hint { margin-top: 10px; color: var(--muted); font-size: 12px; }
    @media (max-width: 980px) {
      .layout { grid-template-columns: 1fr; }
      .grid { grid-template-columns: 1fr; }
      img { height: 220px; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Pair Review Web UI</h1>
    <div class=\"meta\">
      <span class=\"pill\" id=\"progress\">0/0</span>
      <span class=\"pill\" id=\"remaining\">remaining: 0</span>
      <span class=\"pill\" id=\"current\">cluster: -</span>
    </div>
  </header>

  <div class=\"layout\">
    <div class=\"panel\">
      <div class=\"actions\">
        <button onclick=\"jumpPrev()\">Prev</button>
        <button onclick=\"jumpNext()\">Next</button>
        <button onclick=\"jumpNextPending()\">Next Undecided</button>
      </div>
      <div class=\"queue\" id=\"queue\"></div>
    </div>

    <div class=\"panel\">
      <div class=\"actions\">
        <button class=\"primary\" onclick=\"setDecision('same')\">Same (S)</button>
        <button class=\"secondary\" onclick=\"setDecision('different')\">Different (D)</button>
        <button onclick=\"setDecision('error_image')\">Error Image (E)</button>
        <button onclick=\"setDecision('side_mismatch')\">Side Mismatch (M)</button>
        <button onclick=\"setDecision('')\">Clear</button>
      </div>
      <div class=\"grid\">
        <div class=\"card\">
          <h3>Prototype A</h3>
          <img id=\"imgA\" alt=\"Prototype A\" />
          <div class=\"path\" id=\"pathA\"></div>
        </div>
        <div class=\"card\">
          <h3>Prototype B</h3>
          <img id=\"imgB\" alt=\"Prototype B\" />
          <div class=\"path\" id=\"pathB\"></div>
        </div>
        <div class=\"card\">
          <h3>Question</h3>
          <img id=\"imgQ\" alt=\"Question\" />
          <div class=\"path\" id=\"pathQ\"></div>
        </div>
      </div>
      <div class=\"hint\">Keyboard: S same, D different, E error image, M side mismatch, ArrowUp/ArrowDown navigation.</div>
      <div class=\"hint\" id=\"status\"></div>
    </div>
  </div>

  <script>
    let state = null;

    async function refresh(selectedClusterId = null) {
      const url = selectedClusterId ? `/api/state?cluster_id=${encodeURIComponent(selectedClusterId)}` : '/api/state';
      const resp = await fetch(url);
      state = await resp.json();
      render();
    }

    function render() {
      if (!state) return;
      document.getElementById('progress').textContent = state.summary.progress;
      document.getElementById('remaining').textContent = `remaining: ${state.summary.remaining}`;
      document.getElementById('current').textContent = state.current ? `cluster: ${state.current.cluster_id}` : 'cluster: -';

      const queue = document.getElementById('queue');
      queue.innerHTML = '';
      state.rows.forEach((row, idx) => {
        const btn = document.createElement('button');
        const decided = !!row.decision;
        const current = state.current_index === idx;
        btn.className = `queue-item ${decided ? 'done' : 'pending'} ${current ? 'active' : ''}`;
        btn.textContent = `#${idx + 1}  cluster ${row.cluster_id}  ${row.decision || 'pending'}`;
        btn.onclick = () => refresh(row.cluster_id);
        queue.appendChild(btn);
      });

      renderCurrent();
    }

    function imageUrl(path) {
      if (!path) return '';
      return `/image?path=${encodeURIComponent(path)}`;
    }

    function setImage(imgId, pathId, path) {
      const img = document.getElementById(imgId);
      const text = document.getElementById(pathId);
      if (!path) {
        img.removeAttribute('src');
        text.textContent = 'n/a';
        return;
      }
      img.src = imageUrl(path);
      text.textContent = path;
    }

    function renderCurrent() {
      const row = state.current;
      if (!row) {
        setImage('imgA', 'pathA', '');
        setImage('imgB', 'pathB', '');
        setImage('imgQ', 'pathQ', '');
        return;
      }
      setImage('imgA', 'pathA', row.prototype_path);
      setImage('imgB', 'pathB', row.prototype_path_2 || '');
      setImage('imgQ', 'pathQ', row.question_path);
    }

    async function setDecision(decision) {
      if (!state || !state.current) return;
      const payload = { cluster_id: state.current.cluster_id, decision };
      const resp = await fetch('/api/decision', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!resp.ok) {
        document.getElementById('status').textContent = 'Failed to save decision';
        return;
      }
      document.getElementById('status').textContent = `Saved cluster ${state.current.cluster_id} -> ${decision || 'cleared'}`;
      await refresh();
    }

    function jumpRelative(step) {
      if (!state || state.current_index === null) return;
      const idx = Math.max(0, Math.min(state.rows.length - 1, state.current_index + step));
      const row = state.rows[idx];
      if (row) refresh(row.cluster_id);
    }

    function jumpPrev() { jumpRelative(-1); }
    function jumpNext() { jumpRelative(1); }

    function jumpNextPending() {
      if (!state || !state.rows.length) return;
      const start = state.current_index === null ? 0 : state.current_index + 1;
      for (let i = start; i < state.rows.length; i += 1) {
        if (!state.rows[i].decision) {
          refresh(state.rows[i].cluster_id);
          return;
        }
      }
      for (let i = 0; i < start; i += 1) {
        if (!state.rows[i].decision) {
          refresh(state.rows[i].cluster_id);
          return;
        }
      }
    }

    window.addEventListener('keydown', (event) => {
      const key = (event.key || '').toLowerCase();
      if (key in { s: true, d: true, e: true, m: true }) {
        setDecision({ s: 'same', d: 'different', e: 'error_image', m: 'side_mismatch' }[key]);
        event.preventDefault();
        return;
      }
      if (event.key === 'ArrowUp') {
        jumpPrev();
        event.preventDefault();
        return;
      }
      if (event.key === 'ArrowDown') {
        jumpNext();
        event.preventDefault();
      }
    });

    refresh();
  </script>
</body>
</html>
"""


class PairReviewStore:
    def __init__(self, pairs_csv: Path, decisions_csv: Path) -> None:
        self.pairs_csv: Path = pairs_csv
        self.decisions_csv: Path = decisions_csv
        self.rows: list[dict[str, str]] = []
        self.reload()

    def reload(self) -> None:
        pair_rows = load_rows(self.pairs_csv)
        existing = load_existing_decisions(self.decisions_csv)
        self.rows = merge_pairs_with_existing(pair_rows, existing)

    def get_state(self, selected_cluster_id: str | None = None) -> dict[str, object]:
        return build_state(self.rows, selected_cluster_id=selected_cluster_id)

    def set_decision(self, cluster_id: str, decision: str) -> None:
        changed = False
        for row in self.rows:
            if row["cluster_id"] != cluster_id:
                continue
            row["decision"] = decision
            changed = True
            break
        if not changed:
            raise ValueError(f"cluster_id not found: {cluster_id}")
        save_decisions(self.decisions_csv, self.rows)


def create_handler(repo_root: Path, store: PairReviewStore):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, content_type: str = "text/html") -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            _ = self.wfile.write(body)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send(200, INDEX_HTML.encode("utf-8"))
                return

            if parsed.path == "/api/state":
                qs = parse_qs(parsed.query)
                selected_cluster_id = (qs.get("cluster_id", [""])[0] or "").strip() or None
                body = json.dumps(store.get_state(selected_cluster_id=selected_cluster_id)).encode("utf-8")
                self._send(200, body, "application/json")
                return

            if parsed.path == "/image":
                qs = parse_qs(parsed.query)
                raw_path = (qs.get("path", [""])[0] or "").strip()
                safe = repo_relative_path(repo_root, raw_path)
                if safe is None or not safe.exists() or safe.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    self._send(404, b"not found", "text/plain")
                    return
                content_type = "image/png" if safe.suffix.lower() == ".png" else "image/jpeg"
                self._send(200, safe.read_bytes(), content_type)
                return

            self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:
            if self.path != "/api/decision":
                self._send(404, b"not found", "text/plain")
                return

            length = int(self.headers.get("Content-Length", "0"))
            payload_obj = cast(object, json.loads(self.rfile.read(length) or b"{}"))
            if not isinstance(payload_obj, dict):
                self._send(400, b"invalid payload", "text/plain")
                return
            payload = cast(dict[str, object], payload_obj)
            cluster_raw = payload.get("cluster_id", "")
            decision_raw = payload.get("decision", "")
            cluster_id = str(cluster_raw).strip()
            decision = str(decision_raw).strip()

            if decision and decision not in set(KEY_TO_DECISION.values()):
                self._send(400, b"invalid decision", "text/plain")
                return
            if not cluster_id:
                self._send(400, b"missing cluster_id", "text/plain")
                return
            try:
                store.set_decision(cluster_id, decision)
            except ValueError as exc:
                self._send(404, str(exc).encode("utf-8"), "text/plain")
                return

            self._send(200, b"ok", "text/plain")

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    _ = parser.add_argument(
        "--pairs-csv",
        default="data/reid/unified_round5/pair_review_recognizable_plus_manual_round2_k40_sideaware.csv",
    )
    _ = parser.add_argument(
        "--decisions-csv",
        default="data/reid/unified_round5/pair_review_recognizable_plus_manual_round2_k40_sideaware_decisions.csv",
    )
    _ = parser.add_argument("--host", default="127.0.0.1")
    _ = parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()

    parsed_args = cast(dict[str, object], vars(args))
    pairs_raw = parsed_args.get("pairs_csv", "")
    decisions_raw = parsed_args.get("decisions_csv", "")
    host_raw = parsed_args.get("host", "127.0.0.1")
    port_raw = parsed_args.get("port", 8010)

    if not isinstance(pairs_raw, str) or not isinstance(decisions_raw, str):
        raise ValueError("invalid csv path args")
    if not isinstance(host_raw, str):
        raise ValueError("invalid host arg")
    if not isinstance(port_raw, int):
        raise ValueError("invalid port arg")

    repo_root = Path(__file__).resolve().parents[2]
    pairs_csv = Path(pairs_raw)
    decisions_csv = Path(decisions_raw)
    if not pairs_csv.exists():
        raise FileNotFoundError(f"pairs csv not found: {pairs_csv}")

    store = PairReviewStore(pairs_csv=pairs_csv, decisions_csv=decisions_csv)
    handler = create_handler(repo_root=repo_root, store=store)
    host = host_raw
    port = port_raw
    server = HTTPServer((host, port), handler)
    print(f"Pair review UI running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
