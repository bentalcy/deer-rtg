import csv
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


ROOT = Path(".")
CLUSTERS_CSV = Path("data/reid/clusters_hdbscan.csv")
LABELS_CSV = Path("data/reid/labels.csv")
OVERRIDES_CSV = Path("data/reid/cluster_overrides.csv")
MERGES_CSV = Path("data/reid/cluster_merges.csv")
REVIEW_QUEUE_CSV = Path("data/reid/review_queue.csv")
REVIEW_DECISIONS_CSV = Path("data/reid/review_decisions.csv")
INDEX_CSV = Path("data/reid/index.csv")


def load_clusters() -> dict[str, list[str]]:
    clusters: dict[str, list[str]] = {}
    if not CLUSTERS_CSV.exists():
        return clusters
    with CLUSTERS_CSV.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = str(row["cluster_id"])
            clusters.setdefault(cid, []).append(row["image_path"])
    return clusters


def load_review_queue() -> list[dict[str, str]]:
    if not REVIEW_QUEUE_CSV.exists():
        return []
    with REVIEW_QUEUE_CSV.open(newline="") as f:
        return list(csv.DictReader(f))


def load_index() -> dict[str, dict[str, str]]:
    if not INDEX_CSV.exists():
        return {}
    with INDEX_CSV.open(newline="") as f:
        rows = list(csv.DictReader(f))
    by_instance: dict[str, dict[str, str]] = {}
    for row in rows:
        iid = (row.get("instance_id") or "").strip()
        if iid:
            by_instance[iid] = row
    return by_instance


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


INDEX_HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Flank Cluster Review</title>
  <style>
    :root {
      --bg: #f6f4ef;
      --ink: #1c1b19;
      --muted: #6b6760;
      --accent: #1d3b2a;
      --accent-2: #3e5a43;
      --panel: #ffffff;
      --border: #e0d9cf;
      --shadow: 0 6px 22px rgba(20, 18, 16, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      font-family: "Avenir Next", Avenir, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
      margin: 0;
    }
    header {
      padding: 18px 24px;
      background: linear-gradient(135deg, #f1ede4, #e9e1d5);
      border-bottom: 1px solid var(--border);
    }
    header h1 { margin: 0 0 6px 0; font-size: 22px; }
    header p { margin: 0; color: var(--muted); }
    .app { padding: 18px 24px 28px; }
    .row { display: grid; grid-template-columns: 300px 1fr; gap: 18px; }
    .sidebar { display: flex; flex-direction: column; gap: 16px; }
    .panel {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 12px;
      box-shadow: var(--shadow);
    }
    .panel h3 { margin: 0 0 8px 0; font-size: 14px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }
    .viewer { display: flex; flex-direction: column; gap: 16px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 12px; }
    .card { border: 1px solid var(--border); padding: 6px; border-radius: 10px; background: #fbfaf7; }
    img { width: 100%; height: 170px; object-fit: cover; display: block; border-radius: 8px; }
    .controls { margin: 8px 0; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
    .controls input[type=text] { padding: 6px 8px; border-radius: 8px; border: 1px solid var(--border); }
    .pill { background: #f1efe9; border-radius: 12px; padding: 3px 10px; font-size: 12px; color: var(--muted); }
    .btn { padding: 6px 10px; border: 1px solid var(--border); border-radius: 10px; background: #fff; cursor: pointer; }
    .btn.primary { background: var(--accent); color: #fff; border-color: var(--accent); }
    .btn.secondary { background: var(--accent-2); color: #fff; border-color: var(--accent-2); }
    .btn.ghost { background: #f5f1ea; }
    .btn.small { padding: 4px 8px; font-size: 12px; }
    .list { max-height: 46vh; overflow: auto; padding-right: 4px; }
    .list button { width: 100%; text-align: left; margin-bottom: 6px; }
    .pair { display: grid; grid-template-columns: repeat(2, minmax(240px, 1fr)); gap: 14px; }
    .pair img { height: 240px; }
    .hint { font-size: 12px; color: var(--muted); }
    .sticky { position: sticky; top: 12px; background: var(--panel); padding-bottom: 6px; z-index: 1; }
    .step { display: grid; grid-template-columns: 22px 1fr; gap: 8px; align-items: start; margin: 6px 0; }
    .step span { display: inline-block; width: 22px; height: 22px; border-radius: 50%; background: #e7e1d6; color: var(--accent); text-align: center; line-height: 22px; font-weight: 600; }
    .callout { border: 1px dashed var(--border); background: #f4f1ea; border-radius: 10px; padding: 10px; margin-top: 10px; }
    .callout .controls { justify-content: space-between; }
    .callout strong { color: var(--accent); }
    .callout .status { font-size: 12px; color: var(--muted); }
    .section-title { font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); margin: 12px 0 6px; }
    .helper { font-size: 12px; color: var(--muted); margin-top: 4px; }
    .field { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
    .meta { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 8px; margin-top: 6px; }
    .meta .item { background: #f7f4ee; border: 1px solid var(--border); border-radius: 8px; padding: 6px 8px; font-size: 12px; color: var(--muted); }
    .meta .item strong { color: var(--ink); font-weight: 600; }
    @media (max-width: 980px) {
      .row { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Flank Cluster Review</h1>
    <p>Resolve ambiguous pairs first, then clean clusters. All actions are logged locally.</p>
  </header>
  <div class="app">
    <div class="panel" style="margin-bottom: 16px;">
      <div class="step"><span>1</span><div>Review each pair and answer: same deer, not same, don’t know, or bad image.</div></div>
      <div class="step"><span>2</span><div>If “same deer”, the system auto-records the merge in the background.</div></div>
      <div class="step"><span>3</span><div>Move to the next pair until the queue is complete.</div></div>
    </div>
    <div class="row">
      <div class="sidebar">
        <div class="panel">
          <h3>Review Queue</h3>
          <div class="controls">
            <span class="pill" id="queueCount"></span>
            <button class="btn ghost small" onclick="prevQueueItem()">Prev</button>
            <button class="btn ghost small" onclick="nextQueueItem()">Next</button>
          </div>
          <div class="list" id="queueList"></div>
        </div>
      </div>
      <div class="viewer">
        <div class="panel">
          <h3>Review Pair</h3>
          <div class="controls">
            <span class="pill" id="queueMeta"></span>
            <span class="pill" id="queueProgress"></span>
            <input id="reviewNote" type="text" size="30" placeholder="Note (optional)" />
          </div>
          <div class="controls">
            <button class="btn primary" onclick="markDecision('same')">Yes — same deer</button>
            <button class="btn secondary" onclick="markDecision('different')">No — different deer</button>
            <button class="btn ghost" onclick="markDecision('unknown')">Don’t know</button>
            <button class="btn ghost" onclick="markDecision('bad_image')">Bad image</button>
          </div>
          <div class="pair">
            <div>
              <img id="pairImgA" />
              <div id="pairTextA" class="hint"></div>
              <div id="pairMetaA" class="meta"></div>
            </div>
            <div>
              <img id="pairImgB" />
              <div id="pairTextB" class="hint"></div>
              <div id="pairMetaB" class="meta"></div>
            </div>
          </div>
          <div class="hint" id="decisionStatus">Decisions are logged in data/reid/review_decisions.csv.</div>
        </div>
      </div>
    </div>
  </div>

<script>
let clusters = {};
let clusterIds = [];
let filterValue = '';
let reviewQueue = [];
let currentQueueIndex = null;
let indexMeta = {};

async function fetchClusters() {
  const resp = await fetch('/clusters');
  clusters = await resp.json();
  clusterIds = Object.keys(clusters).sort((a,b)=>Number(a)-Number(b));
  document.getElementById('clusterCount').textContent = `${clusterIds.length} clusters`;
  renderClusterList();
}

async function fetchReviewQueue() {
  const resp = await fetch('/review-queue');
  reviewQueue = await resp.json();
  document.getElementById('queueCount').textContent = `${reviewQueue.length} items`;
  renderQueueList();
  if (reviewQueue.length) {
    loadQueueItem(0);
  }
}

async function fetchIndexMeta() {
  const resp = await fetch('/index');
  indexMeta = await resp.json();
}

function renderClusterList() {
  const list = document.getElementById('clusterList');
  list.innerHTML = '';
  const ids = filterValue
    ? clusterIds.filter(cid => cid.includes(filterValue))
    : clusterIds;
  ids.forEach(cid => {
    const btn = document.createElement('button');
    btn.textContent = `${cid} (${clusters[cid].length})`;
    btn.onclick = () => { document.getElementById('clusterId').value = cid; loadCluster(); };
    list.appendChild(btn);
    list.appendChild(document.createElement('br'));
  });
}

function renderQueueList() {
  const list = document.getElementById('queueList');
  list.innerHTML = '';
  reviewQueue.forEach((row, idx) => {
    const btn = document.createElement('button');
    const reason = row.reason || 'review';
    btn.className = 'btn ghost';
    btn.textContent = `${idx + 1}. ${reason}`;
    btn.onclick = () => loadQueueItem(idx);
    list.appendChild(btn);
    list.appendChild(document.createElement('br'));
  });
}

function applyFilter() {
  filterValue = document.getElementById('clusterFilter').value.trim();
  renderClusterList();
}

function clearFilter() {
  filterValue = '';
  document.getElementById('clusterFilter').value = '';
  renderClusterList();
}

function loadCluster() {
  const cid = document.getElementById('clusterId').value.trim();
  const grid = document.getElementById('imageGrid');
  grid.innerHTML = '';
  if (!clusters[cid]) return;
  document.getElementById('clusterSize').textContent = `${clusters[cid].length} images`;
  clusters[cid].forEach(path => {
    const card = document.createElement('div');
    card.className = 'card';
    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.dataset.path = path;
    checkbox.onchange = updateSelectedCount;
    const img = document.createElement('img');
    img.src = `/image?path=${encodeURIComponent(path)}`;
    const p = document.createElement('div');
    p.textContent = path.split('/').pop();
    card.appendChild(checkbox);
    card.appendChild(img);
    card.appendChild(p);
    grid.appendChild(card);
  });
  updateSelectedCount();
}

function loadQueueItem(idx) {
  const row = reviewQueue[idx];
  if (!row) return;
  currentQueueIndex = idx;
  const meta = [row.reason, row.similarity].filter(Boolean).join(' | ');
  document.getElementById('queueMeta').textContent = meta;
  document.getElementById('queueProgress').textContent = `${idx + 1}/${reviewQueue.length}`;
  document.getElementById('pairImgA').src = `/image?path=${encodeURIComponent(row.image_path_a || '')}`;
  document.getElementById('pairImgB').src = `/image?path=${encodeURIComponent(row.image_path_b || '')}`;
  document.getElementById('pairTextA').textContent = `${row.instance_id_a || ''} ${row.cluster_a || row.cluster_a1 || ''}`;
  document.getElementById('pairTextB').textContent = `${row.instance_id_b || ''} ${row.cluster_b || row.cluster_a2 || ''}`;
  renderMeta('pairMetaA', row.instance_id_a, row.cluster_a || row.cluster_a1 || '');
  renderMeta('pairMetaB', row.instance_id_b, row.cluster_b || row.cluster_a2 || '');
  const status = document.getElementById('decisionStatus');
  if (status) {
    status.textContent = 'Decisions are logged in data/reid/review_decisions.csv.';
  }
}

function renderMeta(targetId, instanceId, clusterId) {
  const container = document.getElementById(targetId);
  if (!container) return;
  const meta = indexMeta[instanceId] || {};
  const timeValue = formatTime(meta.time_sec);
  const items = [
    {label: 'Image', value: (meta.image_path || '').split('/').pop() || 'n/a'},
    {label: 'Cluster', value: clusterId || 'n/a'},
    {label: 'Run', value: meta.run_id || 'n/a'},
    {label: 'Video', value: meta.video_id || 'n/a'},
    {label: 'Frame', value: meta.frame_idx || 'n/a'},
    {label: 'Time', value: timeValue},
    {label: 'Track', value: meta.track_id || 'n/a'},
    {label: 'Source video', value: meta.source_video || 'n/a'},
    {label: 'Orig name', value: meta.orig_name || 'n/a'},
    {label: 'Encounter', value: meta.encounter_id || 'n/a'},
  ];
  container.innerHTML = items.map(item => `<div class="item"><strong>${item.label}:</strong> ${item.value}</div>`).join('');
}

function formatTime(raw) {
  if (raw === undefined || raw === null || raw === '') return 'n/a';
  const seconds = Number(raw);
  if (!Number.isFinite(seconds)) return 'n/a';
  const total = Math.max(0, Math.floor(seconds));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const hh = h > 0 ? String(h).padStart(2, '0') + ':' : '';
  const mm = String(m).padStart(2, '0');
  const ss = String(s).padStart(2, '0');
  return `${hh}${mm}:${ss}`;
}

function nextQueueItem() {
  if (!reviewQueue.length) return;
  if (currentQueueIndex === null) {
    loadQueueItem(0);
    return;
  }
  const next = Math.min(currentQueueIndex + 1, reviewQueue.length - 1);
  loadQueueItem(next);
}

function prevQueueItem() {
  if (!reviewQueue.length) return;
  if (currentQueueIndex === null) {
    loadQueueItem(0);
    return;
  }
  const prev = Math.max(currentQueueIndex - 1, 0);
  loadQueueItem(prev);
}

function selectedPaths() {
  const boxes = Array.from(document.querySelectorAll('input[type=checkbox]'));
  return boxes.filter(b => b.checked).map(b => b.dataset.path);
}

function updateSelectedCount() {
  const count = selectedPaths().length;
  document.getElementById('selectedCount').textContent = `${count} selected`;
}

function selectAll() {
  const boxes = Array.from(document.querySelectorAll('input[type=checkbox]'));
  boxes.forEach(b => b.checked = true);
  updateSelectedCount();
}

function clearSelection() {
  const boxes = Array.from(document.querySelectorAll('input[type=checkbox]'));
  boxes.forEach(b => b.checked = false);
  updateSelectedCount();
}

async function labelSelected(action) {
  const deerId = document.getElementById('deerId').value.trim();
  const cid = document.getElementById('clusterId').value.trim();
  const paths = selectedPaths();
  if (!paths.length) return;
  await fetch('/label', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({paths, deer_id: deerId, action, cluster_id: cid})
  });
}

async function moveSelected() {
  const cid = document.getElementById('moveCluster').value.trim();
  const paths = selectedPaths();
  if (!paths.length || !cid) return;
  await fetch('/reassign', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({paths, new_cluster_id: cid})
  });
}

async function mergeClusters(from, to) {
  if (!from || !to) return false;
  await fetch('/merge', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({from_cluster: from, to_cluster: to})
  });
  return true;
}

async function markDecision(decision) {
  if (currentQueueIndex === null) return;
  const row = reviewQueue[currentQueueIndex];
  const note = document.getElementById('reviewNote').value.trim();
  let merged = false;
  if (decision === 'same') {
    const from = row.cluster_a || row.cluster_a1 || '';
    const to = row.cluster_b || row.cluster_a2 || '';
    merged = await mergeClusters(from, to);
  }
  await fetch('/review-decision', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      decision,
      note,
      row,
    })
  });
  const status = document.getElementById('decisionStatus');
  if (status) {
    if (decision === 'same' && merged) {
      status.textContent = 'Decision logged. Merge recorded in data/reid/cluster_merges.csv.';
    } else if (decision === 'same') {
      status.textContent = 'Decision logged. Merge not recorded (missing cluster ids).';
    } else {
      status.textContent = 'Decision logged.';
    }
  }
  nextQueueItem();
}

function nextCluster() {
  const cid = document.getElementById('clusterId').value.trim();
  const idx = clusterIds.indexOf(cid);
  if (idx >= 0 && idx < clusterIds.length - 1) {
    document.getElementById('clusterId').value = clusterIds[idx + 1];
    loadCluster();
  }
}

function prevCluster() {
  const cid = document.getElementById('clusterId').value.trim();
  const idx = clusterIds.indexOf(cid);
  if (idx > 0) {
    document.getElementById('clusterId').value = clusterIds[idx - 1];
    loadCluster();
  }
}

fetchIndexMeta();
fetchReviewQueue();
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    clusters = load_clusters()
    review_queue = load_review_queue()
    index_meta = load_index()

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
        if parsed.path == "/clusters":
            body = json.dumps(self.clusters).encode("utf-8")
            self._send(200, body, "application/json")
            return
        if parsed.path == "/review-queue":
            body = json.dumps(self.review_queue).encode("utf-8")
            self._send(200, body, "application/json")
            return
        if parsed.path == "/index":
            body = json.dumps(self.index_meta).encode("utf-8")
            self._send(200, body, "application/json")
            return
        if parsed.path == "/image":
            qs = parse_qs(parsed.query)
            path = qs.get("path", [""])[0]
            img_path = (ROOT / path).resolve()
            if not img_path.exists() or img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                self._send(404, b"not found")
                return
            body = img_path.read_bytes()
            self._send(200, body, "image/jpeg")
            return
        self._send(404, b"not found")

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        data = self.rfile.read(length)
        payload = json.loads(data or b"{}")

        if self.path == "/label":
            ensure_parent(LABELS_CSV)
            with LABELS_CSV.open("a", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["image_path", "deer_id", "action", "cluster_id"],
                )
                if f.tell() == 0:
                    writer.writeheader()
                for path in payload.get("paths", []):
                    writer.writerow(
                        {
                            "image_path": path,
                            "deer_id": payload.get("deer_id", ""),
                            "action": payload.get("action", ""),
                            "cluster_id": payload.get("cluster_id", ""),
                        }
                    )
            self._send(200, b"ok")
            return

        if self.path == "/reassign":
            ensure_parent(OVERRIDES_CSV)
            with OVERRIDES_CSV.open("a", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["image_path", "new_cluster_id"]
                )
                if f.tell() == 0:
                    writer.writeheader()
                for path in payload.get("paths", []):
                    writer.writerow(
                        {
                            "image_path": path,
                            "new_cluster_id": payload.get("new_cluster_id", ""),
                        }
                    )
            self._send(200, b"ok")
            return

        if self.path == "/merge":
            ensure_parent(MERGES_CSV)
            with MERGES_CSV.open("a", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["from_cluster", "to_cluster"]
                )
                if f.tell() == 0:
                    writer.writeheader()
                writer.writerow(
                    {
                        "from_cluster": payload.get("from_cluster", ""),
                        "to_cluster": payload.get("to_cluster", ""),
                    }
                )
            self._send(200, b"ok")
            return

        if self.path == "/review-decision":
            ensure_parent(REVIEW_DECISIONS_CSV)
            row = payload.get("row", {})
            with REVIEW_DECISIONS_CSV.open("a", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "decision",
                        "note",
                        "reason",
                        "similarity",
                        "instance_id_a",
                        "instance_id_b",
                        "image_path_a",
                        "image_path_b",
                        "cluster_a",
                        "cluster_b",
                        "cluster_a1",
                        "cluster_a2",
                        "cluster_b1",
                        "cluster_b2",
                    ],
                )
                if f.tell() == 0:
                    writer.writeheader()
                writer.writerow(
                    {
                        "decision": payload.get("decision", ""),
                        "note": payload.get("note", ""),
                        "reason": row.get("reason", ""),
                        "similarity": row.get("similarity", ""),
                        "instance_id_a": row.get("instance_id_a", ""),
                        "instance_id_b": row.get("instance_id_b", ""),
                        "image_path_a": row.get("image_path_a", ""),
                        "image_path_b": row.get("image_path_b", ""),
                        "cluster_a": row.get("cluster_a", ""),
                        "cluster_b": row.get("cluster_b", ""),
                        "cluster_a1": row.get("cluster_a1", ""),
                        "cluster_a2": row.get("cluster_a2", ""),
                        "cluster_b1": row.get("cluster_b1", ""),
                        "cluster_b2": row.get("cluster_b2", ""),
                    }
                )
            self._send(200, b"ok")
            return

        self._send(404, b"not found")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    server = HTTPServer((args.host, args.port), Handler)
    print(f"UI running at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
