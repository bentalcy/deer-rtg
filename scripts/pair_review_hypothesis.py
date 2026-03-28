"""
Minimal pair review tool for hypothesis validation.
Shows image pairs side-by-side. User marks each pair: same / different / not clear.
Results printed to terminal.
"""

import json
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

PAIRS = [
    # --- LEFT side: top 5 most similar ---
    {"id": "L-sim-1", "side": "left", "type": "high_sim", "sim": 0.9843,
     "a": "data/labeled_trap_unified_round5/keep/RCNX1041_det00.jpg",
     "b": "data/labeled_trap_unified_round5/keep/RCNX1043_det00.jpg"},
    {"id": "L-sim-2", "side": "left", "type": "high_sim", "sim": 0.9833,
     "a": "data/labeled_trap_unified_round5/keep/RCNX0459_det00.jpg",
     "b": "data/labeled_trap_unified_round5/keep/RCNX0460_det00.jpg"},
    {"id": "L-sim-3", "side": "left", "type": "high_sim", "sim": 0.9734,
     "a": "data/labeled_trap_unified_round5/keep/IMG_0496_det00.jpg",
     "b": "data/labeled_trap_unified_round5/keep/IMG_0806__dup1_det00.jpg"},
    {"id": "L-sim-4", "side": "left", "type": "high_sim", "sim": 0.9655,
     "a": "data/labeled_trap_unified_round5/keep/RCNX0067_det00.jpg",
     "b": "data/labeled_trap_unified_round5/keep/RCNX0131_det01.jpg"},
    {"id": "L-sim-5", "side": "left", "type": "high_sim", "sim": 0.9624,
     "a": "data/labeled_trap_unified_round5/keep/RCNX2974_det00.jpg",
     "b": "data/labeled_trap_unified_round5/keep/RCNX2975_det01.jpg"},
    # --- LEFT side: top 5 most different ---
    {"id": "L-dif-1", "side": "left", "type": "low_sim", "sim": 0.5840,
     "a": "data/labeled_trap_unified_round5/keep/IMG_0226_det00.jpg",
     "b": "data/labeled_trap_unified_round5/keep/IMG_0982_det00.jpg"},
    {"id": "L-dif-2", "side": "left", "type": "low_sim", "sim": 0.6039,
     "a": "data/labeled_trap_unified_round5/keep/RCNX1043_det00.jpg",
     "b": "data/labeled_trap_unified_round5/keep/RCNX2986_det00.jpg"},
    {"id": "L-dif-3", "side": "left", "type": "low_sim", "sim": 0.6262,
     "a": "data/labeled_trap_unified_round5/keep/IMG_0226_det00.jpg",
     "b": "data/labeled_trap_unified_round5/keep/RCNX1043_det00.jpg"},
    {"id": "L-dif-4", "side": "left", "type": "low_sim", "sim": 0.6299,
     "a": "data/labeled_trap_unified_round5/keep/RCNX1041_det00.jpg",
     "b": "data/labeled_trap_unified_round5/keep/RCNX2986_det00.jpg"},
    {"id": "L-dif-5", "side": "left", "type": "low_sim", "sim": 0.6325,
     "a": "data/labeled_trap_unified_round5/keep/IMG_0226_det00.jpg",
     "b": "data/labeled_trap_unified_round5/keep/RCNX0458_det00.jpg"},
    # --- RIGHT side: top 5 most similar ---
    {"id": "R-sim-1", "side": "right", "type": "high_sim", "sim": 0.9817,
     "a": "data/labeled_trap_unified_round5/keep/RCNX3776_det00.jpg",
     "b": "data/labeled_trap_unified_round5/keep/RCNX3902_det01.jpg"},
    {"id": "R-sim-2", "side": "right", "type": "high_sim", "sim": 0.9434,
     "a": "data/labeled_trap_unified_round5/keep/IMG_0977_det00.jpg",
     "b": "data/labeled_trap_unified_round5/keep/IMG_0981_det00.jpg"},
    {"id": "R-sim-3", "side": "right", "type": "high_sim", "sim": 0.8924,
     "a": "data/labeled_trap_unified_round5/keep/IMG_0011_det00.jpg",
     "b": "data/labeled_trap_unified_round5/keep/IMG_0019_det00.jpg"},
    {"id": "R-sim-4", "side": "right", "type": "high_sim", "sim": 0.8878,
     "a": "data/labeled_trap_unified_round5/keep/IMG_0977_det00.jpg",
     "b": "data/labeled_trap_unified_round5/keep/IMG_0983_det00.jpg"},
    {"id": "R-sim-5", "side": "right", "type": "high_sim", "sim": 0.8841,
     "a": "data/labeled_trap_unified_round5/keep/IMG_0318_det00.jpg",
     "b": "data/labeled_trap_unified_round5/keep/RCNX0408_det00.jpg"},
    # --- RIGHT side: top 5 most different ---
    {"id": "R-dif-1", "side": "right", "type": "low_sim", "sim": 0.6636,
     "a": "data/labeled_trap_unified_round5/keep/IMG_0011_det00.jpg",
     "b": "data/labeled_trap_unified_round5/keep/RCNX1196_det00.jpg"},
    {"id": "R-dif-2", "side": "right", "type": "low_sim", "sim": 0.6890,
     "a": "data/labeled_trap_unified_round5/keep/IMG_0019_det00.jpg",
     "b": "data/labeled_trap_unified_round5/keep/RCNX1196_det00.jpg"},
    {"id": "R-dif-3", "side": "right", "type": "low_sim", "sim": 0.7160,
     "a": "data/labeled_trap_unified_round5/keep/IMG_0011_det00.jpg",
     "b": "data/labeled_trap_unified_round5/keep/RCNX2671_det00.jpg"},
    {"id": "R-dif-4", "side": "right", "type": "low_sim", "sim": 0.7216,
     "a": "data/labeled_trap_unified_round5/keep/RCNX0408_det00.jpg",
     "b": "data/labeled_trap_unified_round5/keep/RCNX1196_det00.jpg"},
    {"id": "R-dif-5", "side": "right", "type": "low_sim", "sim": 0.7244,
     "a": "data/labeled_trap_unified_round5/keep/IMG_0058_det00.jpg",
     "b": "data/labeled_trap_unified_round5/keep/RCNX1196_det00.jpg"},
]

decisions: dict[str, str] = {}
REPO_ROOT = Path(__file__).resolve().parents[1]

HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Pair Review — Hypothesis Validation</title>
<style>
  body { font-family: sans-serif; background: #1a1a1a; color: #eee; margin: 0; padding: 20px; }
  h1 { font-size: 1.2em; color: #aaa; }
  .section-title { font-size: 1em; color: #888; margin: 30px 0 10px; border-bottom: 1px solid #333; padding-bottom: 6px; }
  .pair { display: flex; align-items: flex-start; gap: 16px; margin-bottom: 20px;
          background: #2a2a2a; border-radius: 8px; padding: 14px; }
  .pair img { width: 340px; height: 260px; object-fit: contain; background: #111; border-radius: 4px; }
  .meta { font-size: 0.8em; color: #888; margin-bottom: 8px; }
  .sim { font-size: 1.1em; font-weight: bold; color: #ffd; margin-bottom: 10px; }
  .btns { display: flex; flex-direction: column; gap: 8px; justify-content: center; padding: 0 10px; }
  button { padding: 10px 18px; border: none; border-radius: 6px; cursor: pointer;
           font-size: 0.95em; font-weight: bold; min-width: 120px; }
  .same   { background: #2a7a2a; color: #fff; }
  .diff   { background: #7a2a2a; color: #fff; }
  .unclear{ background: #555;    color: #fff; }
  button.selected { outline: 3px solid #fff; }
  .verdict { font-size: 0.85em; color: #aaa; margin-top: 6px; }
  #summary { position: fixed; bottom: 0; left: 0; right: 0; background: #111;
             border-top: 1px solid #333; padding: 10px 20px; font-size: 0.85em; color: #aaa; }
  #submit { margin-top: 30px; padding: 14px 32px; background: #4488cc;
            color: #fff; border: none; border-radius: 8px; font-size: 1em;
            font-weight: bold; cursor: pointer; }
</style>
</head>
<body>
<h1>Pair Review — Is this the same deer?</h1>
<p style="color:#888; font-size:0.9em">
  For each pair, mark: <b style="color:#6f6">Same deer</b> /
  <b style="color:#f66">Different deer</b> /
  <b style="color:#aaa">Not clear</b><br>
  The model similarity score is shown so you can see whether high/low scores correspond to your judgment.
</p>
PAIRS_HTML
<button id="submit" onclick="submitAll()">Submit all decisions</button>
<div id="summary">Decisions: <span id="count">0</span> / TOTAL_COUNT</div>
<script>
const decisions = {};
function mark(id, verdict, btn) {
  decisions[id] = verdict;
  document.querySelectorAll('[data-pair="' + id + '"]').forEach(b => b.classList.remove('selected'));
  btn.classList.add('selected');
  document.getElementById('v-' + id).textContent = verdict.toUpperCase();
  document.getElementById('count').textContent = Object.keys(decisions).length;
}
function submitAll() {
  fetch('/submit', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify(decisions)})
  .then(r => r.json())
  .then(d => { alert('Saved! Check terminal for results.'); });
}
</script>
</body>
</html>"""


def build_pair_card(p: dict, idx: int) -> str:
    type_label = "HIGH similarity" if p["type"] == "high_sim" else "LOW similarity"
    color = "#ffd080" if p["type"] == "high_sim" else "#80c8ff"
    a_name = Path(p["a"]).name
    b_name = Path(p["b"]).name
    return f"""
<div class="pair" id="pair-{p['id']}">
  <img src="/image?path={p['a']}" title="{a_name}">
  <div class="btns">
    <div class="sim" style="color:{color}">{type_label}<br>sim={p['sim']:.4f}</div>
    <div class="meta">Side: {p['side']}<br>{idx+1} of {len(PAIRS)}</div>
    <button class="same"    data-pair="{p['id']}" onclick="mark('{p['id']}','same',this)">Same deer</button>
    <button class="diff"    data-pair="{p['id']}" onclick="mark('{p['id']}','different',this)">Different deer</button>
    <button class="unclear" data-pair="{p['id']}" onclick="mark('{p['id']}','unclear',this)">Not clear</button>
    <div class="verdict" id="v-{p['id']}"></div>
  </div>
  <img src="/image?path={p['b']}" title="{b_name}">
</div>"""


def build_html() -> str:
    sections = {"LEFT — high similarity": [], "LEFT — low similarity": [],
                "RIGHT — high similarity": [], "RIGHT — low similarity": []}
    for i, p in enumerate(PAIRS):
        key = f"{p['side'].upper()} — {'high' if p['type']=='high_sim' else 'low'} similarity"
        sections[key].append(build_pair_card(p, i))
    parts = []
    for title, cards in sections.items():
        if cards:
            parts.append(f'<div class="section-title">{title}</div>' + "".join(cards))
    return (HTML
            .replace("PAIRS_HTML", "".join(parts))
            .replace("TOTAL_COUNT", str(len(PAIRS))))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = build_html().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif parsed.path == "/image":
            qs = parse_qs(parsed.query)
            img_path = REPO_ROOT / qs.get("path", [""])[0]
            if img_path.exists():
                data = img_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_response(404)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/submit":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            decisions.update(body)
            self._print_results()
            resp = json.dumps({"ok": True}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

    def _print_results(self):
        # Save to file so it can be read even when running as background process
        out = REPO_ROOT / "data/reid/pair_review_decisions.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump({"decisions": decisions, "pairs": PAIRS}, f, indent=2)

        lines = ["\n" + "="*60, "  PAIR REVIEW RESULTS", "="*60]
        for p in PAIRS:
            d = decisions.get(p["id"], "—")
            flag = ""
            if p["type"] == "high_sim" and d == "different":
                flag = "  ⚠ HIGH SIM but DIFFERENT — model confused"
            elif p["type"] == "low_sim" and d == "same":
                flag = "  ⚠ LOW SIM but SAME — model missed it"
            elif p["type"] == "high_sim" and d == "same":
                flag = "  ✓ correct"
            elif p["type"] == "low_sim" and d == "different":
                flag = "  ✓ correct"
            lines.append(f"  {p['id']:12s} sim={p['sim']:.4f}  → {d}{flag}")
        lines.append("="*60)
        correct = sum(
            1 for p in PAIRS
            if (p["type"] == "high_sim" and decisions.get(p["id"]) == "same") or
               (p["type"] == "low_sim"  and decisions.get(p["id"]) == "different")
        )
        labeled = sum(1 for p in PAIRS if decisions.get(p["id"]) not in (None, "—", "unclear"))
        lines.append(f"  Model correct on {correct}/{labeled} labeled pairs "
                     f"({100*correct/labeled:.0f}%)" if labeled else "  No labeled pairs yet.")
        lines.append("="*60)
        result_text = "\n".join(lines)
        print(result_text)
        with open(out.with_suffix(".txt"), "w") as f:
            f.write(result_text)


def main():
    port = 8766
    server = HTTPServer(("localhost", port), Handler)
    url = f"http://localhost:{port}"
    print(f"Pair review open at {url} — press Ctrl-C when done.")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    print("\nServer stopped.")


if __name__ == "__main__":
    main()
