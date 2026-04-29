import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import cv2
import pandas as pd


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Tree Label Review</title>
  <style>
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Arial, "Microsoft YaHei", sans-serif;
      background: #111;
      color: #e8e8e8;
      overflow: hidden;
    }
    .app {
      height: 100vh;
      display: grid;
      grid-template-columns: 1fr 280px;
      min-width: 960px;
    }
    .stage {
      position: relative;
      background: #151515;
    }
    canvas {
      width: 100%;
      height: 100%;
      display: block;
      cursor: crosshair;
    }
    .panel {
      border-left: 1px solid #333;
      background: #202020;
      padding: 14px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .title {
      font-size: 16px;
      font-weight: 700;
      line-height: 1.35;
    }
    .meta {
      color: #aaa;
      font-size: 12px;
      line-height: 1.55;
      word-break: break-all;
    }
    .row {
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
    }
    button {
      height: 34px;
      border: 1px solid #555;
      background: #2b2b2b;
      color: #eee;
      padding: 0 10px;
      border-radius: 6px;
      cursor: pointer;
      font-size: 13px;
    }
    button:hover { background: #383838; }
    button.active {
      border-color: #91d95d;
      background: #27411d;
    }
    button.warn {
      border-color: #a64a4a;
      background: #4a2222;
    }
    button.primary {
      border-color: #78b7ff;
      background: #193c62;
    }
    input[type="number"] {
      width: 82px;
      height: 34px;
      border: 1px solid #555;
      border-radius: 6px;
      background: #171717;
      color: #eee;
      padding: 0 8px;
    }
    .stat {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    .card {
      border: 1px solid #383838;
      border-radius: 8px;
      padding: 10px;
      background: #181818;
    }
    .card b {
      display: block;
      font-size: 18px;
      margin-bottom: 3px;
    }
    .card span {
      color: #aaa;
      font-size: 12px;
    }
    .selected {
      min-height: 112px;
      line-height: 1.6;
      font-size: 13px;
    }
    .help {
      margin-top: auto;
      color: #aaa;
      font-size: 12px;
      line-height: 1.65;
    }
    .status {
      color: #91d95d;
      min-height: 20px;
      font-size: 12px;
    }
  </style>
</head>
<body>
  <div class="app">
    <div class="stage"><canvas id="canvas"></canvas></div>
    <aside class="panel">
      <div>
        <div class="title">树冠标注修正</div>
        <div class="meta" id="meta">加载中...</div>
      </div>

      <div class="stat">
        <div class="card"><b id="count">0</b><span>当前圆圈</span></div>
        <div class="card"><b id="dirty">否</b><span>未保存</span></div>
      </div>

      <div class="row">
        <button id="selectBtn" class="active">选择</button>
        <button id="addBtn">补漏</button>
        <button id="fitBtn">适合窗口</button>
      </div>

      <div class="row">
        <button id="deleteBtn" class="warn">删除误检</button>
        <button id="undoBtn">撤销</button>
        <button id="saveBtn" class="primary">保存</button>
      </div>

      <div class="row">
        <span>半径</span>
        <button id="minusBtn">-</button>
        <input id="radiusInput" type="number" min="2" max="160" step="1" value="24" />
        <button id="plusBtn">+</button>
      </div>

      <div class="card selected" id="selectedBox">未选中</div>
      <div class="status" id="status"></div>

      <div class="help">
        左键选中并拖动圆圈。<br>
        补漏模式下左键新增圆圈。<br>
        鼠标滚轮缩放，右键拖动画布。<br>
        Delete 删除，Ctrl+S 保存。
      </div>
    </aside>
  </div>

<script>
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const img = new Image();
let labels = [];
let history = [];
let selected = -1;
let mode = "select";
let scale = 1;
let ox = 0;
let oy = 0;
let dirty = false;
let dragging = false;
let panning = false;
let dragStart = null;
let state = null;

const el = id => document.getElementById(id);

function resize() {
  const rect = canvas.parentElement.getBoundingClientRect();
  canvas.width = Math.max(1, Math.floor(rect.width * devicePixelRatio));
  canvas.height = Math.max(1, Math.floor(rect.height * devicePixelRatio));
  ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  draw();
}

function canvasSize() {
  const rect = canvas.getBoundingClientRect();
  return { w: rect.width, h: rect.height };
}

function fit() {
  const { w, h } = canvasSize();
  scale = Math.min(w / img.naturalWidth, h / img.naturalHeight);
  ox = (w - img.naturalWidth * scale) / 2;
  oy = (h - img.naturalHeight * scale) / 2;
  draw();
}

function imageToScreen(x, y) {
  return { x: x * scale + ox, y: y * scale + oy };
}

function screenToImage(x, y) {
  return { x: (x - ox) / scale, y: (y - oy) / scale };
}

function colorFor(label, idx) {
  if (idx === selected) return "#ffffff";
  if (label.source === "recall") return "#ffcc42";
  if (label.source === "residual") return "#62d4ff";
  if (label.source === "peak") return "#ff7ad9";
  return "#6dff55";
}

function draw() {
  const { w, h } = canvasSize();
  ctx.clearRect(0, 0, w, h);
  if (!img.complete) return;
  ctx.drawImage(img, ox, oy, img.naturalWidth * scale, img.naturalHeight * scale);

  labels.forEach((label, idx) => {
    const p = imageToScreen(label.x, label.y);
    const r = Math.max(2, label.radius * scale);
    ctx.beginPath();
    ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
    ctx.lineWidth = idx === selected ? 3 : 2;
    ctx.strokeStyle = colorFor(label, idx);
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(p.x, p.y, 2.5, 0, Math.PI * 2);
    ctx.fillStyle = idx === selected ? "#ffffff" : "#ff3333";
    ctx.fill();
  });
}

function updateUi() {
  el("count").textContent = labels.length;
  el("dirty").textContent = dirty ? "是" : "否";
  el("selectBtn").classList.toggle("active", mode === "select");
  el("addBtn").classList.toggle("active", mode === "add");
  if (selected >= 0 && labels[selected]) {
    const l = labels[selected];
    el("radiusInput").value = Math.round(l.radius);
    el("selectedBox").innerHTML =
      `ID: ${l.id || selected + 1}<br>` +
      `x: ${Math.round(l.x)}, y: ${Math.round(l.y)}<br>` +
      `radius: ${Math.round(l.radius)}<br>` +
      `source: ${l.source || "manual"}`;
  } else {
    el("selectedBox").textContent = "未选中";
  }
  draw();
}

function setDirty(value) {
  dirty = value;
  updateUi();
}

function snapshot() {
  history.push(JSON.stringify(labels));
  if (history.length > 80) history.shift();
}

function undo() {
  const last = history.pop();
  if (!last) return;
  labels = JSON.parse(last);
  selected = -1;
  setDirty(true);
}

function hitTest(x, y) {
  const p = screenToImage(x, y);
  let best = -1;
  let bestDist = Infinity;
  labels.forEach((label, idx) => {
    const d = Math.hypot(label.x - p.x, label.y - p.y);
    const limit = Math.max(label.radius + 8 / scale, 8 / scale);
    if (d <= limit && d < bestDist) {
      best = idx;
      bestDist = d;
    }
  });
  return best;
}

function addLabel(x, y) {
  const p = screenToImage(x, y);
  if (p.x < 0 || p.y < 0 || p.x > img.naturalWidth || p.y > img.naturalHeight) return;
  snapshot();
  const radius = Number(el("radiusInput").value) || 24;
  labels.push({ id: labels.length + 1, x: p.x, y: p.y, radius, source: "manual" });
  selected = labels.length - 1;
  setDirty(true);
}

function deleteSelected() {
  if (selected < 0) return;
  snapshot();
  labels.splice(selected, 1);
  selected = -1;
  setDirty(true);
}

function adjustRadius(delta) {
  if (selected < 0 || !labels[selected]) return;
  snapshot();
  labels[selected].radius = Math.max(2, Math.min(160, labels[selected].radius + delta));
  setDirty(true);
}

function setRadius(value) {
  if (selected < 0 || !labels[selected]) return;
  snapshot();
  labels[selected].radius = Math.max(2, Math.min(160, Number(value) || 2));
  setDirty(true);
}

canvas.addEventListener("contextmenu", e => e.preventDefault());
canvas.addEventListener("mousedown", e => {
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;

  if (e.button === 2) {
    panning = true;
    dragStart = { x, y, ox, oy };
    return;
  }

  if (mode === "add") {
    addLabel(x, y);
    return;
  }

  selected = hitTest(x, y);
  if (selected >= 0) {
    snapshot();
    const p = screenToImage(x, y);
    dragging = true;
    dragStart = {
      x: p.x,
      y: p.y,
      labelX: labels[selected].x,
      labelY: labels[selected].y,
    };
  }
  updateUi();
});

window.addEventListener("mousemove", e => {
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;

  if (panning && dragStart) {
    ox = dragStart.ox + (x - dragStart.x);
    oy = dragStart.oy + (y - dragStart.y);
    draw();
    return;
  }

  if (dragging && selected >= 0 && dragStart) {
    const p = screenToImage(x, y);
    labels[selected].x = dragStart.labelX + (p.x - dragStart.x);
    labels[selected].y = dragStart.labelY + (p.y - dragStart.y);
    labels[selected].x = Math.max(0, Math.min(img.naturalWidth, labels[selected].x));
    labels[selected].y = Math.max(0, Math.min(img.naturalHeight, labels[selected].y));
    dirty = true;
    updateUi();
  }
});

window.addEventListener("mouseup", () => {
  dragging = false;
  panning = false;
  dragStart = null;
});

canvas.addEventListener("wheel", e => {
  e.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const sx = e.clientX - rect.left;
  const sy = e.clientY - rect.top;
  const before = screenToImage(sx, sy);
  const factor = e.deltaY < 0 ? 1.12 : 0.89;
  scale = Math.max(0.08, Math.min(12, scale * factor));
  ox = sx - before.x * scale;
  oy = sy - before.y * scale;
  draw();
}, { passive: false });

window.addEventListener("keydown", e => {
  if (e.key === "Delete" || e.key === "Backspace") deleteSelected();
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "s") {
    e.preventDefault();
    save();
  }
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") {
    e.preventDefault();
    undo();
  }
});

async function save() {
  const payload = labels.map((l, idx) => ({
    id: idx + 1,
    x: Math.round(l.x),
    y: Math.round(l.y),
    radius: Math.round(l.radius),
    source: l.source || "manual",
  }));
  const res = await fetch("/api/save", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!res.ok) {
    el("status").textContent = data.error || "保存失败";
    return;
  }
  labels = payload;
  dirty = false;
  el("status").textContent = `已保存: ${data.path}`;
  updateUi();
}

el("selectBtn").onclick = () => { mode = "select"; updateUi(); };
el("addBtn").onclick = () => { mode = "add"; updateUi(); };
el("fitBtn").onclick = fit;
el("deleteBtn").onclick = deleteSelected;
el("undoBtn").onclick = undo;
el("saveBtn").onclick = save;
el("minusBtn").onclick = () => adjustRadius(-2);
el("plusBtn").onclick = () => adjustRadius(2);
el("radiusInput").addEventListener("change", e => setRadius(e.target.value));

async function init() {
  state = await (await fetch("/api/state")).json();
  labels = await (await fetch("/api/labels")).json();
  el("meta").innerHTML = `${state.source}<br>输出: ${state.out}`;
  img.onload = () => {
    resize();
    fit();
    updateUi();
  };
  img.src = "/image";
}

window.addEventListener("resize", resize);
init();
</script>
</body>
</html>
"""


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def read_labels(path: Path):
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)
    required = {"x", "y", "radius"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} 缺少列: {', '.join(sorted(missing))}")

    labels = []
    for idx, row in df.iterrows():
        labels.append({
            "id": int(row["id"]) if "id" in df.columns and pd.notna(row["id"]) else idx + 1,
            "x": float(row["x"]),
            "y": float(row["y"]),
            "radius": float(row["radius"]),
            "source": str(row["source"]) if "source" in df.columns and pd.notna(row["source"]) else "loaded",
        })
    return labels


def write_labels(path: Path, labels):
    ensure_dir(path.parent)
    rows = []
    for idx, label in enumerate(labels, 1):
        rows.append({
            "id": idx,
            "x": int(round(float(label["x"]))),
            "y": int(round(float(label["y"]))),
            "radius": int(round(float(label["radius"]))),
            "source": label.get("source", "manual"),
        })
    df = pd.DataFrame(rows)
    df.to_excel(path, index=False)
    df.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")


class ReviewServer(BaseHTTPRequestHandler):
    source = None
    labels_path = None
    out_path = None
    labels = []
    image_size = None

    def _send(self, status=200, content_type="text/plain; charset=utf-8", body=b""):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(status, "application/json; charset=utf-8", body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self._send(200, "text/html; charset=utf-8", INDEX_HTML.encode("utf-8"))
            return
        if path == "/image":
            data = self.source.read_bytes()
            content_type = "image/jpeg" if self.source.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
            self._send(200, content_type, data)
            return
        if path == "/api/state":
            self._json({
                "source": str(self.source),
                "labels": str(self.labels_path),
                "out": str(self.out_path),
                "width": self.image_size[0],
                "height": self.image_size[1],
            })
            return
        if path == "/api/labels":
            self._json(self.labels)
            return
        self._json({"error": "not found"}, status=404)

    def do_POST(self):
        path = urlparse(self.path).path
        if path != "/api/save":
            self._json({"error": "not found"}, status=404)
            return

        length = int(self.headers.get("Content-Length", "0"))
        try:
            labels = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(labels, list):
                raise ValueError("payload must be a list")
            write_labels(self.out_path, labels)
            self.__class__.labels = read_labels(self.out_path)
            self._json({"ok": True, "path": str(self.out_path), "count": len(labels)})
        except Exception as exc:
            self._json({"error": str(exc)}, status=400)

    def log_message(self, fmt, *args):
        return


def parse_args():
    parser = argparse.ArgumentParser(description="Local visual tree crown label correction tool")
    parser.add_argument("--source", default="DJI_0108.JPG", help="原始图片")
    parser.add_argument("--labels", default="output/DJI_0108_trees.xlsx", help="待修正的 Excel/CSV 标签")
    parser.add_argument("--out", default="labels_corrected/DJI_0108_trees_corrected.xlsx", help="修正后的输出 Excel")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=8765, help="监听端口")
    return parser.parse_args()


def main():
    args = parse_args()
    source = Path(args.source).resolve()
    labels_path = Path(args.labels).resolve()
    out_path = Path(args.out).resolve()

    if not source.exists():
        raise FileNotFoundError(source)
    if not labels_path.exists():
        raise FileNotFoundError(labels_path)

    img = cv2.imread(str(source))
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {source}")
    height, width = img.shape[:2]

    ReviewServer.source = source
    ReviewServer.labels_path = labels_path
    ReviewServer.out_path = out_path
    ReviewServer.labels = read_labels(labels_path)
    ReviewServer.image_size = (width, height)

    server = ThreadingHTTPServer((args.host, args.port), ReviewServer)
    print(f"打开浏览器访问: http://{args.host}:{args.port}")
    print(f"读取标签: {labels_path}")
    print(f"保存到: {out_path}")
    server.serve_forever()


if __name__ == "__main__":
    main()
