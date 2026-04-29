import argparse
import json
import math
import subprocess
import sys
import urllib.parse
from urllib.parse import urlparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import pandas as pd

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
DEMO_DIR = Path("demo_half_labeled")
DEFAULT_IMAGE_PATH = "raw_images/DJI_0108.JPG"
DEFAULT_DATA_PATH = "labels_pseudo/DJI_0108_trees.xlsx"

INDEX_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>无人机生态巡检 Demo</title>
    <!-- 引入 Google Fonts 提升质感 -->
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=Outfit:wght@400;600;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #00f0ff;
            --success: #00ff66;
            --warning: #ffcc00;
            --danger: #ff3366;
            --dark: #07070a;
            --panel-bg: rgba(15, 15, 20, 0.75);
            --border: rgba(255, 255, 255, 0.08);
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            padding: 0;
            background: var(--dark);
            background-image: 
                radial-gradient(circle at 15% 50%, rgba(0, 240, 255, 0.03) 0%, transparent 50%),
                radial-gradient(circle at 85% 30%, rgba(0, 255, 102, 0.03) 0%, transparent 50%);
            color: #fff;
            font-family: 'Inter', sans-serif;
            overflow: hidden;
            display: flex;
            height: 100vh;
        }
        
        /* 侧边栏: 玻璃拟态效果 */
        .sidebar {
            width: 360px;
            flex-shrink: 0;
            background: var(--panel-bg);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border-right: 1px solid var(--border);
            padding: 32px 24px;
            display: flex;
            flex-direction: column;
            gap: 28px;
            z-index: 10;
            box-shadow: 10px 0 40px rgba(0,0,0,0.6);
            overflow-y: auto;
        }
        
        .header h1 {
            font-family: 'Outfit', sans-serif;
            margin: 0;
            font-size: 32px;
            font-weight: 800;
            background: linear-gradient(135deg, #fff 0%, #a0a0b0 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: 0.5px;
        }
        .header .subtitle {
            color: var(--primary);
            font-size: 13px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 2px;
            margin-bottom: 8px;
            display: block;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
        }
        .stat-card {
            background: linear-gradient(180deg, rgba(255,255,255,0.03) 0%, rgba(255,255,255,0.01) 100%);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 20px 16px;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        .stat-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; height: 2px;
            background: linear-gradient(90deg, transparent, var(--primary), transparent);
            opacity: 0;
            transition: opacity 0.3s ease;
        }
        .stat-card:hover {
            transform: translateY(-3px);
            border-color: rgba(0, 240, 255, 0.3);
            box-shadow: 0 10px 20px rgba(0, 240, 255, 0.05);
        }
        .stat-card:hover::before { opacity: 1; }
        
        .stat-value {
            font-family: 'Outfit', sans-serif;
            font-size: 36px;
            font-weight: 800;
            color: #fff;
            margin-bottom: 6px;
            line-height: 1;
        }
        .stat-label {
            font-size: 12px;
            color: #889;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .metric-panel {
            background: rgba(255,255,255,0.02);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 18px;
        }
        .metric-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 16px;
            padding: 10px 0;
            border-bottom: 1px solid rgba(255,255,255,0.06);
            font-size: 13px;
            color: #ccd;
        }
        .metric-row:last-child { border-bottom: 0; }
        .metric-value {
            font-family: 'Outfit', sans-serif;
            font-size: 18px;
            font-weight: 800;
            color: #fff;
            text-align: right;
            white-space: nowrap;
        }
        .status-pill {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 76px;
            padding: 5px 9px;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.12);
            background: rgba(255,255,255,0.04);
            font-size: 12px;
            font-weight: 800;
        }
        .layer-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }
        .layer-grid button {
            min-height: 42px;
            padding: 10px;
            border-radius: 10px;
            font-size: 12px;
        }
        
        /* 最优解说明框 */
        .solution-box {
            background: linear-gradient(135deg, rgba(0, 255, 102, 0.08) 0%, rgba(0, 255, 102, 0.02) 100%);
            border: 1px solid rgba(0, 255, 102, 0.2);
            border-radius: 16px;
            padding: 20px;
            position: relative;
        }
        .solution-box::after {
            content: '✓ OPTIMAL';
            position: absolute;
            top: 16px; right: 16px;
            font-size: 10px;
            font-weight: 800;
            color: var(--success);
            letter-spacing: 1px;
            background: rgba(0, 255, 102, 0.1);
            padding: 4px 8px;
            border-radius: 12px;
        }
        .solution-box h3 {
            color: #fff;
            margin: 0 0 12px 0;
            font-size: 16px;
            font-family: 'Outfit', sans-serif;
        }
        .solution-box p {
            margin: 0;
            font-size: 13px;
            color: #bbb;
            line-height: 1.6;
        }

        .dataset-panel {
            background: rgba(255,255,255,0.02);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 18px;
        }
        .dataset-head {
            display: flex;
            justify-content: space-between;
            gap: 12px;
            color: #889;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.7px;
            text-transform: uppercase;
            margin-bottom: 12px;
        }
        #datasetName {
            color: var(--primary);
            max-width: 150px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        select {
            width: 100%;
            min-height: 42px;
            background: rgba(5, 8, 12, 0.88);
            color: #fff;
            border: 1px solid rgba(255,255,255,0.16);
            border-radius: 10px;
            padding: 0 12px;
            font-size: 13px;
            outline: none;
        }
        select:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 2px rgba(0, 240, 255, 0.12);
        }
        
        .legend {
            background: rgba(255,255,255,0.02);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 20px;
        }
        .legend-title {
            font-size: 13px;
            color: #889;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 12px;
            font-weight: 600;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 10px;
            font-size: 13px;
            color: #ccc;
        }
        .legend-item:last-child { margin-bottom: 0; }
        .color-dot {
            width: 12px; height: 12px;
            border-radius: 50%;
            box-shadow: 0 0 10px currentColor;
        }
        
        .controls {
            display: flex;
            gap: 12px;
            margin-top: auto;
        }
        button {
            flex: 1;
            padding: 14px;
            background: rgba(255,255,255,0.03);
            border: 1px solid var(--border);
            color: #fff;
            border-radius: 12px;
            cursor: pointer;
            font-weight: 600;
            font-size: 14px;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        }
        button:hover {
            background: rgba(255,255,255,0.08);
            border-color: rgba(255,255,255,0.3);
            transform: translateY(-2px);
        }
        button.active {
            background: var(--primary);
            color: #000;
            border-color: var(--primary);
            box-shadow: 0 4px 15px rgba(0, 240, 255, 0.3);
        }
        
        .upload-btn {
            background: linear-gradient(135deg, rgba(0, 240, 255, 0.15), rgba(0, 240, 255, 0.05));
            border-color: var(--primary);
            color: var(--primary);
            margin-top: 5px;
            width: 100%;
        }
        .upload-btn:hover {
            background: var(--primary);
            color: #000;
        }

        /* 舞台区域 */
        .stage {
            flex: 1;
            position: relative;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #050505;
        }
        /* 添加网格背景提升科技感 */
        .stage::before {
            content: '';
            position: absolute;
            inset: 0;
            background-image: 
                linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px);
            background-size: 40px 40px;
            pointer-events: none;
        }
        
        .canvas-container {
            position: absolute;
            top: 0;
            left: 0;
            flex: 0 0 auto;
            background: #111;
            box-shadow: 0 20px 60px rgba(0,0,0,0.8);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 4px;
            overflow: hidden;
            transform-origin: top left;
            will-change: transform;
        }
        #baseImage {
            display: block;
            width: 100%;
            height: 100%;
            object-fit: contain;
            user-select: none;
            pointer-events: none;
        }
        canvas {
            display: block;
            position: absolute;
            inset: 0;
            cursor: crosshair;
        }
        
        /* 悬浮提示框 */
        #tooltip {
            position: absolute;
            background: rgba(10, 10, 15, 0.9);
            border: 1px solid var(--primary);
            padding: 16px;
            border-radius: 12px;
            pointer-events: none;
            opacity: 0;
            transform: translateY(10px);
            transition: opacity 0.2s, transform 0.2s;
            font-size: 13px;
            color: #fff;
            z-index: 100;
            backdrop-filter: blur(8px);
            box-shadow: 0 10px 25px rgba(0,240,255,0.15);
            min-width: 160px;
        }
        #tooltip.visible {
            opacity: 1;
            transform: translateY(0);
        }
        .tooltip-title {
            color: var(--primary);
            font-family: 'Outfit', sans-serif;
            font-size: 16px;
            font-weight: 800;
            margin-bottom: 10px;
            padding-bottom: 8px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .tooltip-row {
            display: flex;
            justify-content: space-between;
            margin-bottom: 6px;
        }
        .tooltip-label { color: #889; }
        .tooltip-val { font-weight: 600; font-family: monospace; }
        
        /* Loading 遮罩 */
        #loadingOverlay {
            position: fixed;
            inset: 0;
            background: rgba(0,0,0,0.85);
            backdrop-filter: blur(10px);
            z-index: 1000;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.3s;
        }
        #loadingOverlay.active {
            opacity: 1;
            pointer-events: all;
        }
        .spinner {
            width: 50px; height: 50px;
            border: 3px solid rgba(0, 240, 255, 0.2);
            border-top-color: var(--primary);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-bottom: 20px;
        }
        @keyframes spin { 100% { transform: rotate(360deg); } }
        .loading-text {
            color: var(--primary);
            font-size: 16px;
            font-family: 'Outfit', sans-serif;
            letter-spacing: 2px;
        }
        
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 4px; }
    </style>
</head>
<body>
    <div class="sidebar">
        <div class="header">
            <span class="subtitle">Drone Eco Patrol</span>
            <h1>生态巡检<br>智能分析</h1>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value" id="totalCount">0</div>
                <div class="stat-label">树木数量</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="coverageRate">0%</div>
                <div class="stat-label">林草覆盖率</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="carbonStock">0</div>
                <div class="stat-label">碳汇估算 tCO₂e</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="riskCount">0</div>
                <div class="stat-label">疑似风险点</div>
            </div>
        </div>
        
        <div class="solution-box">
            <h3>巡检能力组合</h3>
            <p>当前 demo 已接入树木计数、树种粗分类、林草覆盖率、荒漠化等级、碳汇估算、疑似垃圾/火点风险提示。垃圾和火点为影像启发式识别，可作为演示与后续模型训练前的原型。</p>
        </div>

        <div class="metric-panel">
            <div class="metric-row"><span>荒漠化/裸地状态</span><span class="status-pill" id="desertStatus">--</span></div>
            <div class="metric-row"><span>疑似垃圾场/堆放区</span><span class="metric-value" id="wasteCount">0</span></div>
            <div class="metric-row"><span>疑似起火点</span><span class="metric-value" id="fireCount">0</span></div>
            <div class="metric-row"><span>主要树种/冠型</span><span class="metric-value" id="dominantSpecies">--</span></div>
        </div>

        <div class="dataset-panel">
            <div class="dataset-head">
                <span id="datasetCount">DEMO 0/0</span>
                <span id="datasetName">未加载</span>
            </div>
            <select id="imageSelect" aria-label="选择演示图片"></select>
        </div>
        
        <div class="legend">
            <div class="legend-title">图层说明</div>
            <div class="legend-item">
                <div class="color-dot" style="color: #00ff66; background: #00ff66;"></div>
                <span>树冠/树种识别</span>
            </div>
            <div class="legend-item">
                <div class="color-dot" style="color: #d8b45a; background: #d8b45a;"></div>
                <span>裸地/荒漠化压力</span>
            </div>
            <div class="legend-item">
                <div class="color-dot" style="color: #00f0ff; background: #00f0ff;"></div>
                <span>疑似垃圾堆放区</span>
            </div>
            <div class="legend-item">
                <div class="color-dot" style="color: #ff3366; background: #ff3366;"></div>
                <span>疑似火点/热风险</span>
            </div>
        </div>
        
        <div class="controls">
            <button id="btnPulse" class="active">动态</button>
            <button id="btnShape" class="active">轮廓</button>
            <button id="btnFit">适应窗口</button>
        </div>
        <div class="layer-grid">
            <button id="btnTrees" class="active">树种/数量</button>
            <button id="btnCoverage" class="active">覆盖/荒漠化</button>
            <button id="btnWaste" class="active">垃圾识别</button>
            <button id="btnFire" class="active">火点风险</button>
        </div>
        <button id="btnUpload" class="upload-btn">⇧ 上传巡检影像</button>
        <input type="file" id="fileInput" accept="image/jpeg, image/png" style="display: none;">
    </div>
    
    <div class="stage" id="stage">
        <div class="canvas-container" id="container">
            <img id="baseImage" alt="检测图片">
            <canvas id="canvas"></canvas>
            <div id="tooltip">
                <div class="tooltip-title" id="tt-title">Tree #001</div>
                <div class="tooltip-row"><span class="tooltip-label">来源:</span> <span class="tooltip-val" id="tt-source">reference</span></div>
                <div class="tooltip-row"><span class="tooltip-label">坐标:</span> <span class="tooltip-val" id="tt-pos">0, 0</span></div>
                <div class="tooltip-row"><span class="tooltip-label">边界:</span> <span class="tooltip-val" id="tt-rad">0 px</span></div>
            </div>
        </div>
    </div>
    
    <div id="loadingOverlay">
        <div class="spinner"></div>
        <div class="loading-text">AI 正在执行生态巡检分析...</div>
    </div>

<script>
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    const container = document.getElementById('container');
    const stage = document.getElementById('stage');
    const tooltip = document.getElementById('tooltip');
    const baseImage = document.getElementById('baseImage');
    const btnUpload = document.getElementById('btnUpload');
    const fileInput = document.getElementById('fileInput');
    const loadingOverlay = document.getElementById('loadingOverlay');
    const imageSelect = document.getElementById('imageSelect');
    
    let trees = [];
    let manifest = [];
    let analysis = null;
    
    // Global state
    let currentImgPath = '';
    let currentDataPath = '';
    
    // View state
    let scale = 1;
    let panX = 0;
    let panY = 0;
    let isDragging = false;
    let startX, startY;
    
    // Animation state
    let enablePulse = true;
    let useContours = true;
    let layers = {trees: true, coverage: true, waste: true, fire: true};
    let time = 0;
    let hoveredTree = null;
    let renderFrame = null;

    // Helper functions
    const colors = {
        'reference': '#00ff66',
        'recall': '#ffcc00',
        'residual': '#00f0ff',
        'peak': '#ff3366',
        'broadleaf': '#00ff66',
        'shrub': '#7dff9a',
        'dark_crown': '#00c2a8',
        'sparse_crown': '#b6ff4a',
        'unknown': '#ffffff'
    };
    
    function getColor(source) {
        return colors[source] || '#ffffff';
    }

    function fmtPercent(value) {
        return `${Number(value || 0).toFixed(1)}%`;
    }

    function updateAnalysisPanel(payload) {
        analysis = payload || null;
        const metrics = analysis?.metrics || {};
        const species = analysis?.species || {};
        const waste = analysis?.waste_regions || [];
        const fire = analysis?.fire_points || [];

        document.getElementById('coverageRate').innerText = fmtPercent(metrics.vegetation_cover_pct);
        document.getElementById('carbonStock').innerText = Number(metrics.carbon_stock_tco2e || 0).toFixed(1);
        document.getElementById('riskCount').innerText = waste.length + fire.length;
        document.getElementById('wasteCount').innerText = waste.length;
        document.getElementById('fireCount').innerText = fire.length;
        document.getElementById('desertStatus').innerText = metrics.desertification_level || '--';
        document.getElementById('dominantSpecies').innerText = species.dominant || '--';
    }

    async function loadConfigAndManifest() {
        try {
            const configRes = await fetch('/api/config');
            if(configRes.ok) {
                const config = await configRes.json();
                currentImgPath = config.default_image || currentImgPath;
                currentDataPath = config.default_data || currentDataPath;
            }
        } catch(e) {
            console.warn('Config load failed', e);
        }

        try {
            const res = await fetch('/api/manifest');
            if(res.ok) {
                const data = await res.json();
                manifest = data.items || [];
                renderManifestSelect();
                if(manifest.length > 0) {
                    const selected = manifest.find(item => item.data_path === currentDataPath) || manifest[0];
                    selectManifestItem(selected);
                }
            }
        } catch(e) {
            manifest = [];
        }
    }

    function renderManifestSelect() {
        imageSelect.innerHTML = '';
        if(!manifest.length) {
            const option = document.createElement('option');
            option.textContent = '未找到半量演示集';
            option.value = '';
            imageSelect.appendChild(option);
            document.getElementById('datasetCount').innerText = 'DEMO 0/0';
            return;
        }

        manifest.forEach((item, index) => {
            const option = document.createElement('option');
            option.value = item.data_path;
            option.textContent = `${String(index + 1).padStart(2, '0')}  ${item.name}  (${item.tree_count} 株)`;
            imageSelect.appendChild(option);
        });
    }

    function selectManifestItem(item) {
        if(!item) return;
        currentImgPath = item.image_path;
        currentDataPath = item.data_path;
        imageSelect.value = item.data_path;
        const index = manifest.findIndex(entry => entry.data_path === item.data_path);
        document.getElementById('datasetCount').innerText = `DEMO ${index + 1}/${manifest.length}`;
        document.getElementById('datasetName').innerText = item.name;
    }

    function hasContour(t) {
        return useContours && Array.isArray(t.contour) && t.contour.length >= 3;
    }

    function buildContourPath(t) {
        const pts = t.contour;
        ctx.beginPath();
        ctx.moveTo(pts[0][0], pts[0][1]);
        for(let i = 1; i < pts.length; i++) {
            ctx.lineTo(pts[i][0], pts[i][1]);
        }
        ctx.closePath();
    }

    function pointInPolygon(x, y, points) {
        let inside = false;
        for(let i = 0, j = points.length - 1; i < points.length; j = i++) {
            const xi = points[i][0], yi = points[i][1];
            const xj = points[j][0], yj = points[j][1];
            const intersects = ((yi > y) !== (yj > y)) &&
                (x < (xj - xi) * (y - yi) / ((yj - yi) || 1e-6) + xi);
            if(intersects) inside = !inside;
        }
        return inside;
    }

    function hitTree(x, y, t) {
        if(hasContour(t) && pointInPolygon(x, y, t.contour)) {
            return true;
        }
        return Math.hypot(t.x - x, t.y - y) <= t.radius;
    }

    async function init() {
        if(renderFrame) cancelAnimationFrame(renderFrame);
        
        try {
            const res = await fetch(`/api/data?file=${encodeURIComponent(currentDataPath)}&image=${encodeURIComponent(currentImgPath)}`);
            if(res.ok) {
                const data = await res.json();
                trees = data.trees;
                document.getElementById('totalCount').innerText = trees.length;
            } else {
                trees = [];
            }
        } catch(e) {
            trees = [];
        }

        try {
            const res = await fetch(`/api/analysis?image=${encodeURIComponent(currentImgPath)}&data=${encodeURIComponent(currentDataPath)}`);
            if(res.ok) {
                updateAnalysisPanel(await res.json());
            } else {
                updateAnalysisPanel(null);
            }
        } catch(e) {
            updateAnalysisPanel(null);
        }
        
        baseImage.onload = () => {
            canvas.width = baseImage.naturalWidth;
            canvas.height = baseImage.naturalHeight;
            container.style.width = `${baseImage.naturalWidth}px`;
            container.style.height = `${baseImage.naturalHeight}px`;
            hoveredTree = null;
            isDragging = false;
            tooltip.classList.remove('visible');
            fitToScreen();
            renderFrame = requestAnimationFrame(render);
        };
        baseImage.onerror = () => {
            const fallbackPath = currentDataPath.replace(/_trees\.xlsx$/i, '_detected.jpg').replace(/\\/g, '/');
            if(fallbackPath && fallbackPath !== currentImgPath) {
                console.warn('Image load failed, fallback to detected image:', currentImgPath);
                currentImgPath = fallbackPath;
                init();
                return;
            }
            alert('图片加载失败，请检查文件是否为浏览器支持的 JPG/PNG：' + currentImgPath);
        };
        baseImage.removeAttribute('src');
        baseImage.src = `/image?file=${encodeURIComponent(currentImgPath)}&t=${Date.now()}`;
    }

    function fitToScreen() {
        if(!baseImage.naturalWidth) return;
        const stageRect = stage.getBoundingClientRect();
        const padding = 60;
        const availableWidth = Math.max(120, stageRect.width - padding);
        const availableHeight = Math.max(120, stageRect.height - padding);
        const scaleX = availableWidth / baseImage.naturalWidth;
        const scaleY = availableHeight / baseImage.naturalHeight;
        scale = Math.min(scaleX, scaleY);
        if (scale > 1) scale = 1;
        if (scale < 0.05) scale = 0.05;
        panX = (stageRect.width - baseImage.naturalWidth * scale) / 2;
        panY = (stageRect.height - baseImage.naturalHeight * scale) / 2;
        applyTransform();
    }

    function applyTransform() {
        container.style.transform = `translate(${panX}px, ${panY}px) scale(${scale})`;
    }

    // Interaction
    stage.addEventListener('wheel', (e) => {
        e.preventDefault();
        const zoomIntensity = 0.1;
        const wheel = e.deltaY < 0 ? 1 : -1;
        const zoomFactor = Math.exp(wheel * zoomIntensity);
        
        // Zoom to mouse position
        const rect = stage.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;
        const nextScale = Math.max(0.05, Math.min(6, scale * zoomFactor));
        const effectiveZoom = nextScale / scale;
        
        panX = mouseX - (mouseX - panX) * effectiveZoom;
        panY = mouseY - (mouseY - panY) * effectiveZoom;
        scale = nextScale;
        
        applyTransform();
    }, {passive: false});

    stage.addEventListener('mousedown', (e) => {
        if(e.button === 0 || e.button === 2) {
            isDragging = true;
            startX = e.clientX - panX;
            startY = e.clientY - panY;
            stage.style.cursor = 'grabbing';
        }
    });

    window.addEventListener('mousemove', (e) => {
        if (isDragging) {
            panX = e.clientX - startX;
            panY = e.clientY - startY;
            applyTransform();
        } else {
            // Check hover
            checkHover(e);
        }
    });

    window.addEventListener('mouseup', () => {
        isDragging = false;
        stage.style.cursor = 'default';
    });
    
    stage.addEventListener('contextmenu', e => e.preventDefault());

    function checkHover(e) {
        const rect = canvas.getBoundingClientRect();
        const x = (e.clientX - rect.left) / scale;
        const y = (e.clientY - rect.top) / scale;
        
        let found = null;
        // Search backwards to pick top layer
        for(let i = trees.length - 1; i >= 0; i--) {
            const t = trees[i];
            if(hitTree(x, y, t)) {
                found = t;
                break;
            }
        }
        
        hoveredTree = found;
        
        if(found) {
            tooltip.classList.add('visible');
            tooltip.style.left = (e.clientX + 15) + 'px';
            tooltip.style.top = (e.clientY + 15) + 'px';
            
            document.getElementById('tt-title').innerText = `Tree #${found.id || '?'}`;
            const color = getColor(found.species_key || found.source);
            document.getElementById('tt-title').style.color = color;
            document.getElementById('tooltip').style.borderColor = color;
            
            document.getElementById('tt-source').innerText = found.species || found.source || 'unknown';
            document.getElementById('tt-pos').innerText = `${Math.round(found.x)}, ${Math.round(found.y)}`;
            if(Array.isArray(found.contour) && found.contour.length >= 3) {
                document.getElementById('tt-rad').innerText = `${found.contour.length} 点 / ${Math.round(found.radius)} px`;
            } else {
                document.getElementById('tt-rad').innerText = `${Math.round(found.radius)} px`;
            }
        } else {
            tooltip.classList.remove('visible');
        }
    }

    function drawRectRegion(region, color, fillAlpha) {
        const box = region.bbox || [];
        if(box.length < 4) return;
        const [x, y, w, h] = box;
        ctx.save();
        ctx.lineWidth = 2 / scale;
        ctx.strokeStyle = color;
        ctx.fillStyle = color + fillAlpha;
        ctx.setLineDash([8 / scale, 5 / scale]);
        ctx.fillRect(x, y, w, h);
        ctx.strokeRect(x, y, w, h);
        ctx.setLineDash([]);
        ctx.fillStyle = color;
        ctx.font = `${12 / scale}px Inter`;
        ctx.fillText(region.label || 'risk', x + 4 / scale, y + 14 / scale);
        ctx.restore();
    }

    function drawAnalysisOverlays() {
        if(!analysis) return;

        if(layers.coverage && analysis.coverage_grid) {
            const grid = analysis.coverage_grid;
            const cellW = canvas.width / grid.cols;
            const cellH = canvas.height / grid.rows;
            ctx.save();
            for(const cell of grid.cells) {
                const x = cell.col * cellW;
                const y = cell.row * cellH;
                const bare = Number(cell.bare || 0);
                const veg = Number(cell.vegetation || 0);
                if(bare > 0.58) {
                    ctx.fillStyle = `rgba(216, 180, 90, ${Math.min(0.34, bare * 0.36)})`;
                    ctx.fillRect(x, y, cellW + 1, cellH + 1);
                } else if(veg > 0.42) {
                    ctx.fillStyle = `rgba(0, 255, 102, ${Math.min(0.18, veg * 0.16)})`;
                    ctx.fillRect(x, y, cellW + 1, cellH + 1);
                }
            }
            ctx.restore();
        }

        if(layers.waste) {
            (analysis.waste_regions || []).forEach(region => drawRectRegion(region, '#00f0ff', '26'));
        }

        if(layers.fire) {
            (analysis.fire_points || []).forEach(point => {
                ctx.save();
                const r = Math.max(16, Number(point.radius || 22));
                const glow = enablePulse ? 0.5 + Math.sin(time * 1.8) * 0.18 : 0.5;
                ctx.beginPath();
                ctx.arc(point.x, point.y, r, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(255, 51, 102, ${glow})`;
                ctx.fill();
                ctx.lineWidth = 3 / scale;
                ctx.strokeStyle = '#fff';
                ctx.stroke();
                ctx.restore();
            });
        }
    }

    // Render Loop
    function render() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Draw dark overlay to make colors pop
        ctx.fillStyle = 'rgba(0,0,0,0.15)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        time += 0.05;
        const pulse = enablePulse ? (Math.sin(time) * 0.2 + 0.8) : 1;
        
        drawAnalysisOverlays();

        // Draw trees
        if(layers.trees) trees.forEach(t => {
            const color = getColor(t.species_key || t.source);
            const isHovered = hoveredTree && hoveredTree.id === t.id;
            const drawContour = hasContour(t);
            
            if(drawContour) {
                buildContourPath(t);
            } else {
                ctx.beginPath();
                ctx.arc(t.x, t.y, t.radius, 0, Math.PI * 2);
            }
            
            if(isHovered) {
                ctx.fillStyle = color + '40'; // 25% opacity
                ctx.fill();
                ctx.lineWidth = 3 / scale;
                ctx.strokeStyle = '#fff';
                ctx.stroke();
                
                // Draw crosshair
                ctx.beginPath();
                ctx.moveTo(t.x - 10/scale, t.y); ctx.lineTo(t.x + 10/scale, t.y);
                ctx.moveTo(t.x, t.y - 10/scale); ctx.lineTo(t.x, t.y + 10/scale);
                ctx.strokeStyle = '#fff';
                ctx.lineWidth = 1/scale;
                ctx.stroke();
            } else {
                ctx.lineWidth = 2 / scale;
                ctx.strokeStyle = color;
                
                // Add glow if pulse enabled
                if(enablePulse && t.source === 'reference') {
                    ctx.shadowBlur = 15 * pulse;
                    ctx.shadowColor = color;
                } else {
                    ctx.shadowBlur = 0;
                }
                
                ctx.stroke();
                
                // Center point
                ctx.shadowBlur = 0;
                ctx.beginPath();
                ctx.arc(t.x, t.y, 1.5/scale, 0, Math.PI * 2);
                ctx.fillStyle = color;
                ctx.fill();
            }
        });
        
        renderFrame = requestAnimationFrame(render);
    }

    // Buttons
    document.getElementById('btnPulse').onclick = function() {
        enablePulse = !enablePulse;
        this.classList.toggle('active');
    };

    document.getElementById('btnShape').onclick = function() {
        useContours = !useContours;
        this.classList.toggle('active');
        this.innerText = useContours ? '轮廓' : '圆形';
    };
    
    document.getElementById('btnFit').onclick = fitToScreen;

    function bindLayerButton(id, key) {
        document.getElementById(id).onclick = function() {
            layers[key] = !layers[key];
            this.classList.toggle('active', layers[key]);
        };
    }
    bindLayerButton('btnTrees', 'trees');
    bindLayerButton('btnCoverage', 'coverage');
    bindLayerButton('btnWaste', 'waste');
    bindLayerButton('btnFire', 'fire');

    imageSelect.onchange = async () => {
        const item = manifest.find(entry => entry.data_path === imageSelect.value);
        if(!item) return;
        selectManifestItem(item);
        await init();
    };
    
    // Upload logic
    btnUpload.onclick = () => fileInput.click();
    
    fileInput.onchange = async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        
        loadingOverlay.classList.add('active');
        try {
            const res = await fetch(`/api/upload?filename=${encodeURIComponent(file.name)}`, {
                method: 'POST',
                body: file
            });
            const result = await res.json();
            if (result.ok) {
                await loadConfigAndManifest();
                currentImgPath = result.img_path;
                currentDataPath = result.data_path;
                await init();
            } else {
                alert('检测失败: ' + result.error);
            }
        } catch (err) {
            alert('上传出错: ' + err);
        } finally {
            loadingOverlay.classList.remove('active');
            fileInput.value = '';
        }
    };
    
    // Start
    loadConfigAndManifest().then(init);
</script>
</body>
</html>
"""

def build_manifest():
    items = []
    if not DEMO_DIR.exists():
        return items

    for label_path in sorted(DEMO_DIR.glob("*_trees.xlsx")):
        stem = label_path.name[:-len("_trees.xlsx")]
        image_path = None
        for suffix in IMAGE_SUFFIXES:
            for image_suffix in (suffix, suffix.upper()):
                candidate = DEMO_DIR / f"{stem}{image_suffix}"
                if candidate.exists():
                    image_path = candidate
                    break
            if image_path is not None:
                break
        if image_path is None:
            for suffix in IMAGE_SUFFIXES:
                candidate = Path("raw_images") / f"{stem}{suffix.upper()}"
                if candidate.exists():
                    image_path = candidate
                    break
        if image_path is None:
            continue

        tree_count = 0
        try:
            tree_count = int(len(pd.read_excel(label_path)))
        except Exception:
            pass

        items.append({
            "name": stem,
            "image_path": str(image_path).replace("\\", "/"),
            "data_path": str(label_path).replace("\\", "/"),
            "tree_count": tree_count,
        })
    return items


def classify_tree(row):
    radius = float(row.get("radius", 0) or 0)
    area_px = float(row.get("area_px", 0) or 0)
    fill_ratio = float(row.get("fill_ratio", 0) or 0)
    source = str(row.get("source", "") or "")

    if source == "dark":
        return "暗绿阔叶树", "dark_crown"
    if radius <= 13 or area_px < 360:
        return "灌木/幼树", "shrub"
    if fill_ratio and fill_ratio < 0.30:
        return "稀疏冠层树", "sparse_crown"
    if radius >= 34 or area_px >= 1800:
        return "成熟阔叶树", "broadleaf"
    return "普通阔叶树", "broadleaf"


def read_tree_rows(data_path):
    rows = []
    excel_path = Path(data_path)
    if not excel_path.exists():
        return rows

    df = pd.read_excel(excel_path)
    for idx, row in df.iterrows():
        contour = []
        raw_contour = row.get("contour", None)
        if raw_contour is not None and not pd.isna(raw_contour):
            try:
                parsed = json.loads(str(raw_contour))
                if isinstance(parsed, list) and len(parsed) >= 3:
                    contour = [
                        [float(pt[0]), float(pt[1])]
                        for pt in parsed
                        if isinstance(pt, list) and len(pt) >= 2
                    ]
            except Exception:
                contour = []

        species, species_key = classify_tree(row)
        rows.append({
            "id": int(row.get("id", idx + 1)),
            "x": float(row["x"]),
            "y": float(row["y"]),
            "radius": float(row["radius"]),
            "source": str(row.get("source", "reference")),
            "area_px": float(row.get("area_px", 0)),
            "fill_ratio": float(row.get("fill_ratio", 0) or 0),
            "contour": contour,
            "species": species,
            "species_key": species_key,
        })
    return rows


def contour_or_circle_area(tree):
    contour = tree.get("contour") or []
    if len(contour) >= 3:
        try:
            import cv2
            import numpy as np
            pts = np.array(contour, dtype=np.float32)
            return float(abs(cv2.contourArea(pts)))
        except Exception:
            pass
    radius = float(tree.get("radius", 0) or 0)
    return math.pi * radius * radius


def refine_tree_species_from_image(img_bgr, trees):
    import cv2
    import numpy as np

    if img_bgr is None or not trees:
        return trees

    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    for tree in trees:
        cx = int(round(tree.get("x", 0)))
        cy = int(round(tree.get("y", 0)))
        radius = max(2, int(round(tree.get("radius", 0))))
        pad = max(4, int(radius * 1.2))
        x1 = max(0, cx - pad)
        y1 = max(0, cy - pad)
        x2 = min(w, cx + pad + 1)
        y2 = min(h, cy + pad + 1)
        if x2 <= x1 or y2 <= y1:
            continue

        local_mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        contour = tree.get("contour") or []
        if len(contour) >= 3:
            pts = np.array([[int(round(px - x1)), int(round(py - y1))] for px, py in contour], dtype=np.int32)
            cv2.fillPoly(local_mask, [pts], 255)
        else:
            cv2.circle(local_mask, (cx - x1, cy - y1), radius, 255, -1)

        mask = local_mask > 0
        if np.count_nonzero(mask) < 20:
            continue

        local_hsv = hsv[y1:y2, x1:x2]
        local_gray = gray[y1:y2, x1:x2]
        h_vals = local_hsv[:, :, 0][mask]
        s_vals = local_hsv[:, :, 1][mask]
        v_vals = local_hsv[:, :, 2][mask]
        gray_vals = local_gray[mask].astype(np.float32)

        green_ratio = float(np.count_nonzero((h_vals >= 25) & (h_vals <= 115) & (s_vals >= 28) & (v_vals >= 35))) / max(1, len(h_vals))
        mean_s = float(np.mean(s_vals))
        mean_v = float(np.mean(v_vals))
        texture = float(np.std(gray_vals))
        area_px = float(tree.get("area_px", 0) or np.count_nonzero(mask))

        if radius <= 13 or area_px < 360:
            species, species_key, confidence = "灌木/幼树", "shrub", 0.72
        elif green_ratio < 0.34:
            species, species_key, confidence = "稀疏冠层树", "sparse_crown", 0.66
        elif mean_v < 82 and mean_s >= 38:
            species, species_key, confidence = "暗绿阔叶树", "dark_crown", 0.68
        elif radius >= 34 and green_ratio >= 0.50 and texture >= 14:
            species, species_key, confidence = "成熟阔叶树", "broadleaf", 0.70
        else:
            species, species_key, confidence = "普通阔叶树", "broadleaf", 0.64

        tree["species"] = species
        tree["species_key"] = species_key
        tree["species_confidence"] = confidence
        tree["green_ratio"] = round(green_ratio, 3)
        tree["texture"] = round(texture, 1)

    return trees


def analyze_image_for_demo(image_path, trees):
    import cv2
    import numpy as np

    img_path = Path(image_path)
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    trees = refine_tree_species_from_image(img, trees)

    h, w = img.shape[:2]
    total_px = max(1, h * w)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    b_ch, g_ch, r_ch = cv2.split(img.astype(np.int16))
    exg = 2 * g_ch - r_ch - b_ch

    channel_delta = np.maximum.reduce([
        np.abs(r_ch - g_ch),
        np.abs(g_ch - b_ch),
        np.abs(r_ch - b_ch),
    ])
    gray_like = (
        (s_ch <= 42)
        & (v_ch >= 58)
        & (v_ch <= 220)
        & (channel_delta <= 26)
    ).astype(np.uint8) * 255
    road_seed = cv2.morphologyEx(
        gray_like,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23)),
        iterations=2,
    )
    road_seed = cv2.morphologyEx(
        road_seed,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
        iterations=1,
    )
    road_mask = np.zeros((h, w), dtype=np.uint8)
    num_road, road_labels, road_stats, _ = cv2.connectedComponentsWithStats(road_seed, connectivity=8)
    for label in range(1, num_road):
        x, y, bw, bh, area = road_stats[label].tolist()
        aspect = max(bw, bh) / max(1, min(bw, bh))
        if area > total_px * 0.004 or (area > 1200 and aspect > 4.0):
            road_mask[road_labels == label] = 255
    road_mask = cv2.dilate(
        road_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
        iterations=1,
    )
    # 可见光航拍里浅色裸地和水泥路很容易混淆；这里不把道路从覆盖率
    # 分母中扣除，只在高置信垃圾/火点规则里做弱过滤。
    road_filter_mask = np.zeros((h, w), dtype=np.uint8)
    road_mask = np.zeros((h, w), dtype=np.uint8)

    green_hue = (h_ch >= 25) & (h_ch <= 115)
    vegetation = (
        green_hue
        & (s_ch >= 28)
        & (v_ch >= 35)
        & (exg >= -2)
        & (g_ch >= r_ch - 8)
    )
    bare = (
        (~vegetation)
        & (v_ch >= 70)
        & (s_ch <= 95)
        & (exg < 12)
    )

    vegetation_mask = vegetation.astype(np.uint8) * 255
    bare_mask = bare.astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    vegetation_mask = cv2.morphologyEx(vegetation_mask, cv2.MORPH_OPEN, k, iterations=1)
    bare_mask = cv2.morphologyEx(bare_mask, cv2.MORPH_OPEN, k, iterations=1)

    land_px = total_px
    vegetation_px = int(np.count_nonzero(vegetation_mask))
    bare_px = int(np.count_nonzero(bare_mask))
    canopy_mask = np.zeros((h, w), dtype=np.uint8)
    for tree in trees:
        contour = tree.get("contour") or []
        if len(contour) >= 3:
            pts = np.array(contour, dtype=np.int32)
            cv2.fillPoly(canopy_mask, [pts], 255)
        else:
            cx = int(round(tree.get("x", 0)))
            cy = int(round(tree.get("y", 0)))
            radius = max(1, int(round(tree.get("radius", 0))))
            cv2.circle(canopy_mask, (cx, cy), radius, 255, -1)
    canopy_px = int(np.count_nonzero(canopy_mask))

    veg_pct = vegetation_px / land_px * 100.0
    bare_pct = bare_px / land_px * 100.0
    canopy_pct = min(100.0, canopy_px / land_px * 100.0)

    if bare_pct > 65 and veg_pct < 20:
        desert_level = "重度"
    elif bare_pct > 48 and veg_pct < 35:
        desert_level = "中度"
    elif bare_pct > 32:
        desert_level = "轻度"
    else:
        desert_level = "稳定"

    pixel_size_m = 0.10
    veg_area_ha = vegetation_px * pixel_size_m * pixel_size_m / 10000.0
    carbon_stock_tco2e = veg_area_ha * 18.0
    annual_sink_tco2e = veg_area_ha * 6.0

    species_counts = {}
    for tree in trees:
        species_counts[tree["species"]] = species_counts.get(tree["species"], 0) + 1
    dominant_species = "--"
    if species_counts:
        dominant_species = max(species_counts.items(), key=lambda item: item[1])[0]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 70, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8), iterations=1)

    soil_like = (
        (h_ch >= 8)
        & (h_ch <= 36)
        & (s_ch <= 115)
        & (v_ch >= 65)
        & (exg < 22)
    )
    white_plastic = (s_ch <= 48) & (v_ch >= 172) & (channel_delta <= 58)
    blue_plastic = (h_ch >= 88) & (h_ch <= 135) & (s_ch >= 55) & (v_ch >= 82)
    red_plastic = (((h_ch <= 8) | (h_ch >= 170)) & (s_ch >= 75) & (v_ch >= 90))
    color_anomaly = (
        (white_plastic | blue_plastic | red_plastic)
        & (~vegetation)
        & (~soil_like)
        & (road_filter_mask == 0)
    ).astype(np.uint8) * 255
    color_anomaly = cv2.morphologyEx(
        color_anomaly,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    )
    color_anomaly = cv2.morphologyEx(color_anomaly, cv2.MORPH_OPEN, k, iterations=1)

    waste_regions = []
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(color_anomaly, connectivity=8)
    min_area = max(180, int(total_px * 0.00012))
    max_area = int(total_px * 0.010)
    for label in range(1, num_labels):
        x, y, bw, bh, area = stats[label].tolist()
        if area < min_area or area > max_area:
            continue
        if x <= 3 or y <= 3 or x + bw >= w - 3 or y + bh >= h - 3:
            continue
        aspect = max(bw, bh) / max(1, min(bw, bh))
        if aspect > 3.2:
            continue

        component = labels[y:y + bh, x:x + bw] == label
        local_img = img[y:y + bh, x:x + bw]
        local_edges = edges[y:y + bh, x:x + bw]
        local_veg = vegetation_mask[y:y + bh, x:x + bw]
        local_road = road_filter_mask[y:y + bh, x:x + bw]
        local_bare = bare_mask[y:y + bh, x:x + bw]

        veg_ratio = float(np.count_nonzero(local_veg[component])) / max(1, area)
        road_ratio = float(np.count_nonzero(local_road[component])) / max(1, area)
        bare_ratio = float(np.count_nonzero(local_bare[component])) / max(1, area)
        edge_density = float(np.count_nonzero(local_edges[component])) / max(1, area)
        pixels = local_img[component].astype(np.float32)
        color_std = float(np.mean(np.std(pixels, axis=0))) if len(pixels) else 0.0
        compactness = float(area) / max(1, bw * bh)
        bright_ratio = float(np.count_nonzero(v_ch[y:y + bh, x:x + bw][component] > 170)) / max(1, area)

        if veg_ratio > 0.08 or road_ratio > 0.06 or bare_ratio > 0.52:
            continue
        if edge_density < 0.09 and color_std < 24.0:
            continue

        size_score = min(1.0, math.sqrt(area / max(1, max_area)))
        score = (
            0.22
            + min(0.20, edge_density * 1.2)
            + min(0.22, color_std / 120.0)
            + min(0.12, compactness * 0.18)
            + min(0.12, bright_ratio * 0.18)
            + min(0.10, size_score * 0.10)
            - bare_ratio * 0.18
        )
        if score < 0.72:
            continue

        waste_regions.append({
            "label": "疑似垃圾/堆放",
            "bbox": [int(x), int(y), int(bw), int(bh)],
            "area_px": int(area),
            "confidence": round(min(0.95, score), 2),
            "edge_density": round(edge_density, 3),
            "color_std": round(color_std, 1),
        })
    waste_regions = sorted(waste_regions, key=lambda item: item["confidence"], reverse=True)[:8]

    # 疑似起火点：红橙高亮区域。真实防火需要热红外/烟雾模型，这里是可视影像演示。
    fire_core = (
        (((h_ch <= 16) | (h_ch >= 172)) & (s_ch >= 115) & (v_ch >= 165) & (r_ch > g_ch + 38) & (r_ch > b_ch + 55))
        | ((h_ch >= 8) & (h_ch <= 28) & (s_ch >= 130) & (v_ch >= 180) & (r_ch > g_ch + 22) & (r_ch > b_ch + 70))
    )
    smoke_like = (
        (s_ch <= 45)
        & (v_ch >= 105)
        & (v_ch <= 218)
        & (channel_delta <= 32)
        & (road_filter_mask == 0)
    )
    fire_mask = (
        fire_core
        & (road_filter_mask == 0)
        & (~vegetation)
        & (~soil_like)
    ).astype(np.uint8) * 255
    fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8), iterations=1)

    fire_points = []
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fire_mask, connectivity=8)
    for label in range(1, num_labels):
        x, y, bw, bh, area = stats[label].tolist()
        if area < 14 or area > total_px * 0.006:
            continue
        cx, cy = centroids[label]
        pad = max(12, int(max(bw, bh) * 2.0))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + bw + pad)
        y2 = min(h, y + bh + pad)
        halo_area = max(1, (x2 - x1) * (y2 - y1))
        smoke_ratio = float(np.count_nonzero(smoke_like[y1:y2, x1:x2])) / halo_area
        component = labels[y:y + bh, x:x + bw] == label
        red_advantage = float(np.mean((r_ch[y:y + bh, x:x + bw] - np.maximum(g_ch[y:y + bh, x:x + bw], b_ch[y:y + bh, x:x + bw]))[component]))
        hot_brightness = float(np.mean(v_ch[y:y + bh, x:x + bw][component]))
        confidence = (
            0.34
            + min(0.24, (red_advantage - 35.0) / 120.0)
            + min(0.22, (hot_brightness - 160.0) / 170.0)
            + min(0.12, math.sqrt(area) / 65.0)
            + min(0.10, smoke_ratio * 1.8)
        )
        if confidence < 0.72:
            continue
        fire_points.append({
            "label": "疑似火点",
            "x": float(cx),
            "y": float(cy),
            "radius": int(max(16, min(50, math.sqrt(area / math.pi) * 3.0))),
            "area_px": int(area),
            "confidence": round(min(0.96, confidence), 2),
            "smoke_ratio": round(smoke_ratio, 3),
        })
    fire_points = sorted(fire_points, key=lambda item: item["confidence"], reverse=True)[:6]

    grid_rows, grid_cols = 6, 8
    cells = []
    for row in range(grid_rows):
        y1 = int(row * h / grid_rows)
        y2 = int((row + 1) * h / grid_rows)
        for col in range(grid_cols):
            x1 = int(col * w / grid_cols)
            x2 = int((col + 1) * w / grid_cols)
            cell_road = int(np.count_nonzero(road_mask[y1:y2, x1:x2]))
            cell_area = max(1, (y2 - y1) * (x2 - x1) - cell_road)
            cells.append({
                "row": row,
                "col": col,
                "vegetation": round(float(np.count_nonzero(vegetation_mask[y1:y2, x1:x2])) / cell_area, 3),
                "bare": round(float(np.count_nonzero(bare_mask[y1:y2, x1:x2])) / cell_area, 3),
            })

    return {
        "metrics": {
            "image_width": w,
            "image_height": h,
            "tree_count": len(trees),
            "vegetation_cover_pct": round(veg_pct, 2),
            "canopy_cover_pct": round(canopy_pct, 2),
            "bare_soil_pct": round(bare_pct, 2),
            "road_excluded_pct": round(float(np.count_nonzero(road_mask)) / total_px * 100.0, 2),
            "desertification_level": desert_level,
            "carbon_stock_tco2e": round(carbon_stock_tco2e, 2),
            "annual_sink_tco2e": round(annual_sink_tco2e, 2),
            "pixel_size_m_assumption": pixel_size_m,
        },
        "species": {
            "dominant": dominant_species,
            "counts": species_counts,
        },
        "waste_regions": waste_regions,
        "fire_points": fire_points,
        "coverage_grid": {
            "rows": grid_rows,
            "cols": grid_cols,
            "cells": cells,
        },
        "notes": [
            "垃圾识别、火点识别为可视影像启发式演示，生产环境建议接入专门检测模型和热红外/多光谱数据。",
            "碳汇估算默认按 0.10 米/像素和经验碳密度计算，需用无人机航高/GSD 或地块面积校准。",
        ],
    }


class DemoServer(BaseHTTPRequestHandler):
    def _send(self, status=200, content_type="text/plain; charset=utf-8", body=b""):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        query = urllib.parse.parse_qs(parsed_path.query)

        if path == "/":
            self._send(200, "text/html; charset=utf-8", INDEX_HTML.encode("utf-8"))
            return

        if path == "/api/config":
            self._send(200, "application/json; charset=utf-8", json.dumps({
                "default_image": DEFAULT_IMAGE_PATH,
                "default_data": DEFAULT_DATA_PATH,
                "demo_dir": str(DEMO_DIR).replace("\\", "/"),
            }).encode("utf-8"))
            return

        if path == "/api/manifest":
            self._send(200, "application/json; charset=utf-8", json.dumps({
                "items": build_manifest(),
            }).encode("utf-8"))
            return
            
        if path == "/image":
            img_file = query.get("file", ["DJI_0108.JPG"])[0]
            img_path = Path(img_file)
            if img_path.exists():
                ext = img_path.suffix.lower()
                content_type = "image/png" if ext == ".png" else "image/jpeg"
                self._send(200, content_type, img_path.read_bytes())
            else:
                self._send(404, "text/plain", b"Image not found")
            return
            
        if path == "/api/data":
            data_file = query.get("file", ["output/DJI_0108_trees.xlsx"])[0]
            image_file = query.get("image", [""])[0]
            trees = []
            try:
                trees = read_tree_rows(data_file)
                if image_file and Path(image_file).exists():
                    import cv2
                    trees = refine_tree_species_from_image(cv2.imread(str(image_file)), trees)
            except Exception as e:
                print(f"Error reading Excel: {e}")
            self._send(200, "application/json; charset=utf-8", json.dumps({"trees": trees}).encode("utf-8"))
            return

        if path == "/api/analysis":
            image_file = query.get("image", [DEFAULT_IMAGE_PATH])[0]
            data_file = query.get("data", [DEFAULT_DATA_PATH])[0]
            try:
                trees = read_tree_rows(data_file)
                analysis = analyze_image_for_demo(image_file, trees)
                self._send(200, "application/json; charset=utf-8", json.dumps(analysis).encode("utf-8"))
            except Exception as e:
                self._send(500, "application/json; charset=utf-8", json.dumps({
                    "error": str(e),
                }).encode("utf-8"))
            return
            
        self._send(404, "text/plain", b"Not found")

    def do_POST(self):
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        query = urllib.parse.parse_qs(parsed_path.query)

        if path == "/api/upload":
            try:
                filename = query.get("filename", ["upload.jpg"])[0]
                content_length = int(self.headers.get("Content-Length", 0))
                file_data = self.rfile.read(content_length)
                
                upload_dir = DEMO_DIR
                upload_dir.mkdir(exist_ok=True)
                
                # Save uploaded file
                safe_name = Path(filename).name
                img_path = upload_dir / safe_name
                img_path.write_bytes(file_data)

                # Normalize camera JPEG/PNG files so browsers can always decode them.
                try:
                    import cv2
                    normalized = cv2.imread(str(img_path))
                    if normalized is None:
                        raise ValueError("OpenCV cannot decode uploaded image")
                    ext = img_path.suffix.lower()
                    if ext in {".jpg", ".jpeg"}:
                        cv2.imwrite(str(img_path), normalized, [cv2.IMWRITE_JPEG_QUALITY, 94])
                    elif ext == ".png":
                        cv2.imwrite(str(img_path), normalized, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                except Exception as e:
                    print(f"Warning: image normalization skipped: {e}")
                
                # Run detection
                print(f"Running detection on {img_path}...")
                subprocess.run([
                    sys.executable, "detect_trees_opencv.py",
                    "--source", str(img_path),
                    "--out", str(upload_dir),
                    "--preset", "ensemble",
                    "--draw-contour",
                    "--work-max-dim", "2048",
                ], check=True)
                
                # Predict output paths
                base_name = img_path.stem
                data_path = upload_dir / f"{base_name}_trees.xlsx"
                
                self._send(200, "application/json", json.dumps({
                    "ok": True,
                    "img_path": str(img_path).replace('\\', '/'),
                    "data_path": str(data_path).replace('\\', '/')
                }).encode("utf-8"))
            except Exception as e:
                self._send(500, "application/json", json.dumps({
                    "ok": False,
                    "error": str(e)
                }).encode("utf-8"))
            return
            
        self._send(404, "text/plain", b"Not found")

    def log_message(self, format, *args):
        pass

def main():
    global DEMO_DIR, DEFAULT_IMAGE_PATH, DEFAULT_DATA_PATH

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--demo-dir", default="demo_half_labeled")
    parser.add_argument("--default-image", default="")
    parser.add_argument("--default-data", default="")
    args = parser.parse_args()

    DEMO_DIR = Path(args.demo_dir)
    manifest = build_manifest()
    if args.default_image:
        DEFAULT_IMAGE_PATH = args.default_image
    elif manifest:
        DEFAULT_IMAGE_PATH = manifest[0]["image_path"]

    if args.default_data:
        DEFAULT_DATA_PATH = args.default_data
    elif manifest:
        DEFAULT_DATA_PATH = manifest[0]["data_path"]
    
    server = ThreadingHTTPServer((args.host, args.port), DemoServer)
    print(f"\n=============================================")
    print(f"Visual demo server started: http://127.0.0.1:{args.port}")
    print(f"=============================================\n")
    server.serve_forever()

if __name__ == "__main__":
    main()
