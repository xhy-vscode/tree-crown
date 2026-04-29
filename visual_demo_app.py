import argparse
import json
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
    <title>AI 树冠检测 - 智能视界 (Visual Demo)</title>
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
            <span class="subtitle">Vision System</span>
            <h1>Tree Crown<br>Analytics</h1>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value" id="totalCount">0</div>
                <div class="stat-label">总检测数量</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="avgRadius">0</div>
                <div class="stat-label">平均半径 (px)</div>
            </div>
        </div>
        
        <div class="solution-box">
            <h3>多策略融合检测方案</h3>
            <p>采用 <b>Ensemble 多策略融合</b>（参考 + 高召回 + 掩膜补漏 + 峰值修复），经智能去重和本地树冠核心过滤后生成多边形轮廓。大图会先在稳定工作分辨率下检测，再映射回原图坐标。</p>
        </div>

        <div class="dataset-panel">
            <div class="dataset-head">
                <span id="datasetCount">DEMO 0/0</span>
                <span id="datasetName">未加载</span>
            </div>
            <select id="imageSelect" aria-label="选择演示图片"></select>
        </div>
        
        <div class="legend">
            <div class="legend-title">检测源分布</div>
            <div class="legend-item">
                <div class="color-dot" style="color: #00ff66; background: #00ff66;"></div>
                <span>Reference (高置信度核心圈)</span>
            </div>
            <div class="legend-item">
                <div class="color-dot" style="color: #ffcc00; background: #ffcc00;"></div>
                <span>Recall (高召回补充)</span>
            </div>
            <div class="legend-item">
                <div class="color-dot" style="color: #00f0ff; background: #00f0ff;"></div>
                <span>Residual (掩膜补漏)</span>
            </div>
            <div class="legend-item">
                <div class="color-dot" style="color: #ff3366; background: #ff3366;"></div>
                <span>Peak (距离峰值盲区修复)</span>
            </div>
        </div>
        
        <div class="controls">
            <button id="btnPulse" class="active">动态呼吸</button>
            <button id="btnShape" class="active">轮廓标注</button>
            <button id="btnFit">适应窗口</button>
        </div>
        <button id="btnUpload" class="upload-btn">⇧ 上传并检测新图片</button>
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
        <div class="loading-text">AI 正在提取树冠轮廓...</div>
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
    let time = 0;
    let hoveredTree = null;
    let renderFrame = null;

    // Helper functions
    const colors = {
        'reference': '#00ff66',
        'recall': '#ffcc00',
        'residual': '#00f0ff',
        'peak': '#ff3366'
    };
    
    function getColor(source) {
        return colors[source] || '#ffffff';
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
            const res = await fetch(`/api/data?file=${encodeURIComponent(currentDataPath)}`);
            if(res.ok) {
                const data = await res.json();
                trees = data.trees;
                document.getElementById('totalCount').innerText = trees.length;
                if(trees.length > 0) {
                    const avg = trees.reduce((acc, t) => acc + t.radius, 0) / trees.length;
                    document.getElementById('avgRadius').innerText = avg.toFixed(1);
                } else {
                    document.getElementById('avgRadius').innerText = '0';
                }
            } else {
                trees = [];
            }
        } catch(e) {
            trees = [];
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
            document.getElementById('tt-title').style.color = getColor(found.source);
            document.getElementById('tooltip').style.borderColor = getColor(found.source);
            
            document.getElementById('tt-source').innerText = found.source || 'unknown';
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

    // Render Loop
    function render() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Draw dark overlay to make colors pop
        ctx.fillStyle = 'rgba(0,0,0,0.15)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        time += 0.05;
        const pulse = enablePulse ? (Math.sin(time) * 0.2 + 0.8) : 1;
        
        // Draw trees
        trees.forEach(t => {
            const color = getColor(t.source);
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
        this.innerText = useContours ? '轮廓标注' : '圆形标注';
    };
    
    document.getElementById('btnFit').onclick = fitToScreen;

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
            excel_path = Path(data_file)
            trees = []
            if excel_path.exists():
                try:
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

                        trees.append({
                            "id": int(row.get("id", idx+1)),
                            "x": float(row["x"]),
                            "y": float(row["y"]),
                            "radius": float(row["radius"]),
                            "source": str(row.get("source", "reference")),
                            "area_px": float(row.get("area_px", 0)),
                            "contour": contour,
                        })
                except Exception as e:
                    print(f"Error reading Excel: {e}")
            self._send(200, "application/json; charset=utf-8", json.dumps({"trees": trees}).encode("utf-8"))
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
