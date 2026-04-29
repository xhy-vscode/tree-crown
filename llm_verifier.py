"""
大模型双重验证模块 — 使用 OpenAI GPT-4o Vision API
对 OpenCV 初筛的树冠候选进行二次验证：确认是否是树 + 识别树种。

环境变量：
    OPENAI_API_KEY: OpenAI API 密钥
    OPENAI_BASE_URL: (可选) 自定义 API 端点，用于代理或兼容服务
"""

import base64
import io
import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np

# openai 为可选依赖，延迟导入
_openai_client = None


def _get_openai_client():
    """延迟初始化 OpenAI 客户端。"""
    global _openai_client
    if _openai_client is not None:
        return _openai_client

    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "请先安装 openai 库: pip install openai>=1.30.0"
        )

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError(
            "请设置环境变量 OPENAI_API_KEY，例如：\n"
            "  Windows:  set OPENAI_API_KEY=sk-xxx\n"
            "  Linux:    export OPENAI_API_KEY=sk-xxx"
        )

    base_url = os.environ.get("OPENAI_BASE_URL", None)
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url

    _openai_client = OpenAI(**kwargs)
    return _openai_client


# ---------------------------------------------------------------------------
# 图像裁切与拼图
# ---------------------------------------------------------------------------

def crop_detection_region(img_bgr, detection, context_scale=1.6):
    """
    从原图中裁切单棵候选树的局部区域。

    Parameters
    ----------
    img_bgr : ndarray
        原始 BGR 图像
    detection : dict
        包含 x, y, radius 的检测结果
    context_scale : float
        裁切区域 = 检测圆的外接正方形 × context_scale（带上下文）

    Returns
    -------
    crop : ndarray
        裁切后的 BGR 图像
    """
    h, w = img_bgr.shape[:2]
    cx, cy, r = detection["x"], detection["y"], detection["radius"]
    margin = int(r * context_scale)

    x1 = max(0, cx - margin)
    y1 = max(0, cy - margin)
    x2 = min(w, cx + margin)
    y2 = min(h, cy + margin)

    crop = img_bgr[y1:y2, x1:x2].copy()

    # 在裁切图上画一个浅色圆圈标记候选区域，方便大模型定位
    local_cx = cx - x1
    local_cy = cy - y1
    cv2.circle(crop, (local_cx, local_cy), r, (0, 255, 0), 1)

    return crop


def build_grid_image(crops, labels, cols=3, cell_size=256):
    """
    将多张裁切图拼成网格，每个子图标注序号。

    Parameters
    ----------
    crops : list[ndarray]
        裁切图列表
    labels : list[str]
        每个子图的标签（序号）
    cols : int
        每行列数
    cell_size : int
        每个子图的统一尺寸

    Returns
    -------
    grid : ndarray
        拼接后的网格图（BGR）
    """
    n = len(crops)
    rows = math.ceil(n / cols)

    grid_h = rows * cell_size
    grid_w = cols * cell_size
    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 240  # 浅灰背景

    for idx, (crop, label) in enumerate(zip(crops, labels)):
        row, col = divmod(idx, cols)
        # 等比缩放到 cell_size
        ch, cw = crop.shape[:2]
        scale = min(cell_size / cw, cell_size / ch)
        new_w, new_h = int(cw * scale), int(ch * scale)
        resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # 居中放置
        y_off = row * cell_size + (cell_size - new_h) // 2
        x_off = col * cell_size + (cell_size - new_w) // 2
        grid[y_off:y_off + new_h, x_off:x_off + new_w] = resized

        # 标注序号（左上角）
        text_x = col * cell_size + 5
        text_y = row * cell_size + 22
        cv2.putText(
            grid, f"#{label}", (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA,
        )

    return grid


def encode_image_to_base64(img_bgr, quality=85):
    """将 BGR 图像编码为 base64 字符串。"""
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, buffer = cv2.imencode(".jpg", img_bgr, encode_params)
    return base64.b64encode(buffer).decode("utf-8")


# ---------------------------------------------------------------------------
# GPT-4o 验证
# ---------------------------------------------------------------------------

_VERIFY_PROMPT = """你是一位专业的林业遥感分析专家。下面是无人机俯拍的树冠局部裁切图拼成的网格，
每个子图的绿色圆圈标记了一个候选树冠区域，左上角有编号（如 #1, #2 ...）。

请对每个编号的候选区域做出判断：
1. 该圆圈标记的区域是否确实是一棵独立的树？（is_tree: true/false）
   - 如果是草斑、阴影、灌木、裸地等非树区域，请判定为 false
2. 如果是树，请推测可能的树种名称（species: 中文名称，如果无法判定就写"未知"）
3. 你的判断置信度（confidence: 0-100 整数）

【重要规则】
- 当前任务更怕漏掉树，而不是多保留几个可疑候选
- 只有在圆圈区域明确是道路、裸地、纯阴影、草斑等非树目标时，才返回 is_tree=false
- 对稀疏树冠、半遮挡树冠、暗绿树冠、边缘被截断的树冠，如果不能确定是非树，请返回 is_tree=true，并降低 confidence
- 从俯视角度看，树冠通常是圆形/椭圆形的绿色区域，有明显的树冠纹理
- 阴影是深黑色的，不是树本体
- 草地是大面积均匀的浅绿色，没有明显的树冠隆起感
- 如果圆圈区域明显不是树，confidence 设为 0

请严格按以下 JSON 数组格式返回（不要添加其他文字）：
[
  {"id": 1, "is_tree": true, "species": "树种名", "confidence": 85},
  {"id": 2, "is_tree": false, "species": "", "confidence": 95}
    ]"""


def _coerce_bool(value, default=True):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "y", "1", "是", "树"}:
            return True
        if normalized in {"false", "no", "n", "0", "否", "不是"}:
            return False
    return default


def _coerce_confidence(value, default=0):
    try:
        confidence = int(round(float(value)))
    except (TypeError, ValueError):
        confidence = default
    return max(0, min(100, confidence))


def _call_gpt4o(grid_image_b64, batch_ids, model="gpt-4o", max_retries=3):
    """
    调用 GPT-4o Vision API 验证一批候选树冠。

    Parameters
    ----------
    grid_image_b64 : str
        网格图的 base64 编码
    batch_ids : list[int]
        本批次候选树的 ID 列表
    model : str
        模型名称
    max_retries : int
        最大重试次数

    Returns
    -------
    results : list[dict]
        每棵候选树的验证结果
    """
    client = _get_openai_client()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": _VERIFY_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{grid_image_b64}",
                        "detail": "high",
                    },
                },
            ],
        }
    ]

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=1024,
                temperature=0.1,
            )
            text = response.choices[0].message.content.strip()
            return _parse_llm_response(text, batch_ids)

        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"  GPT-4o 调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                print(f"  {wait}秒后重试...")
                time.sleep(wait)
            else:
                print(f"  GPT-4o 调用最终失败: {e}")
                # 调用失败时保守返回：全部保留
                return [
                    {"id": bid, "is_tree": True, "species": "未知", "confidence": 0}
                    for bid in batch_ids
                ]


def _parse_llm_response(text, expected_ids):
    """
    解析大模型返回的 JSON 结果。

    对各种格式做兼容处理：
    - 纯 JSON 数组
    - 被 markdown 代码块包裹的 JSON
    - 格式异常时的回退处理
    """
    # 尝试提取 JSON（可能被 ```json ... ``` 包裹）
    json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if json_match:
        text = json_match.group(1).strip()

    # 尝试找到 JSON 数组
    bracket_match = re.search(r'\[.*\]', text, re.DOTALL)
    if bracket_match:
        text = bracket_match.group(0)

    try:
        results = json.loads(text)
        if isinstance(results, list):
            # 确保所有预期 ID 都有结果
            result_ids = {r.get("id") for r in results}
            for eid in expected_ids:
                if eid not in result_ids:
                    results.append({
                        "id": eid, "is_tree": True,
                        "species": "未知", "confidence": 0,
                    })
            return results
    except json.JSONDecodeError:
        pass

    # 解析失败，保守返回全部保留
    print(f"  警告：无法解析大模型返回内容，保守保留所有候选")
    return [
        {"id": bid, "is_tree": True, "species": "未知", "confidence": 0}
        for bid in expected_ids
    ]


# ---------------------------------------------------------------------------
# Tile 级修正与补漏
# ---------------------------------------------------------------------------

_REPAIR_PROMPT_TEMPLATE = """你是一位专业的林业遥感标注员。下面有同一块无人机图像的两张 tile：
1. 原图 tile：没有标注，用来观察真实树冠。
2. 候选叠加 tile：来自 OpenCV 的候选圆，红色编号是候选 id。

tile 信息：
- tile 左上角在原图中的坐标: x={tile_x}, y={tile_y}
- tile 尺寸: width={tile_w}, height={tile_h}
- 你返回的 x/y/radius 必须是 tile 内局部像素坐标，不是原图坐标。
- 当前 tile 内候选 id: {candidate_ids}
- 当前候选的局部坐标和半径: {candidate_table}

请完成四件事：
1. keep：保留确实是独立树冠的候选 id。
2. reject：删除明显不是树的候选，例如水面、道路、裸地、草带、纯阴影。
3. adjust：如果候选是树但中心或半径明显偏了，给出修正后的局部 x/y/radius。
4. add：补出 OpenCV 漏掉的独立树冠，给出局部 x/y/radius。

判断规则：
- 每一个当前候选 id 必须且只能出现在 keep、reject、adjust 三者之一，不要省略。
- 对候选要严格一点：圆心落在阴影、裸地、道路、水面、草带、树间空地时 reject；圆太大圈进多个树或大片空地时 adjust。
- 对真实树冠宁愿 adjust 成贴合的小圆，不要保留明显偏大的旧圆。
- add 只补明显独立的树冠，不要给草地纹理、芦苇带、水面反光加圈。
- 对沿道路、规则果园行列中的圆形/椭圆绿色冠层要积极补漏。
- 半径要贴近树冠，不要把相邻树、阴影或空地一起圈进去。
- 如果一个树冠只有很小一部分出现在 tile 边缘，可以不 add，交给相邻 tile。

请只返回 JSON，不要加解释。格式如下：
{{
  "keep": [{{"id": 1, "confidence": 90, "species": "未知"}}],
  "reject": [{{"id": 2, "confidence": 95, "reason": "water/road/grass/shadow/bare_soil"}}],
  "adjust": [{{"id": 3, "x": 120, "y": 240, "radius": 22, "confidence": 88, "species": "未知"}}],
  "add": [{{"x": 300, "y": 410, "radius": 24, "confidence": 82, "species": "未知"}}]
}}"""


def generate_tiles(width, height, tile_size, overlap):
    """生成带重叠的 tile，返回 (x, y, w, h)。"""
    stride = max(1, int(tile_size * (1.0 - overlap)))
    xs = list(range(0, max(1, width - tile_size + 1), stride))
    ys = list(range(0, max(1, height - tile_size + 1), stride))

    if not xs or xs[-1] != max(0, width - tile_size):
        xs.append(max(0, width - tile_size))
    if not ys or ys[-1] != max(0, height - tile_size):
        ys.append(max(0, height - tile_size))

    seen = set()
    for y in ys:
        for x in xs:
            key = (x, y)
            if key in seen:
                continue
            seen.add(key)
            yield x, y, min(tile_size, width - x), min(tile_size, height - y)


def detections_for_tile(detections, tile):
    """选择中心点落在 tile 内的候选。"""
    tx, ty, tw, th = tile
    out = []
    for det in detections:
        x = int(det.get("x", -1))
        y = int(det.get("y", -1))
        if tx <= x < tx + tw and ty <= y < ty + th:
            out.append(det)
    return out


def draw_tile_candidate_overlay(tile_bgr, tile, candidates):
    """在 tile 上画 OpenCV 候选圆和 id，供 LLM 对照原图修正。"""
    tx, ty, _, _ = tile
    out = tile_bgr.copy()
    for det in candidates:
        cx = int(round(det["x"] - tx))
        cy = int(round(det["y"] - ty))
        radius = int(round(det["radius"]))
        cv2.circle(out, (cx, cy), radius, (0, 0, 255), 2)
        cv2.circle(out, (cx, cy), 2, (0, 255, 255), -1)

        label = str(det["id"])
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        text_x = max(2, min(out.shape[1] - tw - 3, cx + 3))
        text_y = max(th + 3, min(out.shape[0] - 3, cy - 3))
        cv2.rectangle(
            out,
            (text_x - 2, text_y - th - 3),
            (text_x + tw + 2, text_y + 3),
            (255, 255, 255),
            -1,
        )
        cv2.putText(
            out,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return out


def _parse_llm_repair_response(text):
    """解析 tile repair 返回的 JSON 对象。"""
    json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if json_match:
        text = json_match.group(1).strip()

    object_match = re.search(r'\{.*\}', text, re.DOTALL)
    if object_match:
        text = object_match.group(0)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        print("  警告：无法解析 tile 修正结果，跳过该 tile")
        return {"keep": [], "reject": [], "adjust": [], "add": []}

    if isinstance(data, list):
        data = {"keep": [], "reject": [], "adjust": [], "add": data}
    if not isinstance(data, dict):
        return {"keep": [], "reject": [], "adjust": [], "add": []}

    normalized = {}
    for key in ("keep", "reject", "adjust", "add"):
        value = data.get(key, [])
        normalized[key] = value if isinstance(value, list) else []
    return normalized


def _call_gpt4o_repair(
    tile_b64,
    overlay_b64,
    tile,
    candidate_ids,
    candidate_table,
    model="gpt-4o",
    max_retries=3,
):
    """调用多模态模型对一个 tile 做候选修正和补漏。"""
    client = _get_openai_client()
    tx, ty, tw, th = tile
    prompt = _REPAIR_PROMPT_TEMPLATE.format(
        tile_x=tx,
        tile_y=ty,
        tile_w=tw,
        tile_h=th,
        candidate_ids=candidate_ids,
        candidate_table=candidate_table,
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{tile_b64}",
                        "detail": "high",
                    },
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{overlay_b64}",
                        "detail": "high",
                    },
                },
            ],
        }
    ]

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=4096,
                temperature=0.05,
            )
            text = response.choices[0].message.content.strip()
            return _parse_llm_repair_response(text)
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"  tile 修正调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                print(f"  {wait}秒后重试...")
                time.sleep(wait)
            else:
                print(f"  tile 修正调用最终失败: {e}")
                return {"keep": [], "reject": [], "adjust": [], "add": []}


def _safe_int(value, default=0):
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def _local_circle_to_global(item, tile, min_radius, max_radius):
    tx, ty, tw, th = tile
    x = _safe_int(item.get("x"), -1)
    y = _safe_int(item.get("y"), -1)
    radius = _safe_int(item.get("radius"), 0)
    if x < 0 or y < 0 or x >= tw or y >= th:
        return None
    if radius < min_radius or radius > max_radius:
        return None

    return {
        "x": int(tx + x),
        "y": int(ty + y),
        "radius": int(radius),
        "species": item.get("species", "未知") or "未知",
        "llm_confidence": _coerce_confidence(item.get("confidence", 0)),
    }


def _circle_overlap_area(r1, r2, distance):
    if distance >= r1 + r2:
        return 0.0
    if distance <= abs(r1 - r2):
        return math.pi * min(r1, r2) ** 2
    if distance <= 0:
        return math.pi * min(r1, r2) ** 2

    a1 = math.acos((distance * distance + r1 * r1 - r2 * r2) / (2 * distance * r1))
    a2 = math.acos((distance * distance + r2 * r2 - r1 * r1) / (2 * distance * r2))
    a3 = 0.5 * math.sqrt(
        max(
            0.0,
            (-distance + r1 + r2)
            * (distance + r1 - r2)
            * (distance - r1 + r2)
            * (distance + r1 + r2),
        )
    )
    return r1 * r1 * a1 + r2 * r2 * a2 - a3


def _is_same_tree(a, b, overlap_threshold=0.58):
    dx = float(a["x"] - b["x"])
    dy = float(a["y"] - b["y"])
    distance = math.hypot(dx, dy)
    min_radius = max(1.0, float(min(a["radius"], b["radius"])))
    if distance <= max(5.0, min_radius * 0.45):
        return True

    overlap = _circle_overlap_area(float(a["radius"]), float(b["radius"]), distance)
    smaller = math.pi * min_radius * min_radius
    return overlap / smaller >= overlap_threshold


def _dedupe_repaired_detections(detections, overlap_threshold=0.58):
    def rank(det):
        source = det.get("source", "")
        source_rank = {
            "llm_adjust": 5,
            "llm_add": 4,
            "llm_keep": 3,
            "opencv": 2,
        }.get(source, 0)
        return (source_rank, int(det.get("llm_confidence", 0) or 0), int(det.get("radius", 0)))

    kept = []
    for det in sorted(detections, key=rank, reverse=True):
        duplicate = False
        for idx, existing in enumerate(kept):
            if _is_same_tree(det, existing, overlap_threshold=overlap_threshold):
                duplicate = True
                if rank(det) > rank(existing):
                    kept[idx] = det
                break
        if not duplicate:
            kept.append(det)

    kept.sort(key=lambda d: (d["y"], d["x"]))
    for idx, det in enumerate(kept, 1):
        det["id"] = idx
    return kept


def repair_detections_with_tiles(
    img_bgr,
    detections,
    model="gpt-4o",
    tile_size=1024,
    overlap=0.25,
    min_confidence=30,
    reject_confidence=85,
    conservative=True,
    min_radius=5,
    max_radius=80,
    dedupe_overlap=0.58,
    strict_candidates=True,
    max_workers=1,
    verbose=True,
):
    """
    使用原图 tile + OpenCV 候选叠加图，让 LLM 删除、修正并补漏树冠。
    """
    _get_openai_client()

    h, w = img_bgr.shape[:2]
    tiles = list(generate_tiles(w, h, tile_size, overlap))

    if verbose:
        print(f"\n{'='*50}")
        print(f"大模型修正补漏：{len(detections)} 个 OpenCV 候选，{len(tiles)} 个 tile")
        print(f"tile_size={tile_size}, overlap={overlap:.2f}, model={model}")
        print(f"{'='*50}")

    det_by_id = {int(det["id"]): dict(det) for det in detections}

    jobs = []
    for tile_idx, tile in enumerate(tiles):
        tx, ty, tw, th = tile
        tile_img = img_bgr[ty:ty + th, tx:tx + tw]
        tile_candidates = detections_for_tile(detections, tile)
        overlay = draw_tile_candidate_overlay(tile_img, tile, tile_candidates)
        tile_b64 = encode_image_to_base64(tile_img)
        overlay_b64 = encode_image_to_base64(overlay)
        candidate_ids = [int(det["id"]) for det in tile_candidates]
        candidate_table = [
            {
                "id": int(det["id"]),
                "x": int(round(det["x"] - tx)),
                "y": int(round(det["y"] - ty)),
                "radius": int(round(det["radius"])),
            }
            for det in tile_candidates
        ]
        jobs.append((tile_idx, tile, candidate_ids, candidate_table, tile_b64, overlay_b64))

    tile_results = []
    if max_workers <= 1:
        for tile_idx, tile, candidate_ids, candidate_table, tile_b64, overlay_b64 in jobs:
            if verbose:
                print(f"  修正 tile {tile_idx + 1}/{len(jobs)}，候选 {len(candidate_ids)} 个")
            result = _call_gpt4o_repair(
                tile_b64,
                overlay_b64,
                tile,
                candidate_ids,
                candidate_table,
                model=model,
            )
            tile_results.append((tile, candidate_ids, result))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for tile_idx, tile, candidate_ids, candidate_table, tile_b64, overlay_b64 in jobs:
                future = executor.submit(
                    _call_gpt4o_repair,
                    tile_b64,
                    overlay_b64,
                    tile,
                    candidate_ids,
                    candidate_table,
                    model,
                )
                futures[future] = (tile_idx, tile, candidate_ids)

            for future in as_completed(futures):
                tile_idx, tile, candidate_ids = futures[future]
                if verbose:
                    print(f"  tile {tile_idx + 1}/{len(jobs)} 完成，候选 {len(candidate_ids)} 个")
                try:
                    result = future.result()
                except Exception as e:
                    print(f"  tile {tile_idx + 1} 异常: {e}")
                    result = {"keep": [], "reject": [], "adjust": [], "add": []}
                tile_results.append((tile, candidate_ids, result))

    rejected_ids = set()
    adjusted_by_id = {}
    kept_ids = set()
    reviewed_ids = set()
    strict_reviewed_scope = set()
    added = []

    for tile, candidate_ids, result in tile_results:
        candidate_id_set = set(candidate_ids)
        candidate_actions = (
            result.get("keep", [])
            + result.get("reject", [])
            + result.get("adjust", [])
        )
        if candidate_actions:
            strict_reviewed_scope.update(candidate_id_set)

        for item in result.get("keep", []):
            det_id = _safe_int(item.get("id"), -1)
            if det_id in candidate_id_set:
                reviewed_ids.add(det_id)
                kept_ids.add(det_id)
                if det_id in det_by_id:
                    det_by_id[det_id]["species"] = item.get("species", det_by_id[det_id].get("species", "未知"))
                    det_by_id[det_id]["llm_confidence"] = max(
                        int(det_by_id[det_id].get("llm_confidence", 0) or 0),
                        _coerce_confidence(item.get("confidence", 0)),
                    )

        for item in result.get("reject", []):
            det_id = _safe_int(item.get("id"), -1)
            confidence = _coerce_confidence(item.get("confidence", 0))
            if det_id in candidate_id_set:
                reviewed_ids.add(det_id)
                if conservative and confidence < reject_confidence:
                    kept_ids.add(det_id)
                    if det_id in det_by_id:
                        det_by_id[det_id]["llm_decision"] = "kept_uncertain_non_tree"
                        det_by_id[det_id]["llm_confidence"] = confidence
                else:
                    rejected_ids.add(det_id)
                    if det_id in det_by_id:
                        det_by_id[det_id]["llm_reject_reason"] = item.get("reason", "")
                        det_by_id[det_id]["llm_confidence"] = confidence

        for item in result.get("adjust", []):
            det_id = _safe_int(item.get("id"), -1)
            confidence = _coerce_confidence(item.get("confidence", 0))
            if det_id not in candidate_id_set or confidence < min_confidence:
                continue
            reviewed_ids.add(det_id)
            circle = _local_circle_to_global(item, tile, min_radius, max_radius)
            if circle is None:
                continue
            adjusted = dict(det_by_id.get(det_id, {}))
            adjusted.update(circle)
            adjusted["source"] = "llm_adjust"
            adjusted["llm_decision"] = "adjusted_tree"
            adjusted_by_id[det_id] = adjusted
            kept_ids.add(det_id)

        for item in result.get("add", []):
            confidence = _coerce_confidence(item.get("confidence", 0))
            if confidence < min_confidence:
                continue
            circle = _local_circle_to_global(item, tile, min_radius, max_radius)
            if circle is None:
                continue
            circle.update({
                "source": "llm_add",
                "area_px": round(math.pi * circle["radius"] * circle["radius"], 2),
                "llm_decision": "added_tree",
            })
            added.append(circle)

    repaired = []
    rejected = []
    for det in detections:
        det_id = int(det["id"])
        if det_id in rejected_ids:
            rejected_det = dict(det_by_id.get(det_id, det))
            rejected_det["llm_decision"] = "rejected_non_tree"
            rejected.append(rejected_det)
            continue

        if strict_candidates and det_id in strict_reviewed_scope and det_id not in reviewed_ids:
            rejected_det = dict(det_by_id.get(det_id, det))
            rejected_det["llm_decision"] = "rejected_unreviewed_by_repair"
            rejected.append(rejected_det)
            continue

        if det_id in adjusted_by_id:
            repaired.append(adjusted_by_id[det_id])
            continue

        kept = dict(det_by_id.get(det_id, det))
        kept.setdefault("source", "llm_keep" if det_id in kept_ids else "opencv")
        kept.setdefault("species", "未知")
        kept.setdefault("llm_confidence", 0)
        kept.setdefault("llm_decision", "kept_by_repair")
        repaired.append(kept)

    before_dedupe = len(repaired) + len(added)
    repaired = _dedupe_repaired_detections(
        repaired + added,
        overlap_threshold=dedupe_overlap,
    )

    if verbose:
        adjusted_count = len(adjusted_by_id)
        added_count = sum(1 for det in repaired if det.get("source") == "llm_add")
        print("\n修正补漏结果：")
        print(f"  删除误检: {len(rejected)} 个")
        print(f"  调整候选: {adjusted_count} 个")
        print(f"  补新增树: {added_count} 个")
        print(f"  去重前候选: {before_dedupe} 个")
        print(f"  最终树冠: {len(repaired)} 棵")
        print(f"{'='*50}\n")

    return repaired, rejected


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def verify_detections(
    img_bgr,
    detections,
    batch_size=6,
    model="gpt-4o",
    context_scale=1.6,
    cell_size=256,
    min_confidence=30,
    conservative=True,
    reject_confidence=85,
    max_workers=3,
    verbose=True,
):
    """
    使用 GPT-4o 对 OpenCV 检测的候选树冠进行二次验证。

    Parameters
    ----------
    img_bgr : ndarray
        原始 BGR 图像
    detections : list[dict]
        OpenCV 检测结果列表，每个元素包含 id, x, y, radius 等
    batch_size : int
        每批发送几棵树（拼成一张网格图）
    model : str
        OpenAI 模型名称
    context_scale : float
        裁切区域放大系数
    cell_size : int
        网格图中每个子图的尺寸
    min_confidence : int
        大模型判定为树的最低置信度，低于此值标记为低置信保留
    conservative : bool
        是否启用保守过滤。启用时，低置信的非树判断会保留，避免漏检扩大
    reject_confidence : int
        保守模式下，非树判断达到该置信度才真正过滤
    max_workers : int
        并发请求数
    verbose : bool
        是否打印进度

    Returns
    -------
    verified : list[dict]
        验证通过的检测结果（增加 species, llm_confidence 字段）
    rejected : list[dict]
        被大模型过滤掉的检测结果
    """
    if not detections:
        return [], []

    # 预检 API Key，提前失败而不是每个批次都报错
    _get_openai_client()

    if verbose:
        print(f"\n{'='*50}")
        print(f"大模型验证：共 {len(detections)} 个候选树冠")
        print(f"批量大小: {batch_size}，预计 API 调用: {math.ceil(len(detections) / batch_size)} 次")
        print(f"模型: {model}")
        print(f"{'='*50}")

    # 1. 裁切所有候选树的局部图像
    crops = []
    for det in detections:
        crop = crop_detection_region(img_bgr, det, context_scale)
        crops.append(crop)

    # 2. 分批构建网格图并调用 API
    batches = []
    for i in range(0, len(detections), batch_size):
        batch_dets = detections[i:i + batch_size]
        batch_crops = crops[i:i + batch_size]
        batch_ids = [d["id"] for d in batch_dets]
        batch_labels = [str(d["id"]) for d in batch_dets]

        cols = min(3, len(batch_crops))
        grid = build_grid_image(batch_crops, batch_labels, cols=cols, cell_size=cell_size)
        grid_b64 = encode_image_to_base64(grid)

        batches.append((grid_b64, batch_ids))

    # 3. 并发调用 API
    all_results = {}

    if max_workers <= 1:
        # 串行执行
        for batch_idx, (grid_b64, batch_ids) in enumerate(batches):
            if verbose:
                print(f"  验证批次 {batch_idx + 1}/{len(batches)}，"
                      f"候选 #{batch_ids[0]}-#{batch_ids[-1]}...")
            results = _call_gpt4o(grid_b64, batch_ids, model=model)
            for r in results:
                all_results[r["id"]] = r
    else:
        # 并发执行
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for batch_idx, (grid_b64, batch_ids) in enumerate(batches):
                future = executor.submit(_call_gpt4o, grid_b64, batch_ids, model=model)
                futures[future] = (batch_idx, batch_ids)

            for future in as_completed(futures):
                batch_idx, batch_ids = futures[future]
                if verbose:
                    print(f"  批次 {batch_idx + 1}/{len(batches)} 完成，"
                          f"候选 #{batch_ids[0]}-#{batch_ids[-1]}")
                try:
                    results = future.result()
                    for r in results:
                        all_results[r["id"]] = r
                except Exception as e:
                    print(f"  批次 {batch_idx + 1} 异常: {e}")
                    for bid in batch_ids:
                        all_results[bid] = {
                            "id": bid, "is_tree": True,
                            "species": "未知", "confidence": 0,
                        }

    # 4. 合并结果
    verified = []
    rejected = []

    for det in detections:
        det_id = det["id"]
        llm_result = all_results.get(det_id, {
            "is_tree": True, "species": "未知", "confidence": 0,
        })

        det_copy = dict(det)
        confidence = _coerce_confidence(llm_result.get("confidence", 0))
        is_tree = _coerce_bool(llm_result.get("is_tree", True), default=True)

        det_copy["species"] = llm_result.get("species", "未知")
        det_copy["llm_confidence"] = confidence
        det_copy["llm_is_tree"] = is_tree

        if is_tree and confidence < min_confidence:
            det_copy["llm_decision"] = "kept_low_confidence_tree"
        elif is_tree:
            det_copy["llm_decision"] = "verified_tree"
        elif conservative and confidence < reject_confidence:
            det_copy["llm_decision"] = "kept_uncertain_non_tree"
        else:
            det_copy["llm_decision"] = "rejected_non_tree"

        if det_copy["llm_decision"] != "rejected_non_tree":
            verified.append(det_copy)
        else:
            rejected.append(det_copy)

    # 重新编号
    for idx, det in enumerate(verified, 1):
        det["id"] = idx

    if verbose:
        print(f"\n验证结果：")
        print(f"  确认是树: {len(verified)} 棵")
        print(f"  过滤误检: {len(rejected)} 个")
        if conservative:
            kept_uncertain = sum(
                1 for det in verified
                if det.get("llm_decision") == "kept_uncertain_non_tree"
            )
            if kept_uncertain:
                print(f"  保守保留不确定候选: {kept_uncertain} 个")

        # 统计树种
        species_count = {}
        for det in verified:
            sp = det.get("species", "未知")
            species_count[sp] = species_count.get(sp, 0) + 1
        if species_count:
            print(f"  树种分布:")
            for sp, count in sorted(species_count.items(), key=lambda x: -x[1]):
                print(f"    {sp}: {count} 棵")
        print(f"{'='*50}\n")

    return verified, rejected


def draw_verified_detections(
    img_bgr,
    verified,
    rejected=None,
    show_species=True,
    show_id=False,
    line_color=(0, 255, 0),
    line_thickness=2,
    rejected_color=(0, 0, 255),
    font_scale=0.35,
):
    """
    绘制经大模型验证后的检测结果。

    Parameters
    ----------
    img_bgr : ndarray
        原始 BGR 图像
    verified : list[dict]
        验证通过的检测列表
    rejected : list[dict]
        被过滤的检测列表（用红色虚圈标记）
    show_species : bool
        是否在图上标注树种名称
    show_id : bool
        是否在图上显示编号
    line_color : tuple
        确认树冠的圆圈颜色
    line_thickness : int
        圆圈线宽
    rejected_color : tuple
        被过滤候选的颜色
    font_scale : float
        文字大小

    Returns
    -------
    out : ndarray
        标注后的图像
    """
    out = img_bgr.copy()

    # 先画被过滤的（红色虚圈）
    if rejected:
        for det in rejected:
            center = (det["x"], det["y"])
            radius = det["radius"]
            # 用虚线效果：画短线段
            _draw_dashed_circle(out, center, radius, rejected_color, 1)

    # 再画确认的树冠
    for det in verified:
        center = (det["x"], det["y"])
        radius = det["radius"]

        cv2.circle(out, center, radius, line_color, line_thickness)
        cv2.circle(out, center, 2, (0, 200, 0), -1)

        # 标注树种名称
        label_parts = []
        if show_id:
            label_parts.append(str(det["id"]))
        if show_species and det.get("species") and det["species"] != "未知":
            label_parts.append(det["species"])

        if label_parts:
            label = " ".join(label_parts)
            text_x = det["x"] + radius + 3
            text_y = det["y"] - 3

            # 文字背景
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )
            cv2.rectangle(
                out,
                (text_x - 1, text_y - th - 2),
                (text_x + tw + 1, text_y + 2),
                (0, 0, 0), -1,
            )
            cv2.putText(
                out, label, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (255, 255, 255), 1, cv2.LINE_AA,
            )

    return out


def _draw_dashed_circle(img, center, radius, color, thickness, dash_len=8):
    """绘制虚线圆。"""
    circumference = 2 * math.pi * radius
    n_dashes = max(8, int(circumference / (dash_len * 2)))

    for i in range(n_dashes):
        angle_start = (2 * math.pi * i) / n_dashes
        angle_end = (2 * math.pi * (i + 0.5)) / n_dashes

        pt1 = (
            int(center[0] + radius * math.cos(angle_start)),
            int(center[1] + radius * math.sin(angle_start)),
        )
        pt2 = (
            int(center[0] + radius * math.cos(angle_end)),
            int(center[1] + radius * math.sin(angle_end)),
        )
        cv2.line(img, pt1, pt2, color, thickness, cv2.LINE_AA)
