import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def build_tree_mask(img_bgr, green_s=25, green_v=40, exg_min=10):
    """
    基于 HSV + ExG 植被指数提取绿色树冠。
    对你的这种无人机俯拍果园图，未训练模型也能做演示。
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # HSV 中绿色范围。H: 28~105 覆盖树叶从黄绿到深绿的变化。
    lower = np.array([28, green_s, green_v], dtype=np.uint8)
    upper = np.array([105, 255, 255], dtype=np.uint8)
    hsv_mask = cv2.inRange(hsv, lower, upper)

    # ExG = 2G - R - B，直接用原始差值过滤灰色道路、阴影和裸土。
    b, g, r = cv2.split(img_bgr.astype(np.int16))
    exg = 2 * g - r - b
    exg_mask = (exg >= exg_min).astype(np.uint8) * 255

    mask = cv2.bitwise_and(hsv_mask, exg_mask)

    # 去噪 + 适度填补树冠内部空洞，用更小的闭核避免把纹理噪点也扩大成树冠。
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    return mask


def build_tree_mask_v2(img_bgr, green_s=30, green_v=50, exg_min=15,
                      use_lab=True, shadow_filter=True, min_blob=60):
    """
    增强版树冠掩膜：HSV + Lab(a通道) + ExG 三通道融合，
    附加阴影排除、大核闭运算和膨胀，让掩膜完整覆盖整个树冠。
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)

    # --- 通道 1：HSV 绿色范围 ---
    lower = np.array([25, green_s, green_v], dtype=np.uint8)
    upper = np.array([110, 255, 255], dtype=np.uint8)
    hsv_mask = cv2.inRange(hsv, lower, upper)

    # --- 通道 2：ExG 植被指数 ---
    b, g, r = cv2.split(img_bgr.astype(np.int16))
    exg = 2 * g - r - b
    exg_mask = (exg >= exg_min).astype(np.uint8) * 255

    # --- 通道 3：Lab a 通道（绿色为负值）---
    if use_lab:
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
        l_ch, a_ch, _ = cv2.split(lab)
        # a 通道 <122 偏绿（128 为中性点），稍宽松
        lab_mask = (a_ch < 122).astype(np.uint8) * 255
    else:
        lab_mask = np.ones(hsv_mask.shape, dtype=np.uint8) * 255

    # --- 三通道融合：至少两个通道同意即可 ---
    vote = (hsv_mask // 255).astype(np.uint8) + \
           (exg_mask // 255).astype(np.uint8) + \
           (lab_mask // 255).astype(np.uint8)
    mask = (vote >= 2).astype(np.uint8) * 255

    # --- 阴影排除：只剔除非常暗的区域（避免误删树冠暗部）---
    if shadow_filter:
        shadow = (v_ch < 40).astype(np.uint8) * 255
        if use_lab:
            shadow_lab = (l_ch < 35).astype(np.uint8) * 255
            shadow = cv2.bitwise_or(shadow, shadow_lab)
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(shadow))

    # --- 形态学处理：去噪 → 中大核闭运算填补树冠内部空洞 ---
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open, iterations=1)

    # 两级闭运算：填补树冠内部空洞，但不过度连接相邻树冠
    k_close_1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    k_close_2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close_1, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close_2, iterations=1)

    # --- 小块移除：面积过小的连通域直接清除 ---
    if min_blob > 0:
        num_labels, labels_map, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_blob:
                mask[labels_map == i] = 0

    return mask


def build_tree_mask_original(img_bgr, green_s=35, green_v=45):
    """
    最初版本的树冠掩膜：HSV + 归一化 ExG Otsu。
    这套逻辑误检偏少，作为默认基线保留。
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    lower = np.array([30, green_s, green_v], dtype=np.uint8)
    upper = np.array([100, 255, 255], dtype=np.uint8)
    hsv_mask = cv2.inRange(hsv, lower, upper)

    b, g, r = cv2.split(img_bgr.astype(np.int16))
    exg = 2 * g - r - b
    exg = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, exg_mask = cv2.threshold(exg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    mask = cv2.bitwise_and(hsv_mask, exg_mask)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    return mask


def build_dark_crown_mask(img_bgr):
    """
    参考图里有一部分树冠偏暗、偏灰绿，单纯绿色阈值会漏掉。
    这里单独提取“暗绿冠层候选”，后续只作为补检，避免把阴影大量误圈。
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)

    b, g, r = cv2.split(img_bgr.astype(np.int16))
    exg = 2 * g - r - b

    dark_hsv = (
        (v_ch < 100)
        & (v_ch > 24)
        & (s_ch > 32)
        & (h_ch >= 25)
        & (h_ch <= 115)
    )
    dark_green_bias = (
        (v_ch < 78)
        & (s_ch > 22)
        & (exg > -5)
        & (g >= r - 6)
    )
    mask = (dark_hsv | dark_green_bias).astype(np.uint8) * 255

    # 排除道路等低饱和灰色区域。
    road_like = ((s_ch < 26) & (v_ch > 68)).astype(np.uint8) * 255
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(road_like))

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    return mask


def distance_watershed_labels(mask, peak_distance=25, min_peak_radius=5):
    """
    用距离变换寻找每棵树冠的中心峰值，再通过分水岭把相邻树冠拆开。
    peak_distance <= 0 时启用自适应模式，
    根据距离变换的统计分布自动计算最佳参数。
    使用 h-dome 变换抑制微弱局部极大值，减少假峰。
    """
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    distance = cv2.GaussianBlur(distance, (0, 0), 2.0)

    nonzero_dist = distance[distance > 0]
    if len(nonzero_dist) < 50:
        peak_distance = max(peak_distance, 25)
        min_peak_radius = max(min_peak_radius, 5)

    median_dist = float(np.median(nonzero_dist)) if len(nonzero_dist) > 0 else 10.0

    # --- 自适应参数计算 ---
    if peak_distance <= 0:
        peak_distance = max(13, int(median_dist * 1.35))
    if min_peak_radius <= 0:
        min_peak_radius = max(3, int(median_dist * 0.28))

    # h-dome 抑制：减去 h=1.5 的常量偏移后再找峰值，消除微小波动造成的假峰
    h_dome = 1.5
    distance_ft = distance.astype(np.float32)
    distance_suppressed = np.maximum(distance_ft - h_dome, 0.0)

    peak_size = max(3, int(peak_distance) | 1)
    peak_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (peak_size, peak_size))
    local_max = (
        (distance_suppressed == cv2.dilate(distance_suppressed, peak_kernel))
        & (distance_suppressed >= max(min_peak_radius, 2.0))
    )
    local_max = local_max.astype(np.uint8) * 255

    marker_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    local_max = cv2.dilate(local_max, marker_kernel, iterations=1)

    marker_count, markers = cv2.connectedComponents(local_max)

    if marker_count <= 1:
        return contours_as_labels(mask), distance

    markers = markers.astype(np.int32) + 1
    markers[(mask > 0) & (local_max == 0)] = 0
    markers[mask == 0] = 1

    elevation = cv2.normalize(255 - distance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    elevation[mask == 0] = 255
    cv2.watershed(cv2.cvtColor(elevation, cv2.COLOR_GRAY2BGR), markers)

    return markers, distance


def split_crowns_by_distance(mask, peak_distance=0, min_peak_radius=0):
    labels, _ = distance_watershed_labels(mask, peak_distance, min_peak_radius)
    return labels


def detections_from_labels(
    labels,
    mask,
    min_area,
    max_area,
    circularity_min,
    fill_ratio_min,
    radius_scale,
    min_radius,
    max_radius,
    use_ellipse_fit=False,
):
    detections = []
    label_ids = [label for label in np.unique(labels) if label > 1]

    for label in label_ids:
        label_mask = np.zeros(mask.shape, dtype=np.uint8)
        label_mask[(labels == label) & (mask > 0)] = 255

        contours, _ = cv2.findContours(label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue

        perimeter = cv2.arcLength(c, True)
        if perimeter <= 0:
            continue

        circularity = 4 * math.pi * area / (perimeter * perimeter)
        if circularity < circularity_min:
            continue

        (enclose_x, enclose_y), enclosing_radius = cv2.minEnclosingCircle(c)
        if enclosing_radius <= 0:
            continue

        fill_ratio = area / (math.pi * enclosing_radius * enclosing_radius)
        if fill_ratio < fill_ratio_min:
            continue

        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = max(w, h) / max(1, min(w, h))
        if aspect_ratio > 2.8:
            continue

        # --- 中心点和半径计算 ---
        if use_ellipse_fit and len(c) >= 5:
            # 椭圆拟合 + 外接圆取较大值，确保圆能完整覆盖树冠
            try:
                ellipse = cv2.fitEllipse(c)
                (ex, ey), (ma, MA), angle = ellipse
                # ma, MA 是短轴和长轴的全长，半径取平均半轴
                ellipse_radius = (ma + MA) / 4.0
                cx, cy = ex, ey
                # 取椭圆半径和外接圆半径中的较大值，确保覆盖整个树冠
                base_radius = max(ellipse_radius, enclosing_radius * 0.85)
                # 再乘以放大系数
                radius = int(round(base_radius * radius_scale))
            except cv2.error:
                # fitEllipse 失败时回退到外接圆
                moments = cv2.moments(c)
                if moments["m00"] == 0:
                    cx, cy = enclose_x, enclose_y
                else:
                    cx = moments["m10"] / moments["m00"]
                    cy = moments["m01"] / moments["m00"]
                radius = int(round(enclosing_radius * radius_scale))
        else:
            moments = cv2.moments(c)
            if moments["m00"] == 0:
                cx, cy = enclose_x, enclose_y
            else:
                cx = moments["m10"] / moments["m00"]
                cy = moments["m01"] / moments["m00"]

            # 用外接圆半径 * 放大系数，不再用 min() 导致偏小
            radius = int(round(enclosing_radius * radius_scale))

        if radius < min_radius or radius > max_radius:
            continue

        detections.append({
            "x": int(round(cx)),
            "y": int(round(cy)),
            "radius": radius,
            "area_px": round(float(area), 2),
            "circularity": round(float(circularity), 3),
            "fill_ratio": round(float(fill_ratio), 3),
        })

    return detections


def detections_from_labels_peak_radius(
    labels,
    mask,
    distance,
    min_area=150,
    max_area=7500,
    circularity_min=0.12,
    fill_ratio_min=0.22,
    min_radius=8,
    max_radius=58,
    peak_radius_scale=1.70,
    peak_radius_offset=3.0,
    area_radius_scale=0.95,
):
    """
    中心由分水岭树冠区域决定，半径优先使用距离变换峰值估算。
    降低了半径放大系数，增加面积约束，避免过大的标注圆。
    """
    detections = []
    label_ids = [label for label in np.unique(labels) if label > 1]

    for label in label_ids:
        label_mask = np.zeros(mask.shape, dtype=np.uint8)
        label_mask[(labels == label) & (mask > 0)] = 255

        contours, _ = cv2.findContours(label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue

        perimeter = cv2.arcLength(c, True)
        if perimeter <= 0:
            continue

        circularity = 4 * math.pi * area / (perimeter * perimeter)
        if circularity < circularity_min:
            continue

        (enclose_x, enclose_y), enclosing_radius = cv2.minEnclosingCircle(c)
        if enclosing_radius <= 0:
            continue

        fill_ratio = area / (math.pi * enclosing_radius * enclosing_radius)
        if fill_ratio < fill_ratio_min:
            continue

        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = max(w, h) / max(1, min(w, h))
        if aspect_ratio > 2.7:
            continue

        moments = cv2.moments(c)
        if moments["m00"] == 0:
            cx, cy = enclose_x, enclose_y
        else:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]

        region = (labels == label) & (mask > 0)
        peak_radius = float(distance[region].max()) if np.any(region) else 0.0
        equivalent_radius = math.sqrt(area / math.pi)

        candidate_radius = max(
            peak_radius * peak_radius_scale + peak_radius_offset,
            equivalent_radius * area_radius_scale,
            float(min_radius),
        )

        # 同时用 enclosing_radius 和 candidate_radius 取平衡值
        radius = int(round(min(enclosing_radius * 0.95, candidate_radius)))

        if radius < min_radius or radius > max_radius:
            continue

        detections.append({
            "x": int(round(cx)),
            "y": int(round(cy)),
            "radius": radius,
            "area_px": round(float(area), 2),
            "circularity": round(float(circularity), 3),
            "fill_ratio": round(float(fill_ratio), 3),
            "peak_radius": round(float(peak_radius), 2),
        })

    return detections


def contours_as_labels(mask):
    labels = np.zeros(mask.shape, dtype=np.int32)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for idx, contour in enumerate(contours, 2):
        cv2.drawContours(labels, [contour], -1, idx, -1)

    return labels


def detect_tree_crowns_original(
    img_bgr,
    min_area=80,
    max_area=9000,
    green_s=35,
    green_v=45,
    circularity_min=0.08,
):
    mask = build_tree_mask_original(img_bgr, green_s=green_s, green_v=green_v)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []

    h, w = img_bgr.shape[:2]
    img_area = h * w

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue

        perimeter = cv2.arcLength(c, True)
        if perimeter <= 0:
            continue

        circularity = 4 * math.pi * area / (perimeter * perimeter)
        if circularity < circularity_min:
            continue

        (x, y), radius = cv2.minEnclosingCircle(c)
        x, y, radius = int(x), int(y), int(radius)

        if radius < 4 or radius > min(h, w) * 0.08:
            continue

        if area > img_area * 0.02:
            continue

        detections.append({
            "x": x,
            "y": y,
            "radius": radius,
            "area_px": round(float(area), 2),
            "circularity": round(float(circularity), 3),
        })

    detections.sort(key=lambda d: (d["y"], d["x"]))
    for idx, det in enumerate(detections, 1):
        det["id"] = idx

    return detections, mask


def circle_overlap_area(r1, r2, distance):
    if distance >= r1 + r2:
        return 0.0

    if distance <= abs(r1 - r2):
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


def dedupe_overlapping_detections(detections, merge_overlap=0.78):
    if merge_overlap <= 0:
        return detections

    kept = []
    ranked = sorted(detections, key=lambda d: (d["area_px"], d["radius"]), reverse=True)

    for det in ranked:
        duplicate = False
        for existing in kept:
            dx = det["x"] - existing["x"]
            dy = det["y"] - existing["y"]
            distance = math.hypot(dx, dy)
            overlap = circle_overlap_area(det["radius"], existing["radius"], distance)
            smaller_area = math.pi * min(det["radius"], existing["radius"]) ** 2

            if smaller_area > 0 and overlap / smaller_area >= merge_overlap:
                duplicate = True
                break

        if not duplicate:
            kept.append(det)

    return kept


def tag_detections(detections, source):
    tagged = []
    for det in detections:
        det_copy = dict(det)
        det_copy["source"] = source
        tagged.append(det_copy)
    return tagged


def renumber_detections(detections):
    detections.sort(key=lambda d: (d["y"], d["x"]))
    for idx, det in enumerate(detections, 1):
        det["id"] = idx
    return detections


def _candidate_quality(det):
    source_rank = {
        "reference": 10,
        "recall": 5,
        "lab": 7,
        "combined": 6,
        "dark": 4,
        "residual": 3,
        "peak": 2,
    }.get(det.get("source", ""), 1)

    circularity = float(det.get("circularity", 0.0) or 0.0)
    fill_ratio = float(det.get("fill_ratio", 0.0) or 0.0)
    green_ratio = float(det.get("green_ratio", 0.0) or 0.0)
    radius = float(det.get("radius", 0.0) or 0.0)
    area = float(det.get("area_px", 0.0) or 0.0)

    radius_penalty = max(0.0, radius - 58.0) / 12.0
    area_bonus = min(area, 2500.0) / 2500.0

    return (
        source_rank * 100.0
        + circularity * 14.0
        + fill_ratio * 18.0
        + green_ratio * 16.0
        + area_bonus * 5.0
        - radius_penalty
    )


def _is_duplicate_candidate(det, existing, merge_overlap=0.62):
    dx = det["x"] - existing["x"]
    dy = det["y"] - existing["y"]
    distance = math.hypot(dx, dy)

    r1 = float(det["radius"])
    r2 = float(existing["radius"])
    min_radius = min(r1, r2)
    if min_radius <= 0:
        return False

    overlap = circle_overlap_area(r1, r2, distance)
    smaller_area = math.pi * min_radius * min_radius
    overlap_ratio_smaller = overlap / smaller_area if smaller_area > 0 else 0.0

    # 也检测较大圆被覆盖的程度，防止大圆几乎包含小圆时未被去重
    max_radius = max(r1, r2)
    larger_area = math.pi * max_radius * max_radius
    overlap_ratio_larger = overlap / larger_area if larger_area > 0 else 0.0

    same_center = distance <= max(3.0, min_radius * 0.25)
    center_close = distance <= max(5.0, min_radius * 0.50)
    almost_contained = (
        overlap_ratio_smaller >= merge_overlap
        or overlap_ratio_larger >= merge_overlap * 0.70
    ) and distance <= max(10.0, min_radius * 0.85)
    medium_overlap = (
        overlap_ratio_smaller >= merge_overlap * 0.65
        or overlap_ratio_larger >= merge_overlap * 0.50
    ) and distance <= max(6.0, min_radius * 0.55)

    return same_center or (center_close and overlap_ratio_smaller >= 0.38) or almost_contained or medium_overlap


def dedupe_candidate_pool(detections, merge_overlap=0.74):
    if not detections:
        return []

    kept = []
    ranked = sorted(detections, key=_candidate_quality, reverse=True)

    for det in ranked:
        if any(_is_duplicate_candidate(det, existing, merge_overlap) for existing in kept):
            continue
        kept.append(det)

    return kept


def _is_dense_duplicate_candidate(
    det,
    existing,
    center_factor=1.80,
    min_overlap=0.10,
    min_radius_ratio=0.35,
):
    dx = det["x"] - existing["x"]
    dy = det["y"] - existing["y"]
    distance = math.hypot(dx, dy)

    r1 = float(det["radius"])
    r2 = float(existing["radius"])
    min_radius = min(r1, r2)
    max_radius = max(r1, r2)
    if min_radius <= 0 or max_radius <= 0:
        return False

    radius_ratio = min_radius / max_radius
    overlap = circle_overlap_area(r1, r2, distance)
    smaller_area = math.pi * min_radius * min_radius
    overlap_ratio_smaller = overlap / smaller_area if smaller_area > 0 else 0.0

    # 密集区重复标注常表现为：中心落在同一树冠核心附近，但圆被局部收缩后
    # IoU 不够高。这里用中心距离作为主约束，再要求半径不能差得过离谱。
    very_close_center = distance <= max(7.0, min_radius * 0.82)
    dense_close_center = distance <= max(10.0, min_radius * center_factor)
    almost_contained = (
        distance + min_radius <= max_radius * 1.08
        and overlap_ratio_smaller >= max(min_overlap, 0.35)
    )

    return (
        radius_ratio >= min_radius_ratio
        and (
            very_close_center
            or almost_contained
            or (dense_close_center and overlap_ratio_smaller >= min_overlap)
        )
    )


def dedupe_dense_detections(
    detections,
    center_factor=1.80,
    min_overlap=0.10,
    min_radius_ratio=0.35,
):
    if not detections:
        return [], []

    kept = []
    removed = []
    ranked = sorted(detections, key=_candidate_quality, reverse=True)

    for det in ranked:
        duplicate_of = None
        for existing in kept:
            if _is_dense_duplicate_candidate(
                det,
                existing,
                center_factor=center_factor,
                min_overlap=min_overlap,
                min_radius_ratio=min_radius_ratio,
            ):
                duplicate_of = existing
                break

        if duplicate_of is None:
            kept.append(det)
            continue

        rejected = dict(det)
        rejected["local_refine_decision"] = "rejected_dense_duplicate"
        rejected["duplicate_of_x"] = duplicate_of.get("x")
        rejected["duplicate_of_y"] = duplicate_of.get("y")
        rejected["duplicate_of_radius"] = duplicate_of.get("radius")
        removed.append(rejected)

    return renumber_detections(kept), removed


def detections_from_mask_peak_radius(
    mask,
    source,
    min_area=120,
    max_area=6800,
    circularity_min=0.10,
    fill_ratio_min=0.18,
    min_radius=6,
    max_radius=58,
    peak_distance=0,
    min_peak_radius=0,
    peak_radius_scale=1.70,
    peak_radius_offset=3.0,
    area_radius_scale=0.95,
):
    labels, distance = distance_watershed_labels(mask, peak_distance, min_peak_radius)
    detections = detections_from_labels_peak_radius(
        labels=labels,
        mask=mask,
        distance=distance,
        min_area=min_area,
        max_area=max_area,
        circularity_min=circularity_min,
        fill_ratio_min=fill_ratio_min,
        min_radius=min_radius,
        max_radius=max_radius,
        peak_radius_scale=peak_radius_scale,
        peak_radius_offset=peak_radius_offset,
        area_radius_scale=area_radius_scale,
    )
    return tag_detections(detections, source)


def detection_coverage_mask(shape, detections, coverage_scale=0.74):
    coverage = np.zeros(shape, dtype=np.uint8)
    for det in detections:
        radius = max(2, int(round(det["radius"] * coverage_scale)))
        cv2.circle(coverage, (det["x"], det["y"]), radius, 255, -1)
    return coverage


def residual_supplement_detections(mask, existing_detections):
    coverage = detection_coverage_mask(mask.shape, existing_detections, coverage_scale=1.0)
    residual = cv2.bitwise_and(mask, cv2.bitwise_not(coverage))

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    residual = cv2.morphologyEx(residual, cv2.MORPH_OPEN, kernel_open, iterations=1)
    residual = cv2.morphologyEx(residual, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    return detections_from_mask_peak_radius(
        residual,
        source="residual",
        min_area=300,
        max_area=4500,
        circularity_min=0.14,
        fill_ratio_min=0.28,
        min_radius=8,
        max_radius=52,
        peak_distance=20,
        min_peak_radius=0,
        peak_radius_scale=1.70,
        peak_radius_offset=3.0,
    )


def _has_nearby_detection(candidate, detections, factor=0.72):
    for det in detections:
        distance = math.hypot(candidate["x"] - det["x"], candidate["y"] - det["y"])
        threshold = max(6.0, min(candidate["radius"], det["radius"]) * factor)
        if distance <= threshold:
            return True
    return False


def peak_supplement_detections(mask, existing_detections):
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    distance = cv2.GaussianBlur(distance, (0, 0), 2.0)

    # 使用 19x19 的核对峰值做局部极大值检测（减小了窗口）
    peak_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
    local_max = (distance == cv2.dilate(distance, peak_kernel)) & (distance >= 8.0)
    local_max = local_max.astype(np.uint8) * 255

    num_labels, labels_map, _, _ = cv2.connectedComponentsWithStats(local_max, connectivity=8)
    detections = []

    for label in range(1, num_labels):
        ys, xs = np.where(labels_map == label)
        if len(xs) == 0:
            continue

        peak_values = distance[ys, xs]
        best_idx = int(np.argmax(peak_values))
        cx = int(xs[best_idx])
        cy = int(ys[best_idx])
        peak_radius = float(peak_values[best_idx])

        radius = int(round(max(6.0, min(50.0, peak_radius * 1.70 + 3.0))))
        circle_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.circle(circle_mask, (cx, cy), radius, 255, -1)

        tree_px = float(np.count_nonzero(cv2.bitwise_and(mask, circle_mask)))
        circle_area = math.pi * radius * radius
        fill_ratio = tree_px / circle_area if circle_area > 0 else 0.0

        if tree_px < 300 or fill_ratio < 0.28:
            continue

        candidate = {
            "x": cx,
            "y": cy,
            "radius": radius,
            "area_px": round(tree_px, 2),
            "circularity": 0.0,
            "fill_ratio": round(float(fill_ratio), 3),
            "peak_radius": round(float(peak_radius), 2),
            "source": "peak",
        }

        if _has_nearby_detection(candidate, existing_detections + detections, factor=0.90):
            continue

        detections.append(candidate)

    return detections


def modern_detection_params_from_preset(args):
    if args.preset == "precision":
        return {
            "min_area": 100,
            "max_area": 8000,
            "green_s": 22,
            "green_v": 38,
            "exg_min": 10,
            "circularity_min": 0.10,
            "fill_ratio_min": 0.22,
            "radius_scale": 1.15,
            "min_radius": 6,
            "max_radius": 62,
            "split_crowns": not args.no_split,
            "peak_distance": 0,
            "min_peak_radius": 0,
            "merge_overlap": 0.70,
            "use_lab": True,
            "shadow_filter": True,
            "use_ellipse_fit": True,
        }

    if args.preset == "balanced":
        return {
            "min_area": 150,
            "max_area": 6000,
            "green_s": 38,
            "green_v": 42,
            "exg_min": 18,
            "circularity_min": 0.15,
            "fill_ratio_min": 0.30,
            "radius_scale": 1.15,
            "min_radius": 8,
            "max_radius": 56,
            "split_crowns": not args.no_split,
            "peak_distance": 0,
            "min_peak_radius": 0,
            "merge_overlap": 0.74,
            "use_lab": False,
            "shadow_filter": False,
            "use_ellipse_fit": False,
        }

    if args.preset == "recall":
        return {
            "min_area": 90,
            "max_area": 6800,
            "green_s": 28,
            "green_v": 38,
            "exg_min": 10,
            "circularity_min": 0.10,
            "fill_ratio_min": 0.22,
            "radius_scale": 1.15,
            "min_radius": 6,
            "max_radius": 62,
            "split_crowns": not args.no_split,
            "peak_distance": 0,
            "min_peak_radius": 0,
            "merge_overlap": 0.76,
            "use_lab": False,
            "shadow_filter": False,
            "use_ellipse_fit": False,
        }

    return {
        "min_area": args.min_area,
        "max_area": args.max_area,
        "green_s": args.green_s,
        "green_v": args.green_v,
        "exg_min": args.exg_min,
        "circularity_min": args.circularity_min,
        "fill_ratio_min": args.fill_ratio_min,
        "radius_scale": args.radius_scale,
        "min_radius": args.min_radius,
        "max_radius": args.max_radius,
        "split_crowns": not args.no_split,
        "peak_distance": args.peak_distance,
        "min_peak_radius": args.min_peak_radius,
        "merge_overlap": args.merge_overlap,
        "use_lab": getattr(args, 'use_lab', False),
        "shadow_filter": getattr(args, 'shadow_filter', False),
        "use_ellipse_fit": getattr(args, 'use_ellipse_fit', False),
    }


def detect_tree_crowns_reference(
    img_bgr,
    min_area=150,
    max_area=7500,
    green_s=30,
    green_v=40,
    exg_min=14,
    circularity_min=0.12,
    fill_ratio_min=0.22,
    min_radius=8,
    max_radius=58,
    peak_distance=16,
    min_peak_radius=0,
    merge_overlap=0.72,
):
    """
    树冠检测：用树冠中心峰值估算半径，更精准的分水岭分裂和更紧的过滤。
    peak_distance 降低为 16 以分离密集树冠。
    min_peak_radius=0 启用自适应。
    """
    mask = build_tree_mask(img_bgr, green_s=green_s, green_v=green_v, exg_min=exg_min)
    labels, distance = distance_watershed_labels(mask, peak_distance, min_peak_radius)

    detections = detections_from_labels_peak_radius(
        labels=labels,
        mask=mask,
        distance=distance,
        min_area=min_area,
        max_area=max_area,
        circularity_min=circularity_min,
        fill_ratio_min=fill_ratio_min,
        min_radius=min_radius,
        max_radius=max_radius,
    )
    detections = dedupe_overlapping_detections(detections, merge_overlap)

    detections.sort(key=lambda d: (d["y"], d["x"]))
    for idx, det in enumerate(detections, 1):
        det["id"] = idx

    return detections, mask


def detect_tree_crowns_ensemble(
    img_bgr,
    use_gap_fill=True,
    use_peak_supplement=True,
    merge_overlap=0.60,
):
    """
    多策略融合：reference 提供高质量基准，recall 用稍宽松阈值补漏但不过度激进，
    再经温和去重。
    """
    reference_detections, reference_mask = detect_tree_crowns_reference(img_bgr)
    reference_detections = tag_detections(reference_detections, "reference")

    # recall 策略：仅适度放宽，不再用极端低阈值
    recall_detections, recall_mask = detect_tree_crowns_reference(
        img_bgr,
        min_area=120,
        max_area=6800,
        green_s=26,
        green_v=36,
        exg_min=10,
        circularity_min=0.08,
        fill_ratio_min=0.16,
        min_radius=6,
        max_radius=62,
        peak_distance=12,
        min_peak_radius=0,
        merge_overlap=0.70,
    )
    recall_detections = tag_detections(recall_detections, "recall")

    combined_mask = cv2.bitwise_or(reference_mask, recall_mask)

    candidate_pool = reference_detections + recall_detections
    candidate_pool = dedupe_candidate_pool(candidate_pool, merge_overlap=merge_overlap)

    if use_gap_fill:
        residual_detections = residual_supplement_detections(combined_mask, candidate_pool)
        candidate_pool = dedupe_candidate_pool(
            candidate_pool + residual_detections,
            merge_overlap=merge_overlap,
        )

    if use_peak_supplement:
        peak_detections = peak_supplement_detections(combined_mask, candidate_pool)
        candidate_pool = dedupe_candidate_pool(
            candidate_pool + peak_detections,
            merge_overlap=merge_overlap,
        )

    return renumber_detections(candidate_pool), combined_mask


def detect_with_preset(img_bgr, args):
    if args.preset == "original":
        return detect_tree_crowns_original(
            img_bgr,
            min_area=args.min_area,
            max_area=args.max_area,
            green_s=args.green_s,
            green_v=args.green_v,
            circularity_min=args.circularity_min,
        )

    if args.preset == "reference":
        return detect_tree_crowns_reference(img_bgr)

    if args.preset == "ensemble":
        return detect_tree_crowns_ensemble(
            img_bgr,
            use_gap_fill=not getattr(args, "no_gap_fill", False),
            use_peak_supplement=not getattr(args, "no_peak_supplement", False),
            merge_overlap=getattr(args, "ensemble_merge_overlap", 0.60),
        )

    return detect_tree_crowns(img_bgr, **modern_detection_params_from_preset(args))


def detect_tree_crowns(
    img_bgr,
    min_area=120,
    max_area=6800,
    green_s=35,
    green_v=45,
    exg_min=12,
    circularity_min=0.12,
    fill_ratio_min=0.28,
    radius_scale=1.15,
    min_radius=8,
    max_radius=58,
    split_crowns=True,
    peak_distance=0,
    min_peak_radius=0,
    merge_overlap=0.74,
    use_lab=False,
    shadow_filter=False,
    use_ellipse_fit=False,
):
    # 根据参数选择掩膜生成方式
    if use_lab or shadow_filter:
        mask = build_tree_mask_v2(
            img_bgr, green_s=green_s, green_v=green_v, exg_min=exg_min,
            use_lab=use_lab, shadow_filter=shadow_filter, min_blob=min_area // 2,
        )
    else:
        mask = build_tree_mask(img_bgr, green_s=green_s, green_v=green_v, exg_min=exg_min)

    labels = split_crowns_by_distance(mask, peak_distance, min_peak_radius) if split_crowns else contours_as_labels(mask)

    detections = detections_from_labels(
        labels=labels,
        mask=mask,
        min_area=min_area,
        max_area=max_area,
        circularity_min=circularity_min,
        fill_ratio_min=fill_ratio_min,
        radius_scale=radius_scale,
        min_radius=min_radius,
        max_radius=max_radius,
        use_ellipse_fit=use_ellipse_fit,
    )
    detections = dedupe_overlapping_detections(detections, merge_overlap)

    # 按 y、x 排序，方便查看
    detections.sort(key=lambda d: (d["y"], d["x"]))
    for idx, det in enumerate(detections, 1):
        det["id"] = idx

    return detections, mask


def _simplify_contour_points(contour, max_points=96):
    perimeter = cv2.arcLength(contour, True)
    epsilon = max(0.8, perimeter * 0.006)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    points = approx.reshape(-1, 2)

    if len(points) > max_points:
        keep = np.linspace(0, len(points) - 1, max_points, dtype=int)
        points = points[keep]

    return [[int(x), int(y)] for x, y in points]


def _crown_contour_for_detection(mask, det, padding=1.22):
    h, w = mask.shape[:2]
    cx = int(round(det["x"]))
    cy = int(round(det["y"]))
    radius = max(4, int(round(det.get("radius", 8) * padding)))

    x1 = max(0, cx - radius)
    y1 = max(0, cy - radius)
    x2 = min(w, cx + radius + 1)
    y2 = min(h, cy + radius + 1)
    if x2 <= x1 or y2 <= y1:
        return None

    roi = mask[y1:y2, x1:x2]
    if np.count_nonzero(roi) == 0:
        return None

    gate = np.zeros(roi.shape, dtype=np.uint8)
    local_center = (cx - x1, cy - y1)
    cv2.circle(gate, local_center, radius, 255, -1)
    local = cv2.bitwise_and(roi, gate)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    local = cv2.morphologyEx(local, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    def contour_score(contour):
        area = cv2.contourArea(contour)
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            mx, my = local_center
        else:
            mx = moments["m10"] / moments["m00"]
            my = moments["m01"] / moments["m00"]
        center_distance = math.hypot(mx - local_center[0], my - local_center[1])
        contains_center = cv2.pointPolygonTest(contour, local_center, False) >= 0
        return (1 if contains_center else 0, area / (1.0 + center_distance), area)

    best = max(contours, key=contour_score)
    if cv2.contourArea(best) < 6:
        return None

    best = best.copy()
    best[:, :, 0] += x1
    best[:, :, 1] += y1
    return best


def _label_near_detection(labels, cx, cy, search_radius=7):
    h, w = labels.shape[:2]
    if 0 <= cy < h and 0 <= cx < w and labels[cy, cx] > 1:
        return int(labels[cy, cx])

    x1 = max(0, cx - search_radius)
    y1 = max(0, cy - search_radius)
    x2 = min(w, cx + search_radius + 1)
    y2 = min(h, cy + search_radius + 1)
    window = labels[y1:y2, x1:x2]
    candidates = window[window > 1]
    if candidates.size == 0:
        return -1

    values, counts = np.unique(candidates, return_counts=True)
    return int(values[np.argmax(counts)])


def _label_contour_for_detection(mask, labels, det, padding=1.08):
    h, w = mask.shape[:2]
    cx = int(round(det["x"]))
    cy = int(round(det["y"]))
    label_id = _label_near_detection(labels, cx, cy)
    if label_id <= 1:
        return None

    radius = max(4, int(round(det.get("radius", 8) * padding)))
    x1 = max(0, cx - radius)
    y1 = max(0, cy - radius)
    x2 = min(w, cx + radius + 1)
    y2 = min(h, cy + radius + 1)
    if x2 <= x1 or y2 <= y1:
        return None

    label_roi = ((labels[y1:y2, x1:x2] == label_id) & (mask[y1:y2, x1:x2] > 0)).astype(np.uint8) * 255
    if np.count_nonzero(label_roi) == 0:
        return None

    gate = np.zeros(label_roi.shape, dtype=np.uint8)
    cv2.circle(gate, (cx - x1, cy - y1), radius, 255, -1)
    local = cv2.bitwise_and(label_roi, gate)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    local = cv2.morphologyEx(local, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best = max(contours, key=cv2.contourArea)
    if cv2.contourArea(best) < 6:
        return None

    best = cv2.convexHull(best)
    best[:, :, 0] += x1
    best[:, :, 1] += y1
    return best


def attach_detection_contours(mask, detections):
    """
    给每个候选附加树冠轮廓。轮廓来自植被掩膜，并被限制在候选半径附近，
    比直接画圆更贴近树冠边缘，也避免连通的大树丛把边界拉得过大。
    """
    labels = split_crowns_by_distance(mask, peak_distance=0, min_peak_radius=0)
    for det in detections:
        contour = _label_contour_for_detection(mask, labels, det)
        if contour is None:
            contour = _crown_contour_for_detection(mask, det)
        if contour is None:
            continue
        points = _simplify_contour_points(contour)
        if len(points) >= 3:
            det["_contour_points"] = points
            det["contour_points"] = len(points)
    return detections


def scale_detection_geometry(detections, scale_x, scale_y):
    radius_scale = (scale_x + scale_y) / 2.0

    for det in detections:
        for key in ("x", "x1", "x2"):
            if key in det:
                det[key] = int(round(det[key] * scale_x))
        for key in ("y", "y1", "y2"):
            if key in det:
                det[key] = int(round(det[key] * scale_y))
        for key in ("radius", "peak_radius"):
            if key in det:
                det[key] = int(round(det[key] * radius_scale))
        if "area_px" in det:
            det["area_px"] = round(float(det["area_px"]) * scale_x * scale_y, 2)
        if "_contour_points" in det:
            det["_contour_points"] = [
                [int(round(x * scale_x)), int(round(y * scale_y))]
                for x, y in det["_contour_points"]
            ]

    return detections


def serialize_detection_contours(detections):
    for det in detections:
        points = det.pop("_contour_points", None)
        if points:
            det["contour"] = json.dumps(points, separators=(",", ":"))
            det["contour_points"] = len(points)
    return detections


def contour_points_from_detection(det):
    points = det.get("_contour_points") or det.get("contour")
    if not points:
        return None
    if isinstance(points, str):
        try:
            points = json.loads(points)
        except json.JSONDecodeError:
            return None
    if len(points) < 3:
        return None
    return np.array(points, dtype=np.int32).reshape((-1, 1, 2))


def resize_for_detection(img_bgr, work_max_dim):
    if not work_max_dim or work_max_dim <= 0:
        return img_bgr, 1.0

    h, w = img_bgr.shape[:2]
    max_side = max(h, w)
    if max_side <= work_max_dim:
        return img_bgr, 1.0

    scale = work_max_dim / float(max_side)
    work_w = max(1, int(round(w * scale)))
    work_h = max(1, int(round(h * scale)))
    work_img = cv2.resize(img_bgr, (work_w, work_h), interpolation=cv2.INTER_AREA)
    return work_img, scale


def detect_for_output(img_bgr, args):
    work_max_dim = getattr(args, "work_max_dim", 2560)
    work_img, work_scale = resize_for_detection(img_bgr, work_max_dim)

    detections, work_mask = detect_with_preset(work_img, args)
    attach_detection_contours(work_mask, detections)

    if work_scale != 1.0:
        inv_scale = 1.0 / work_scale
        scale_detection_geometry(detections, inv_scale, inv_scale)
        h, w = img_bgr.shape[:2]
        mask = cv2.resize(work_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        mask = work_mask

    serialize_detection_contours(detections)
    return detections, mask, work_scale


def build_crown_core_mask(img_bgr):
    """
    提取更保守的树冠核心区域，用于过滤裸地/道路/阴影上的坏候选。
    提高了阈值以区分树冠（暗绿、高饱和、有纹理）和草地/下层植被。
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)

    b_ch, g_ch, r_ch = cv2.split(img_bgr.astype(np.int16))
    exg = 2 * g_ch - r_ch - b_ch

    # 计算局部纹理方差：树冠有树叶纹理，比草地纹理丰富
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    local_var = cv2.GaussianBlur(gray.astype(np.float32), (3, 3), 0)
    local_var_sq = cv2.GaussianBlur(gray.astype(np.float32) ** 2, (3, 3), 0)
    texture_var = np.maximum(local_var_sq - local_var ** 2, 0)

    green_hue = (h_ch >= 26) & (h_ch <= 110)
    # 树冠核心：较高饱和度和亮度，绿色偏向明显
    crown_core = (
        green_hue
        & (s_ch >= 48)
        & (v_ch >= 52)
        & (exg >= 10)
        & (g_ch >= r_ch + 6)
    )
    # 较暗树冠核心：稍低亮度但饱和度仍较高
    dark_crown = (
        green_hue
        & (s_ch >= 40)
        & (v_ch >= 38)
        & (v_ch <= 75)
        & (exg >= 6)
        & (g_ch >= r_ch)
    )

    # 道路/裸土通常低饱和且偏亮
    road_like = (s_ch < 22) & (v_ch > 68)
    pure_shadow = (v_ch < 30) & (exg < 6)

    mask = ((crown_core | dark_crown) & (~road_like) & (~pure_shadow)).astype(np.uint8) * 255

    # 只保留有纹理（方差>阈值）的区域，过滤平坦草地
    texture_threshold = 30.0
    mask = cv2.bitwise_and(mask, (texture_var > texture_threshold).astype(np.uint8) * 255)

    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=1)
    return mask


def refine_detections_by_crown_core(
    img_bgr,
    detections,
    min_green_ratio=0.28,
    min_component_area=30,
    max_radius=120,
):
    """
    根据局部绿色树冠核心过滤误检，并收缩明显过大的圆。
    提高了 min_green_ratio（不含树冠的圆过滤）和 max_radius（不再过度截断）。
    """
    if not detections:
        return detections, []

    core_mask = build_crown_core_mask(img_bgr)
    h, w = core_mask.shape[:2]
    refined = []
    rejected = []

    for det in detections:
        cx = int(round(det["x"]))
        cy = int(round(det["y"]))
        radius = int(round(det["radius"]))
        if radius <= 0:
            continue

        pad = max(8, int(round(radius * 1.25)))
        x1 = max(0, cx - pad)
        y1 = max(0, cy - pad)
        x2 = min(w, cx + pad + 1)
        y2 = min(h, cy + pad + 1)
        if x2 <= x1 or y2 <= y1:
            continue

        local_mask = core_mask[y1:y2, x1:x2]
        gate = np.zeros(local_mask.shape, dtype=np.uint8)
        local_center = (cx - x1, cy - y1)
        cv2.circle(gate, local_center, max(3, radius), 255, -1)
        gated = cv2.bitwise_and(local_mask, gate)

        circle_area = max(1, int(np.count_nonzero(gate)))
        green_ratio = float(np.count_nonzero(gated)) / float(circle_area)
        if green_ratio < min_green_ratio:
            bad = dict(det)
            bad["local_refine_decision"] = "rejected_low_green_core"
            bad["green_ratio"] = round(green_ratio, 4)
            rejected.append(bad)
            continue

        num_labels, labels_map, stats, centroids = cv2.connectedComponentsWithStats(gated, connectivity=8)
        best_label = -1
        best_score = -1.0
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < min_component_area:
                continue
            px, py = centroids[label]
            distance = math.hypot(px - local_center[0], py - local_center[1])
            if distance > max(10.0, radius * 0.90):
                continue
            score = area / (1.0 + distance * 0.18)
            if score > best_score:
                best_score = score
                best_label = label

        if best_label < 0:
            bad = dict(det)
            bad["local_refine_decision"] = "rejected_no_green_component"
            bad["green_ratio"] = round(green_ratio, 4)
            rejected.append(bad)
            continue

        component = (labels_map == best_label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        area = max(1.0, float(cv2.contourArea(contour)))
        (enc_x, enc_y), enc_radius = cv2.minEnclosingCircle(contour)
        equivalent_radius = math.sqrt(area / math.pi)

        new_cx = int(round(x1 + enc_x))
        new_cy = int(round(y1 + enc_y))
        estimated_radius = max(equivalent_radius * 1.45, enc_radius * 0.85, 5.0)

        # 只在旧圆明显偏大时收缩，避免把稀疏树冠裁得过小
        if radius > estimated_radius * 1.35:
            new_radius = int(round(min(radius, estimated_radius)))
        else:
            new_radius = radius
        new_radius = int(max(5, min(max_radius, new_radius)))

        updated = dict(det)
        updated["x"] = max(0, min(w - 1, new_cx))
        updated["y"] = max(0, min(h - 1, new_cy))
        updated["radius"] = new_radius
        updated["green_ratio"] = round(green_ratio, 4)
        updated["local_refine_decision"] = "kept_refined"
        refined.append(updated)

    refined = dedupe_candidate_pool(refined, merge_overlap=0.58)
    refined = renumber_detections(refined)
    return refined, rejected


def draw_detections(
    img_bgr,
    detections,
    show_id=False,
    line_color=(0, 255, 0),
    line_thickness=3,
    center_color=(0, 200, 0),
    center_radius=3,
):
    out = img_bgr.copy()
    for det in detections:
        center = (det["x"], det["y"])
        radius = det["radius"]
        cv2.circle(out, center, radius, line_color, line_thickness)
        if center_color is not None and center_radius > 0:
            cv2.circle(out, center, center_radius, center_color, -1)
        if show_id:
            cv2.putText(
                out,
                str(det["id"]),
                (det["x"] + 3, det["y"] - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
    return out


def draw_style_for_preset(preset):
    if preset in {"reference", "ensemble"}:
        return {
            "line_color": (80, 210, 80),
            "line_thickness": 2,
            "center_color": None,
            "center_radius": 0,
        }

    return {}


def draw_detections_contour(img_bgr, mask, labels, detections, show_id=False):
    """
    用实际分割轮廓替代圆圈进行可视化，更精确展示树冠边界。
    """
    out = img_bgr.copy()

    for det in detections:
        cx, cy = det["x"], det["y"]
        contour = contour_points_from_detection(det)
        if contour is not None:
            cv2.drawContours(out, [contour], -1, (0, 255, 255), 2)
        else:
            # 旧结果没有 contour 字段时，尝试从 labels 找对应区域。
            if labels is not None and 0 <= cy < labels.shape[0] and 0 <= cx < labels.shape[1]:
                label_id = labels[cy, cx]
            else:
                label_id = -1

            if label_id > 1:
                label_mask = np.zeros(mask.shape, dtype=np.uint8)
                label_mask[(labels == label_id) & (mask > 0)] = 255
                contours, _ = cv2.findContours(label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    cv2.drawContours(out, contours, -1, (0, 255, 255), 2)
                else:
                    cv2.circle(out, (cx, cy), det["radius"], (0, 255, 255), 2)
            else:
                cv2.circle(out, (cx, cy), det["radius"], (0, 255, 255), 2)

        cv2.circle(out, (cx, cy), 2, (0, 0, 255), -1)

        if show_id:
            cv2.putText(
                out,
                str(det["id"]),
                (cx + 3, cy - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
    return out


def process_image(source: Path, out_dir: Path, args):
    img = cv2.imread(str(source))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {source}")

    detections, mask, work_scale = detect_for_output(img, args)

    print(f"OpenCV 初筛完成：{source.name}")
    if work_scale != 1.0:
        print(f"工作分辨率缩放：{work_scale:.3f}（结果已映射回原图坐标）")
    print(f"候选树冠数量：{len(detections)}")

    local_rejected = []
    if not getattr(args, "no_local_refine", False):
        before_refine = len(detections)
        detections, local_rejected = refine_detections_by_crown_core(
            img,
            detections,
            min_green_ratio=getattr(args, "local_refine_min_green", 0.28),
            max_radius=getattr(args, "local_refine_max_radius", 120),
        )
        print(f"本地树冠核心过滤：{before_refine} -> {len(detections)}，过滤 {len(local_rejected)} 个")

    if not getattr(args, "no_dense_dedupe", False):
        before_dedupe = len(detections)
        detections, dense_rejected = dedupe_dense_detections(
            detections,
            center_factor=getattr(args, "dense_dedupe_factor", 1.80),
            min_overlap=getattr(args, "dense_dedupe_overlap", 0.10),
        )
        local_rejected.extend(dense_rejected)
        if dense_rejected:
            print(f"密集区重复去重：{before_dedupe} -> {len(detections)}，合并 {len(dense_rejected)} 个")

    # --- 大模型双重验证 ---
    llm_verify = getattr(args, 'llm_verify', False) or getattr(args, 'llm_repair', False)
    rejected = list(local_rejected)

    if llm_verify:
        try:
            from llm_verifier import (
                draw_verified_detections,
                repair_detections_with_tiles,
                verify_detections,
            )

            llm_model = getattr(args, 'llm_model', 'gpt-4o')
            llm_batch = getattr(args, 'llm_batch_size', 6)
            show_species = getattr(args, 'show_species', True)
            show_rejected = getattr(args, 'show_rejected', False)
            llm_min_confidence = getattr(args, 'llm_min_confidence', 30)
            llm_reject_confidence = getattr(args, 'llm_reject_confidence', 85)
            llm_conservative = not getattr(args, 'llm_strict', False)
            llm_max_workers = getattr(args, 'llm_max_workers', 1)

            if getattr(args, 'llm_repair', False):
                verified, llm_rejected = repair_detections_with_tiles(
                    img_bgr=img,
                    detections=detections,
                    model=llm_model,
                    tile_size=getattr(args, 'llm_tile_size', 1024),
                    overlap=getattr(args, 'llm_tile_overlap', 0.25),
                    min_confidence=llm_min_confidence,
                    conservative=llm_conservative,
                    reject_confidence=llm_reject_confidence,
                    min_radius=getattr(args, 'llm_min_radius', 5),
                    max_radius=getattr(args, 'llm_max_radius', 80),
                    dedupe_overlap=getattr(args, 'llm_dedupe_overlap', 0.58),
                    strict_candidates=getattr(args, 'llm_repair_strict_candidates', False),
                    max_workers=llm_max_workers,
                    verbose=True,
                )
                rejected.extend(llm_rejected)
            else:
                verified, llm_rejected = verify_detections(
                    img_bgr=img,
                    detections=detections,
                    batch_size=llm_batch,
                    model=llm_model,
                    min_confidence=llm_min_confidence,
                    conservative=llm_conservative,
                    reject_confidence=llm_reject_confidence,
                    max_workers=llm_max_workers,
                    verbose=True,
                )
                rejected.extend(llm_rejected)

            # 使用大模型验证后的可视化
            vis = draw_verified_detections(
                img, verified,
                rejected=rejected if show_rejected else None,
                show_species=show_species,
                show_id=args.show_id,
            )
            detections = verified

        except ImportError as e:
            print(f"警告：大模型验证模块加载失败: {e}")
            print("回退到纯 OpenCV 检测结果")
            llm_verify = False
        except ValueError as e:
            print(f"警告：{e}")
            print("回退到纯 OpenCV 检测结果")
            llm_verify = False
        except Exception as e:
            print(f"警告：大模型验证异常: {e}")
            print("回退到纯 OpenCV 检测结果")
            llm_verify = False

    if not llm_verify:
        # 根据 --draw-contour 选项选择可视化方式
        draw_contour = getattr(args, 'draw_contour', False)
        if draw_contour:
            vis = draw_detections_contour(img, mask, None, detections, show_id=args.show_id)
        else:
            vis = draw_detections(img, detections, show_id=args.show_id, **draw_style_for_preset(args.preset))

    stem = source.stem
    suffix = "_llm_verified" if llm_verify else "_detected"
    img_out = out_dir / f"{stem}{suffix}.jpg"
    mask_out = out_dir / f"{stem}_mask.jpg"
    excel_out = out_dir / f"{stem}_trees.xlsx"

    cv2.imwrite(str(img_out), vis)
    cv2.imwrite(str(mask_out), mask)
    excel_out = write_excel_safely(pd.DataFrame(detections), excel_out)

    # 如果有被过滤的候选，也保存到单独的 Excel
    if rejected:
        rejected_out = out_dir / f"{stem}_rejected.xlsx"
        write_excel_safely(pd.DataFrame(rejected), rejected_out)
        print(f"被过滤候选：{rejected_out}")

    print(f"\n最终结果：{source.name}")
    print(f"确认树木数量：{len(detections)}")
    if rejected:
        print(f"过滤误检数量：{len(rejected)}")
    print(f"圈画图片：{img_out}")
    print(f"掩膜图片：{mask_out}")
    print(f"Excel：{excel_out}")


def process_video(source: Path, out_dir: Path, args):
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot read video: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = out_dir / f"{source.stem}_detected.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # 演示用：每帧独立检测；正式视频建议加 ByteTrack 做跨帧跟踪
        detections, _ = detect_with_preset(frame, args)
        vis = draw_detections(frame, detections, show_id=False, **draw_style_for_preset(args.preset))
        cv2.putText(
            vis,
            f"Trees: {len(detections)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(vis)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"视频处理完成：{out_path}")


def is_video(path: Path):
    return path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


def write_excel_safely(df: pd.DataFrame, path: Path) -> Path:
    try:
        df.to_excel(path, index=False)
        return path
    except PermissionError:
        fallback = path.with_name(f"{path.stem}_new{path.suffix}")
        df.to_excel(fallback, index=False)
        print(f"Excel 文件被占用，已改写到：{fallback}")
        return fallback


def parse_args():
    parser = argparse.ArgumentParser(description="Drone tree crown detection demo")
    parser.add_argument("--source", required=True, help="图片或视频路径")
    parser.add_argument("--out", default="output", help="输出目录")
    parser.add_argument("--preset", choices=["ensemble", "reference", "original", "balanced", "recall", "precision", "custom"], default="ensemble", help="检测预设：ensemble=多策略高召回并补漏，reference=参考大模型标注风格，original=最初版本，balanced=均衡，recall=高召回")
    parser.add_argument("--min-area", type=float, default=120, help="最小树冠面积像素")
    parser.add_argument("--max-area", type=float, default=6800, help="最大树冠面积像素")
    parser.add_argument("--green-s", type=int, default=35, help="HSV 饱和度阈值，越大越不容易把阴影识别成树")
    parser.add_argument("--green-v", type=int, default=45, help="HSV 亮度阈值")
    parser.add_argument("--exg-min", type=int, default=12, help="ExG 植被指数下限，越大越严格")
    parser.add_argument("--circularity-min", type=float, default=0.12, help="圆度过滤阈值")
    parser.add_argument("--fill-ratio-min", type=float, default=0.28, help="树冠填充率过滤阈值，过滤狭长/松散误检")
    parser.add_argument("--radius-scale", type=float, default=1.15, help="按树冠面积估算圆半径时的放大系数")
    parser.add_argument("--min-radius", type=int, default=8, help="最小标注圆半径")
    parser.add_argument("--max-radius", type=int, default=58, help="最大标注圆半径")
    parser.add_argument("--no-split", action="store_true", help="关闭距离变换/分水岭切分，使用旧的连通块方式")
    parser.add_argument("--peak-distance", type=int, default=16, help="相邻树冠中心的最小像素间距")
    parser.add_argument("--min-peak-radius", type=int, default=0, help="作为树冠中心候选的最小距离半径（0=自适应）")
    parser.add_argument("--merge-overlap", type=float, default=0.62, help="重叠圆去重阈值，0 表示不去重")
    parser.add_argument("--show-id", action="store_true", help="是否在图片上显示编号")
    parser.add_argument("--draw-contour", action="store_true", help="用实际轮廓替代圆圈进行可视化")
    parser.add_argument("--work-max-dim", type=int, default=2048, help="检测工作图最长边。大图会先缩放检测再映射回原图，0 表示关闭")
    parser.add_argument("--use-lab", action="store_true", help="启用 Lab 色彩空间辅助检测（custom 模式）")
    parser.add_argument("--shadow-filter", action="store_true", help="启用阴影过滤（custom 模式）")
    parser.add_argument("--use-ellipse-fit", action="store_true", help="启用椭圆拟合计算半径（custom 模式）")
    parser.add_argument("--no-gap-fill", action="store_true", help="ensemble 模式关闭未覆盖掩膜补漏")
    parser.add_argument("--no-peak-supplement", action="store_true", help="ensemble 模式关闭距离峰值补点")
    parser.add_argument("--ensemble-merge-overlap", type=float, default=0.60, help="ensemble 模式候选去重重叠阈值")
    parser.add_argument("--no-local-refine", action="store_true",
                        help="关闭本地树冠核心过滤/半径收缩")
    parser.add_argument("--local-refine-min-green", type=float, default=0.28,
                        help="本地过滤要求圆内绿色树冠核心占比")
    parser.add_argument("--local-refine-max-radius", type=int, default=120,
                        help="本地过滤后允许的最大圆半径")
    parser.add_argument("--no-dense-dedupe", action="store_true",
                        help="关闭密集区域重复候选的最终去重")
    parser.add_argument("--dense-dedupe-factor", type=float, default=1.80,
                        help="密集区去重中心距离系数，越大越激进")
    parser.add_argument("--dense-dedupe-overlap", type=float, default=0.10,
                        help="密集区去重要求的最小小圆覆盖比例")

    # --- 大模型验证参数 ---
    parser.add_argument("--llm-verify", action="store_true",
                        help="启用 GPT-4o 大模型双重验证（需要设置 OPENAI_API_KEY 环境变量）")
    parser.add_argument("--llm-model", default="gpt-4o",
                        help="OpenAI 模型名称（默认 gpt-4o）")
    parser.add_argument("--llm-batch-size", type=int, default=9,
                        help="大模型验证每批发送几棵树（默认 9）")
    parser.add_argument("--llm-max-workers", type=int, default=1,
                        help="大模型并发请求数。修正补漏建议先用 1，避免请求过多")
    parser.add_argument("--llm-min-confidence", type=int, default=30,
                        help="LLM 判为树但低于该置信度时标记为低置信保留")
    parser.add_argument("--llm-reject-confidence", type=int, default=85,
                        help="保守模式下，LLM 判为非树且达到该置信度才过滤")
    parser.add_argument("--llm-strict", action="store_true",
                        help="关闭保守过滤，LLM 判为非树就直接过滤")
    parser.add_argument("--llm-repair", action="store_true",
                        help="启用 tile 级 LLM 修正补漏：保留/删除/调整/OpenCV 漏检新增")
    parser.add_argument("--llm-tile-size", type=int, default=1024,
                        help="LLM 修正补漏的 tile 尺寸")
    parser.add_argument("--llm-tile-overlap", type=float, default=0.25,
                        help="LLM 修正补漏 tile 重叠比例")
    parser.add_argument("--llm-min-radius", type=int, default=5,
                        help="LLM 新增/调整树冠的最小半径")
    parser.add_argument("--llm-max-radius", type=int, default=80,
                        help="LLM 新增/调整树冠的最大半径")
    parser.add_argument("--llm-dedupe-overlap", type=float, default=0.58,
                        help="LLM 修正补漏后圆形去重阈值")
    parser.add_argument("--llm-repair-strict-candidates", action="store_true",
                        help="修正补漏时强制要求 LLM 覆盖每个候选；未被提到的候选会被过滤（风险较高）")
    parser.add_argument("--show-species", action="store_true", default=True,
                        help="在图上标注树种名称（默认开启）")
    parser.add_argument("--no-show-species", dest="show_species", action="store_false",
                        help="不在图上标注树种名称")
    parser.add_argument("--show-rejected", action="store_true",
                        help="在图上用红色虚圈标记被大模型过滤的候选")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    source = Path(args.source)
    out_dir = Path(args.out)
    ensure_dir(out_dir)

    if is_video(source):
        process_video(source, out_dir, args)
    else:
        process_image(source, out_dir, args)
