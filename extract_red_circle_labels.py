import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def red_mask(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 70, 60], dtype=np.uint8)
    upper1 = np.array([12, 255, 255], dtype=np.uint8)
    lower2 = np.array([168, 70, 60], dtype=np.uint8)
    upper2 = np.array([179, 255, 255], dtype=np.uint8)
    mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1), cv2.inRange(hsv, lower2, upper2))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def ring_coverage(mask, x, y, r, samples=96, thickness=3):
    hits = 0
    total = 0
    h, w = mask.shape[:2]
    for idx in range(samples):
        angle = 2.0 * math.pi * idx / samples
        for offset in range(-thickness, thickness + 1):
            rr = max(1, r + offset)
            px = int(round(x + rr * math.cos(angle)))
            py = int(round(y + rr * math.sin(angle)))
            if 0 <= px < w and 0 <= py < h:
                total += 1
                if mask[py, px] > 0:
                    hits += 1
    if total == 0:
        return 0.0
    return hits / total


def circle_overlap_area(r1, r2, distance):
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


def dedupe_circles(circles, overlap_threshold=0.62):
    kept = []
    for circle in sorted(circles, key=lambda c: (c["score"], c["radius"]), reverse=True):
        duplicate = False
        for existing in kept:
            distance = math.hypot(circle["x"] - existing["x"], circle["y"] - existing["y"])
            min_r = max(1.0, min(circle["radius"], existing["radius"]))
            if distance <= max(4.0, min_r * 0.38):
                duplicate = True
                break
            overlap = circle_overlap_area(circle["radius"], existing["radius"], distance)
            if overlap / (math.pi * min_r * min_r) >= overlap_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(circle)
    kept.sort(key=lambda c: (c["y"], c["x"]))
    for idx, circle in enumerate(kept, 1):
        circle["id"] = idx
    return kept


def extract_circles(
    overlay_img,
    original_img=None,
    min_radius=8,
    max_radius=90,
    param2=13,
    min_coverage=0.16,
    overlap_threshold=0.62,
):
    mask = red_mask(overlay_img)
    blurred = cv2.GaussianBlur(mask, (9, 9), 1.5)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(6, min_radius),
        param1=80,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    candidates = []
    if circles is not None:
        for x, y, r in np.round(circles[0]).astype(int):
            coverage = ring_coverage(mask, x, y, r)
            if coverage < min_coverage:
                continue
            candidates.append({
                "x": int(x),
                "y": int(y),
                "radius": int(r),
                "score": round(float(coverage), 4),
                "source": "red_overlay",
            })

    circles = dedupe_circles(candidates, overlap_threshold=overlap_threshold)

    if original_img is not None:
        oh, ow = original_img.shape[:2]
        hh, ww = overlay_img.shape[:2]
        sx = ow / float(ww)
        sy = oh / float(hh)
        sr = (sx + sy) / 2.0
        for circle in circles:
            circle["overlay_x"] = circle["x"]
            circle["overlay_y"] = circle["y"]
            circle["overlay_radius"] = circle["radius"]
            circle["x"] = int(round(circle["x"] * sx))
            circle["y"] = int(round(circle["y"] * sy))
            circle["radius"] = int(round(circle["radius"] * sr))

    return circles, mask


def draw_preview(img_bgr, circles):
    out = img_bgr.copy()
    for circle in circles:
        center = (int(circle["x"]), int(circle["y"]))
        radius = int(circle["radius"])
        cv2.circle(out, center, radius, (0, 0, 255), 2)
        cv2.circle(out, center, 2, (0, 255, 255), -1)
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="Extract x/y/radius labels from a red-circle overlay image")
    parser.add_argument("--overlay", required=True, help="带红圈的大模型标注图")
    parser.add_argument("--original", required=True, help="干净原图，用于坐标缩放和预览")
    parser.add_argument("--out", required=True, help="输出 Excel/CSV 标签")
    parser.add_argument("--preview", default=None, help="输出预览图")
    parser.add_argument("--min-radius", type=int, default=8, help="红圈最小半径（overlay 图坐标）")
    parser.add_argument("--max-radius", type=int, default=95, help="红圈最大半径（overlay 图坐标）")
    parser.add_argument("--param2", type=float, default=13, help="HoughCircles 累加阈值，越小检出越多")
    parser.add_argument("--min-coverage", type=float, default=0.16, help="圆周红色覆盖率下限")
    parser.add_argument("--overlap-threshold", type=float, default=0.62, help="重复圆去重阈值")
    return parser.parse_args()


def main():
    args = parse_args()
    overlay_path = Path(args.overlay)
    original_path = Path(args.original)
    out_path = Path(args.out)

    overlay = cv2.imread(str(overlay_path))
    if overlay is None:
        raise FileNotFoundError(f"Cannot read overlay: {overlay_path}")
    original = cv2.imread(str(original_path))
    if original is None:
        raise FileNotFoundError(f"Cannot read original: {original_path}")

    circles, mask = extract_circles(
        overlay,
        original_img=original,
        min_radius=args.min_radius,
        max_radius=args.max_radius,
        param2=args.param2,
        min_coverage=args.min_coverage,
        overlap_threshold=args.overlap_threshold,
    )

    ensure_dir(out_path.parent)
    df = pd.DataFrame(circles)
    if out_path.suffix.lower() == ".csv":
        df.to_csv(out_path, index=False)
    else:
        df.to_excel(out_path, index=False)

    preview_path = Path(args.preview) if args.preview else out_path.with_name(f"{out_path.stem}_preview.jpg")
    ensure_dir(preview_path.parent)
    cv2.imwrite(str(preview_path), draw_preview(original, circles))
    cv2.imwrite(str(out_path.with_name(f"{out_path.stem}_red_mask.jpg")), mask)

    print(f"提取红圈数量: {len(circles)}")
    print(f"标签输出: {out_path}")
    print(f"预览图片: {preview_path}")


if __name__ == "__main__":
    main()
