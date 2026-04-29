import argparse
import math
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
DETECTION_COLUMNS = ["id", "x", "y", "radius", "conf", "x1", "y1", "x2", "y2"]


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def generate_tiles(width, height, tile_size, overlap):
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


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / (area_a + area_b - inter)


def is_center_duplicate(det, existing, center_factor, min_radius_ratio):
    if center_factor <= 0:
        return False

    r1 = float(det["radius"])
    r2 = float(existing["radius"])
    min_radius = min(r1, r2)
    max_radius = max(r1, r2)
    if min_radius <= 0 or max_radius <= 0:
        return False

    radius_ratio = min_radius / max_radius
    if radius_ratio < min_radius_ratio:
        return False

    distance = math.hypot(det["x"] - existing["x"], det["y"] - existing["y"])
    return distance <= max(8.0, min_radius * center_factor)


def nms_detections(detections, iou_threshold, center_factor=1.25, min_radius_ratio=0.40):
    kept = []
    for det in sorted(detections, key=lambda d: d["conf"], reverse=True):
        box = (det["x1"], det["y1"], det["x2"], det["y2"])
        duplicate = False
        for existing in kept:
            existing_box = (existing["x1"], existing["y1"], existing["x2"], existing["y2"])
            if (
                box_iou(box, existing_box) >= iou_threshold
                or is_center_duplicate(det, existing, center_factor, min_radius_ratio)
            ):
                duplicate = True
                break
        if not duplicate:
            kept.append(det)
    return kept


def collect_sources(source_args):
    sources = []
    for item in source_args:
        path = Path(item)
        if any(char in item for char in "*?[]"):
            matches = sorted(p for p in path.parent.glob(path.name) if p.suffix.lower() in IMAGE_SUFFIXES)
            sources.extend(matches)
        elif path.is_dir():
            matches = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
            sources.extend(matches)
        else:
            sources.append(path)

    unique = []
    seen = set()
    for source in sources:
        key = source.resolve()
        if key in seen:
            continue
        seen.add(key)
        unique.append(source)
    return unique


def predict_full_image(model, source: Path, conf: float, imgsz: int):
    results = model.predict(str(source), conf=conf, imgsz=imgsz, verbose=False)
    detections = []

    for result in results:
        if result.boxes is not None:
            for idx, box in enumerate(result.boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
                conf_score = float(box.conf[0].cpu().numpy())
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                radius = int(max(x2 - x1, y2 - y1) / 2)

                detections.append({
                    "id": idx + 1,
                    "x": cx,
                    "y": cy,
                    "radius": radius,
                    "conf": round(conf_score, 4),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                })

    return detections


def predict_tiled(
    model,
    img,
    conf: float,
    tile_size: int,
    overlap: float,
    nms_iou: float,
    nms_center_factor: float,
):
    h, w = img.shape[:2]
    detections = []

    for tx, ty, tw, th in generate_tiles(w, h, tile_size, overlap):
        crop = img[ty:ty + th, tx:tx + tw]
        results = model.predict(crop, conf=conf, imgsz=tile_size, verbose=False)
        result = results[0]
        if result.boxes is None:
            continue

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
            conf_score = float(box.conf[0].cpu().numpy())

            gx1 = max(0, min(w - 1, x1 + tx))
            gy1 = max(0, min(h - 1, y1 + ty))
            gx2 = max(0, min(w - 1, x2 + tx))
            gy2 = max(0, min(h - 1, y2 + ty))
            if gx2 <= gx1 or gy2 <= gy1:
                continue

            cx = int((gx1 + gx2) / 2)
            cy = int((gy1 + gy2) / 2)
            radius = int(max(gx2 - gx1, gy2 - gy1) / 2)
            detections.append({
                "x": cx,
                "y": cy,
                "radius": radius,
                "conf": round(conf_score, 4),
                "x1": gx1,
                "y1": gy1,
                "x2": gx2,
                "y2": gy2,
            })

    detections = nms_detections(
        detections,
        nms_iou,
        center_factor=nms_center_factor,
    )
    for idx, det in enumerate(sorted(detections, key=lambda d: (d["y"], d["x"])), 1):
        det["id"] = idx
    return detections


def draw_detections(img, detections):
    out = img.copy()
    for det in detections:
        center = (det["x"], det["y"])
        cv2.circle(out, center, det["radius"], (0, 255, 255), 2)
        cv2.putText(
            out,
            str(det["id"]),
            (det["x"] + 3, det["y"] - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),
            1,
        )
    return out


def process_image(source: Path, model, out_dir: Path, conf: float,
                  tile_size: int, overlap: float, nms_iou: float,
                  nms_center_factor: float, full_image: bool):
    img = cv2.imread(str(source))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {source}")

    if full_image:
        detections = predict_full_image(model, source, conf=conf, imgsz=max(tile_size, 1280))
        for idx, det in enumerate(sorted(detections, key=lambda d: (d["y"], d["x"])), 1):
            det["id"] = idx
    else:
        detections = predict_tiled(
            model,
            img,
            conf=conf,
            tile_size=tile_size,
            overlap=overlap,
            nms_iou=nms_iou,
            nms_center_factor=nms_center_factor,
        )

    vis = draw_detections(img, detections)
    stem = source.stem
    cv2.imwrite(str(out_dir / f"{stem}_yolo_detected.jpg"), vis)
    pd.DataFrame(detections, columns=DETECTION_COLUMNS).to_excel(out_dir / f"{stem}_yolo_trees.xlsx", index=False)
    print(f"{source}: 检测完成 {len(detections)} 棵")


def process_batch(sources, model_path: Path, out_dir: Path, conf: float,
                  tile_size: int, overlap: float, nms_iou: float,
                  nms_center_factor: float, full_image: bool):
    if not sources:
        raise FileNotFoundError("没有找到可处理的图片")

    model = YOLO(str(model_path))
    for source in sources:
        process_image(
            source,
            model,
            out_dir,
            conf=conf,
            tile_size=tile_size,
            overlap=overlap,
            nms_iou=nms_iou,
            nms_center_factor=nms_center_factor,
            full_image=full_image,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8-seg tree crown detection")
    parser.add_argument("--source", required=True, nargs="+", help="图片、目录或通配符；可一次传入多个")
    parser.add_argument("--model", required=True, help="训练好的 YOLOv8-seg 模型，例如 best.pt")
    parser.add_argument("--out", default="output", help="输出目录")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--tile-size", type=int, default=640, help="滑窗推理切片尺寸")
    parser.add_argument("--overlap", type=float, default=0.30, help="滑窗推理重叠比例")
    parser.add_argument("--nms-iou", type=float, default=0.45, help="滑窗结果合并 IoU 阈值")
    parser.add_argument("--nms-center-factor", type=float, default=1.25,
                        help="滑窗结果中心距离去重系数，0 表示关闭")
    parser.add_argument("--full-image", action="store_true", help="关闭滑窗，直接整图推理")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_dir = Path(args.out)
    ensure_dir(out_dir)
    process_batch(
        collect_sources(args.source),
        Path(args.model),
        out_dir,
        args.conf,
        tile_size=args.tile_size,
        overlap=args.overlap,
        nms_iou=args.nms_iou,
        nms_center_factor=args.nms_center_factor,
        full_image=args.full_image,
    )
