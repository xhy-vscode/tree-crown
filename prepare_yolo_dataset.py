import argparse
import random
import shutil
from pathlib import Path

import cv2
import pandas as pd


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
LABEL_SUFFIXES = {".xlsx", ".csv"}
LABEL_NAME_PATTERNS = (
    "{stem}_trees_corrected{suffix}",
    "{stem}_trees{suffix}",
    "{stem}_yolo_trees_corrected{suffix}",
    "{stem}_yolo_trees{suffix}",
    "{stem}{suffix}",
)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def read_labels(label_path: Path):
    if label_path.suffix.lower() == ".csv":
        df = pd.read_csv(label_path)
    else:
        df = pd.read_excel(label_path)
    required = {"x", "y", "radius"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{label_path} 缺少列: {', '.join(sorted(missing))}")

    labels = []
    for _, row in df.iterrows():
        x = float(row["x"])
        y = float(row["y"])
        r = float(row["radius"])
        if r <= 0:
            continue
        labels.append((x, y, r))
    return labels


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


def box_intersection_area(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)


def labels_for_tile(labels, tile, min_visible):
    tx, ty, tw, th = tile
    tile_box = (tx, ty, tx + tw, ty + th)
    yolo_labels = []

    for x, y, r in labels:
        full_box = (x - r, y - r, x + r, y + r)
        full_area = max(1.0, (full_box[2] - full_box[0]) * (full_box[3] - full_box[1]))
        visible = box_intersection_area(full_box, tile_box) / full_area
        if visible < min_visible:
            continue

        x1 = max(full_box[0], tile_box[0]) - tx
        y1 = max(full_box[1], tile_box[1]) - ty
        x2 = min(full_box[2], tile_box[2]) - tx
        y2 = min(full_box[3], tile_box[3]) - ty
        if x2 <= x1 or y2 <= y1:
            continue

        cx = ((x1 + x2) / 2.0) / tw
        cy = ((y1 + y2) / 2.0) / th
        bw = (x2 - x1) / tw
        bh = (y2 - y1) / th
        if bw <= 0 or bh <= 0:
            continue

        yolo_labels.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    return yolo_labels


def write_yaml(path: Path, dataset_root: Path):
    root = dataset_root.resolve().as_posix()
    text = (
        f"path: {root}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: tree\n"
    )
    path.write_text(text, encoding="utf-8")


def find_label_for_image(image_path: Path, label_dir: Path):
    for suffix in LABEL_SUFFIXES:
        for pattern in LABEL_NAME_PATTERNS:
            candidate = label_dir / pattern.format(stem=image_path.stem, suffix=suffix)
            if candidate.exists():
                return candidate
    return None


def collect_pairs(source_args, label_args):
    if len(source_args) == 1 and len(label_args) == 1:
        source_path = Path(source_args[0])
        label_path = Path(label_args[0])
        if source_path.is_dir() and label_path.is_dir():
            pairs = []
            images = sorted(p for p in source_path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
            for image_path in images:
                found = find_label_for_image(image_path, label_path)
                if found is not None:
                    pairs.append((image_path, found))
            if not pairs:
                raise FileNotFoundError(f"没有在 {label_path} 中找到与 {source_path} 图片对应的标签")
            return pairs

    if len(source_args) != len(label_args):
        raise ValueError("--source 和 --labels 的数量必须一致；或同时传入一个图片目录和一个标签目录")

    return [(Path(source), Path(labels)) for source, labels in zip(source_args, label_args)]


def build_dataset_multi(
    pairs,
    out_dir: Path,
    tile_size=640,
    overlap=0.25,
    val_ratio=0.18,
    min_visible=0.65,
    include_empty=False,
    seed=7,
):
    if out_dir.exists():
        shutil.rmtree(out_dir)

    for split in ("train", "val"):
        ensure_dir(out_dir / "images" / split)
        ensure_dir(out_dir / "labels" / split)

    rng = random.Random(seed)
    written = {"train": 0, "val": 0}
    object_count = {"train": 0, "val": 0}
    image_summaries = []

    for source, labels_xlsx in pairs:
        img = cv2.imread(str(source))
        if img is None:
            raise FileNotFoundError(f"无法读取图片: {source}")

        labels = read_labels(labels_xlsx)
        if not labels:
            raise ValueError(f"没有可用标注: {labels_xlsx}")

        h, w = img.shape[:2]
        tiles = list(generate_tiles(w, h, tile_size, overlap))
        rng.shuffle(tiles)
        stem = source.stem
        image_written = {"train": 0, "val": 0}
        image_objects = {"train": 0, "val": 0}

        for tile in tiles:
            tile_labels = labels_for_tile(labels, tile, min_visible=min_visible)
            if not tile_labels and not include_empty:
                continue

            split = "val" if rng.random() < val_ratio else "train"
            tx, ty, tw, th = tile
            crop = img[ty:ty + th, tx:tx + tw]
            name = f"{stem}_{tx}_{ty}_{tw}x{th}"

            cv2.imwrite(str(out_dir / "images" / split / f"{name}.jpg"), crop)
            (out_dir / "labels" / split / f"{name}.txt").write_text(
                "\n".join(tile_labels) + ("\n" if tile_labels else ""),
                encoding="utf-8",
            )
            written[split] += 1
            object_count[split] += len(tile_labels)
            image_written[split] += 1
            image_objects[split] += len(tile_labels)

        image_summaries.append((source, w, h, len(labels), image_written, image_objects))

    write_yaml(out_dir / "data.yaml", out_dir)

    print(f"数据集已生成: {out_dir}")
    for source, w, h, label_count, image_written, image_objects in image_summaries:
        print(
            f"{source}: {w}x{h}, 输入标注: {label_count}, "
            f"训练切片/框: {image_written['train']}/{image_objects['train']}, "
            f"验证切片/框: {image_written['val']}/{image_objects['val']}"
        )
    print(f"训练切片: {written['train']} 张, 标注框: {object_count['train']}")
    print(f"验证切片: {written['val']} 张, 标注框: {object_count['val']}")
    print(f"配置文件: {out_dir / 'data.yaml'}")


def build_dataset(
    source: Path,
    labels_xlsx: Path,
    out_dir: Path,
    tile_size=640,
    overlap=0.25,
    val_ratio=0.18,
    min_visible=0.65,
    include_empty=False,
    seed=7,
):
    build_dataset_multi(
        pairs=[(source, labels_xlsx)],
        out_dir=out_dir,
        tile_size=tile_size,
        overlap=overlap,
        val_ratio=val_ratio,
        min_visible=min_visible,
        include_empty=include_empty,
        seed=seed,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Convert tree crown Excel labels to a YOLO detection dataset")
    parser.add_argument("--source", required=True, nargs="+", help="原始无人机图片；可一次传入多张")
    parser.add_argument("--labels", required=True, nargs="+", help="包含 x/y/radius 的 Excel 或 CSV 标注；顺序需与 --source 对应")
    parser.add_argument("--out", default="datasets/tree_crowns_yolo", help="输出 YOLO 数据集目录")
    parser.add_argument("--tile-size", type=int, default=640, help="切片尺寸")
    parser.add_argument("--overlap", type=float, default=0.25, help="切片重叠比例")
    parser.add_argument("--val-ratio", type=float, default=0.18, help="验证集比例")
    parser.add_argument("--min-visible", type=float, default=0.65, help="目标框在切片中至少可见比例")
    parser.add_argument("--include-empty", action="store_true", help="保留没有树的负样本切片")
    parser.add_argument("--seed", type=int, default=7, help="随机种子")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_dataset_multi(
        pairs=collect_pairs(args.source, args.labels),
        out_dir=Path(args.out),
        tile_size=args.tile_size,
        overlap=args.overlap,
        val_ratio=args.val_ratio,
        min_visible=args.min_visible,
        include_empty=args.include_empty,
        seed=args.seed,
    )
