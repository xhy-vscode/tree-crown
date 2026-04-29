import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def image_files(path: Path):
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def label_path_for(image: Path, labels_dir: Path):
    return labels_dir / f"{image.stem}_trees.xlsx"


def prepare_demo(source_dir: Path, labels_dir: Path, out_dir: Path, limit: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    images = image_files(source_dir)
    if limit > 0:
        images = images[:limit]

    items = []
    missing = []

    for image in images:
        label_path = label_path_for(image, labels_dir)
        if not label_path.exists():
            missing.append(image.name)
            continue

        out_image = out_dir / image.name
        out_label = out_dir / label_path.name
        shutil.copy2(image, out_image)
        shutil.copy2(label_path, out_label)

        tree_count = int(len(pd.read_excel(out_label)))
        items.append({
            "name": image.stem,
            "image_path": str(out_image).replace("\\", "/"),
            "data_path": str(out_label).replace("\\", "/"),
            "tree_count": tree_count,
        })

    summary = {
        "source_dir": str(source_dir),
        "labels_dir": str(labels_dir),
        "out_dir": str(out_dir),
        "requested_limit": limit,
        "prepared_images": len(items),
        "total_trees": sum(item["tree_count"] for item in items),
        "missing_labels": missing,
        "items": items,
    }
    (out_dir / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare a fast half-labeled tree crown demo set")
    parser.add_argument("--source-dir", default="raw_images", help="原始无人机图片目录")
    parser.add_argument("--labels-dir", default="labels_pseudo", help="已有伪标注 Excel 目录")
    parser.add_argument("--out", default="demo_half_labeled", help="演示集输出目录")
    parser.add_argument("--limit", type=int, default=33, help="整理前 N 张图片；默认 33，约等于 66 张的一半")
    return parser.parse_args()


def main():
    args = parse_args()
    summary = prepare_demo(
        source_dir=Path(args.source_dir),
        labels_dir=Path(args.labels_dir),
        out_dir=Path(args.out),
        limit=args.limit,
    )
    print(f"演示集已准备: {summary['out_dir']}")
    print(f"图片数量: {summary['prepared_images']}")
    print(f"标注树冠总数: {summary['total_trees']}")
    if summary["missing_labels"]:
        print("缺少标注:", ", ".join(summary["missing_labels"]))


if __name__ == "__main__":
    main()
