import argparse
from pathlib import Path


def train(
    data_yaml: Path,
    model: str,
    out_dir: Path,
    name: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    workers: int,
):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "当前 Python 环境缺少 ultralytics。请先安装训练依赖: "
            "pip install ultralytics torch torchvision"
        ) from exc

    if not data_yaml.exists():
        raise FileNotFoundError(f"找不到数据集配置: {data_yaml}")

    yolo = YOLO(model)
    results = yolo.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        project=str(out_dir),
        name=name,
        exist_ok=True,
        patience=30,
        cache=False,
        pretrained=True,
    )

    run_dir = Path(results.save_dir)
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    print(f"训练完成: {run_dir}")
    if best.exists():
        print(f"最佳模型: {best}")
    if last.exists():
        print(f"最后模型: {last}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a local YOLO tree crown detector")
    parser.add_argument("--data", default="datasets/tree_crowns_yolo/data.yaml", help="YOLO data.yaml")
    parser.add_argument("--model", default="yolov8n.pt", help="预训练模型，如 yolov8n.pt / yolov8s.pt")
    parser.add_argument("--out", default="runs/tree_crown_train", help="训练输出目录")
    parser.add_argument("--name", default="tree_crown_yolov8n", help="本次训练名称")
    parser.add_argument("--epochs", type=int, default=80, help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=640, help="训练尺寸")
    parser.add_argument("--batch", type=int, default=8, help="批大小；显存不足时调小")
    parser.add_argument("--device", default="cpu", help="cpu 或 0/0,1 等 GPU 编号")
    parser.add_argument("--workers", type=int, default=0, help="Windows 上默认 0 更稳")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        data_yaml=Path(args.data),
        model=args.model,
        out_dir=Path(args.out),
        name=args.name,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
    )
