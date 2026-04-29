import argparse
import os
import subprocess
import sys
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def load_dotenv(path: Path):
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def collect_sources(source_args):
    sources = []
    for item in source_args:
        path = Path(item)
        if any(char in item for char in "*?[]"):
            sources.extend(sorted(p for p in path.parent.glob(path.name) if p.suffix.lower() in IMAGE_SUFFIXES))
        elif path.is_dir():
            sources.extend(sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES))
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


def parse_args():
    parser = argparse.ArgumentParser(description="Batch LLM tree crown labeling")
    parser.add_argument("--source", required=True, nargs="+", help="待标注图片、目录或通配符")
    parser.add_argument("--out", default="labels_llm", help="输出目录")
    parser.add_argument("--env-file", default=".env", help="包含 OPENAI_API_KEY/OPENAI_BASE_URL 的 .env 文件")
    parser.add_argument("--llm-model", default="gpt-4o", help="大模型名称")
    parser.add_argument("--preset", default="ensemble", choices=["ensemble", "reference", "original", "balanced", "recall", "precision", "custom"])
    parser.add_argument("--llm-tile-size", type=int, default=1024)
    parser.add_argument("--llm-tile-overlap", type=float, default=0.25)
    parser.add_argument("--llm-min-confidence", type=int, default=30)
    parser.add_argument("--llm-reject-confidence", type=int, default=85)
    parser.add_argument("--llm-max-workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 张；0 表示不限制")
    parser.add_argument("--start-after", default="", help="从某个文件名之后开始处理，用于中断后续跑")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的 *_trees.xlsx")
    parser.add_argument("--show-rejected", action="store_true", help="可视化中显示被过滤候选")
    parser.add_argument("--dry-run", action="store_true", help="只打印将要执行的命令")
    return parser.parse_args()


def main():
    args = parse_args()
    load_dotenv(Path(args.env_file))

    if not args.dry_run and not os.environ.get("OPENAI_API_KEY"):
        raise ValueError(
            "没有找到 OPENAI_API_KEY。请先设置环境变量，或在 .env 里写入：\n"
            "OPENAI_API_KEY=sk-xxx\n"
            "如果你用兼容接口，也可以加 OPENAI_BASE_URL=https://your-proxy/v1"
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = collect_sources(args.source)
    if args.start_after:
        sources = [source for source in sources if source.name > args.start_after]
    if args.limit > 0:
        sources = sources[:args.limit]
    if not sources:
        raise FileNotFoundError("没有找到需要标注的图片")

    print(f"待处理图片: {len(sources)} 张")
    print(f"输出目录: {out_dir}")
    print(f"模型: {args.llm_model}")

    for idx, source in enumerate(sources, 1):
        excel_out = out_dir / f"{source.stem}_trees.xlsx"
        if excel_out.exists() and not args.overwrite:
            print(f"[{idx}/{len(sources)}] 跳过已存在: {excel_out}")
            continue

        cmd = [
            sys.executable,
            "detect_trees_opencv.py",
            "--source",
            str(source),
            "--out",
            str(out_dir),
            "--preset",
            args.preset,
            "--llm-repair",
            "--llm-model",
            args.llm_model,
            "--llm-tile-size",
            str(args.llm_tile_size),
            "--llm-tile-overlap",
            str(args.llm_tile_overlap),
            "--llm-min-confidence",
            str(args.llm_min_confidence),
            "--llm-reject-confidence",
            str(args.llm_reject_confidence),
            "--llm-max-workers",
            str(args.llm_max_workers),
            "--no-show-species",
        ]
        if args.show_rejected:
            cmd.append("--show-rejected")

        print(f"\n[{idx}/{len(sources)}] 大模型标注: {source}")
        print(" ".join(cmd))
        if args.dry_run:
            continue

        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
