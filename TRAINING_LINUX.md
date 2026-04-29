# Linux 服务器训练 YOLO 树冠模型

这份说明使用你当前的两张人工修正图：

- `raw_images/DJI_0108.JPG` + `labels_corrected/DJI_0108_trees.xlsx`
- `raw_images/DJI_0109.JPG` + `labels_corrected/DJI_0109_trees.xlsx`

Excel 标注格式是 `x, y, radius`，脚本会把圆形树冠标注转换成 YOLO 检测框，并把大图切成小图训练。

## 1. 上传项目到服务器

在本机 PowerShell 可以打包这些文件：

```powershell
Compress-Archive -Force -DestinationPath tree_crown_train_package.zip -Path `
  prepare_yolo_dataset.py, `
  train_yolo_local.py, `
  detect_trees_yolo_seg.py, `
  requirements.txt, `
  yolov8n.pt, `
  raw_images, `
  labels_corrected
```

上传到服务器：

```bash
scp tree_crown_train_package.zip user@your-server:/home/user/
```

服务器上解压：

```bash
mkdir -p ~/tree-crown-demo
unzip ~/tree_crown_train_package.zip -d ~/tree-crown-demo
cd ~/tree-crown-demo
```

## 2. 创建 Python 环境

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如果服务器运行 `import cv2` 报 `libGL.so.1` 或 `libgthread-2.0.so.0` 之类错误，Ubuntu/Debian 可补系统库：

```bash
sudo apt-get update
sudo apt-get install -y libgl1 libglib2.0-0
```

如果是 NVIDIA GPU 服务器，先用下面命令确认显卡：

```bash
nvidia-smi
```

如果 `torch.cuda.is_available()` 是 `False`，需要按服务器 CUDA 版本从 PyTorch 官网选择对应的 pip 安装命令，然后再装 `requirements.txt`：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## 3. 生成 YOLO 数据集

```bash
python prepare_yolo_dataset.py \
  --source raw_images/DJI_0108.JPG raw_images/DJI_0109.JPG \
  --labels labels_corrected/DJI_0108_trees.xlsx labels_corrected/DJI_0109_trees.xlsx \
  --out datasets/tree_crowns_yolo_0108_0109 \
  --tile-size 640 \
  --overlap 0.30 \
  --val-ratio 0.20 \
  --include-empty
```

生成后会得到：

```text
datasets/tree_crowns_yolo_0108_0109/
  data.yaml
  images/train/*.jpg
  images/val/*.jpg
  labels/train/*.txt
  labels/val/*.txt
```

## 4. 训练

GPU 训练：

```bash
python train_yolo_local.py \
  --data datasets/tree_crowns_yolo_0108_0109/data.yaml \
  --model yolov8n.pt \
  --epochs 120 \
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --workers 4 \
  --name tree_crown_yolov8n_0108_0109
```

显存不够时优先改小：

```bash
--batch 4
```

没有 GPU 也能跑，只是会慢很多：

```bash
python train_yolo_local.py \
  --data datasets/tree_crowns_yolo_0108_0109/data.yaml \
  --model yolov8n.pt \
  --epochs 80 \
  --imgsz 640 \
  --batch 4 \
  --device cpu \
  --workers 2 \
  --name tree_crown_yolov8n_cpu
```

训练完成后重点看这个文件：

```text
runs/tree_crown_train/tree_crown_yolov8n_0108_0109/weights/best.pt
```

## 5. 用模型预标注新图片

```bash
python detect_trees_yolo_seg.py \
  --source raw_images/DJI_0108.JPG \
  --model runs/tree_crown_train/tree_crown_yolov8n_0108_0109/weights/best.pt \
  --out output_yolo_local_0108 \
  --conf 0.08 \
  --tile-size 640 \
  --overlap 0.30
```

输出里会有：

- `*_yolo_detected.jpg`：可视化圈图
- `*_yolo_trees.xlsx`：模型生成的树冠坐标，可以继续人工修正

## 6. 继续迭代

1. 用 `best.pt` 给更多未标注图片生成 Excel。
2. 人工删除错框、补漏框、调半径。
3. 把修正后的图片和 Excel 加入 `--source` / `--labels`。
4. 重新运行数据集生成和训练命令。

只用 2 张图训练时，模型主要是“帮你预标注”，不要直接当最终结果。每轮人工修正后再训练，效果会明显稳一些。
