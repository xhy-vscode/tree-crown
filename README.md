# AI Tree Crown Detection

无人机航拍树冠识别、计数、复核和可视化 Demo。项目面向果园、林地、绿化巡检等场景，把高分辨率航拍图中的树冠自动圈画出来，生成树木数量、中心点、半径、面积、来源策略和大模型复核结果，并导出 Excel。

这个项目不是单一 OpenCV 脚本，而是一条可迭代的 AI/Agent 工作流：传统视觉先生成高召回候选，GPT-4o Vision 做二次验证和补漏，本地 YOLO 模型再用修正后的标注继续训练，最后通过 Web Demo 展示结果。

## 项目亮点

- 自动处理无人机俯拍图，输出树冠圈画图、掩膜图、Excel 统计表和 rejected 误检表。
- 使用 HSV、ExG、Lab、距离变换、分水岭和多策略候选融合，优先减少漏检。
- 支持 GPT-4o Vision 二次验证，过滤草斑、阴影、裸地、灌木等误检，并可输出树种、置信度和判断原因。
- 支持 LLM repair 模式，让视觉模型同时查看原图 tile 和候选叠加图，执行保留、删除、调整和补漏。
- 支持人工复核、伪标签生成、YOLOv8 本地训练、滑窗推理和批量预标注。
- 内置可视化交互大屏，可展示图像、树冠圈画、统计指标和算法来源占比。

## 当前成果

当前仓库中已有一轮可验证结果：

- 原始无人机航拍图：66 张。
- 伪标注/检测结果表：128 个 Excel。
- YOLO 切片训练样本：207 张图像切片和 207 个标签文件。
- 单张示例 `DJI_0108` 在增强检测流程中输出 1272 个树冠候选。
- GPT-4o Vision 复核示例输出 659 条验证后树冠记录，并带有 `species`、`llm_confidence`、`llm_decision` 字段。
- 本地 YOLOv8 训练已生成 `best.pt`、训练曲线、验证预测图和混淆矩阵。

## 结果截图

### 多策略树冠检测

![Tree crown detection](docs/assets/tree-crown-detection.jpg)

### GPT-4o Vision 二次验证

![LLM verified result](docs/assets/llm-verified-result.jpg)

### YOLOv8 训练曲线

![YOLO training results](docs/assets/yolo-training-results.png)

### YOLO 验证集预测

![YOLO validation prediction](docs/assets/yolo-val-prediction.jpg)

更多截图和说明见 [docs/RESULTS.md](docs/RESULTS.md)。

## 工作流

```text
无人机原图
  -> OpenCV 植被掩膜和候选生成
  -> 多策略合并、补漏、去重
  -> GPT-4o Vision 验证/修正/补漏
  -> Excel 结构化结果
  -> 人工复核
  -> YOLO 数据集生成
  -> 本地 YOLO 训练和滑窗推理
  -> Web Demo 可视化展示
```

## 安装

```bash
python -m venv .venv
```

Windows:

```powershell
.\.venv\Scripts\activate
pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## 快速运行

对单张航拍图做树冠检测：

```bash
python detect_trees_opencv.py --source ./DJI_0108.JPG --out ./output
```

输出文件：

- `output/xxx_detected.jpg`：树冠圈画图。
- `output/xxx_mask.jpg`：树冠植被掩膜。
- `output/xxx_trees.xlsx`：每棵树的中心点、半径、面积和候选来源。
- `output/xxx_rejected.xlsx`：开启大模型验证后，被过滤的候选。

默认流程使用 `--preset ensemble`，会先运行参考检测，再运行更宽松的高召回检测，并通过未覆盖区域和距离峰值做补漏，最后统一合并去重。

## GPT-4o Vision 验证

设置 API Key：

```powershell
$env:OPENAI_API_KEY="sk-your-api-key"
```

基础二次验证：

```bash
python detect_trees_opencv.py --source ./DJI_0108.JPG --out ./output_llm --llm-verify
```

修正和补漏模式：

```bash
python detect_trees_opencv.py --source ./DJI_0108.JPG --out ./output_llm_repair --llm-repair --llm-tile-size 1024 --llm-tile-overlap 0.25 --no-show-species
```

批量调用大模型标注：

```bash
python batch_llm_label.py --source ./raw_images --out ./labels_llm --llm-model gpt-4o --llm-tile-size 1024 --llm-tile-overlap 0.25
```

成本控制策略：系统会把候选树冠裁切成网格图批量发送给视觉模型，默认每 9 棵候选树拼成一张图；约 200 棵候选树需要 23 次左右 API 调用。

## 人工复核

打开复核工具，手工删除误检、补充漏检、调整半径：

```powershell
.\.train_venv\Scripts\python.exe .\review_labels_app.py --source .\DJI_0108.JPG --labels .\output\DJI_0108_trees.xlsx --out .\labels_corrected\DJI_0108_trees_corrected.xlsx
```

## 训练本地 YOLO 模型

从 Excel 圆形标注生成 YOLO 数据集：

```powershell
.\.train_venv\Scripts\python.exe .\prepare_yolo_dataset.py --source .\DJI_0108.JPG --labels .\labels_corrected\DJI_0108_trees_corrected.xlsx --out .\datasets\tree_crowns_yolo
```

训练模型：

```powershell
.\.train_venv\Scripts\python.exe .\train_yolo_local.py --data .\datasets\tree_crowns_yolo\data.yaml --model yolov8n.pt --epochs 80 --device cpu
```

滑窗推理：

```powershell
.\.train_venv\Scripts\python.exe .\detect_trees_yolo_seg.py --source .\DJI_0108.JPG --model .\runs\detect\runs\tree_crown_train\tree_crown_yolov8n_pseudo\weights\best.pt --out .\output_yolo_local --conf 0.08 --tile-size 640 --overlap 0.30
```

## Visual Demo

准备半量演示集：

```powershell
.\.train_venv\Scripts\python.exe .\prepare_fast_demo.py --source-dir .\raw_images --labels-dir .\labels_pseudo --out .\demo_half_labeled --limit 33
```

启动可视化演示：

```powershell
.\.train_venv\Scripts\python.exe .\visual_demo_app.py --port 8888 --demo-dir .\demo_half_labeled
```

浏览器打开：

```text
http://127.0.0.1:8888
```

完整 Demo 说明见 [docs/DEMO.md](docs/DEMO.md)。

## 目录说明

```text
detect_trees_opencv.py       OpenCV 多策略树冠检测主流程
llm_verifier.py              GPT-4o Vision 候选验证模块
batch_llm_label.py           批量大模型标注入口
review_labels_app.py         人工复核工具
prepare_yolo_dataset.py      Excel 标注转 YOLO 数据集
train_yolo_local.py          本地 YOLO 训练入口
detect_trees_yolo_seg.py     YOLO 滑窗推理
visual_demo_app.py           可视化 Web Demo
docs/                        项目展示材料
docs/assets/                 README 和 Demo 使用的结果截图
```

## 备注

当前项目中的伪标签和训练结果主要用于演示完整工作流。真实生产部署建议继续增加人工修正样本，并针对不同季节、光照、树种、飞行高度和地块类型做数据扩充。
