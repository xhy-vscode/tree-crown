# Demo 说明

这份 Demo 用于向评审或合作方展示“无人机航拍图 -> AI 树冠识别 -> 大模型复核 -> 统计导出 -> 可视化展示”的完整流程。

## 演示目标

演示系统如何从一张或一批无人机航拍图中自动识别树冠，并输出可复核、可统计、可继续训练的数据。重点展示：

- 自动圈画树冠和统计数量。
- 输出每棵树的中心点、半径、面积和检测来源。
- 使用 GPT-4o Vision 过滤非树目标，并补充置信度和判断原因。
- 将检测结果转换为 YOLO 数据集，用于后续本地模型训练。
- 在 Web Demo 中浏览多张图片、结果图和统计信息。

## 推荐演示流程

### 1. 单张图片检测

```powershell
.\.train_venv\Scripts\python.exe .\detect_trees_opencv.py --source .\DJI_0108.JPG --out .\output_demo_single --preset ensemble
```

讲解点：

- 系统先从航拍图中提取植被区域。
- 再用多策略候选池减少漏检。
- 最终输出圈画图和 Excel。

### 2. 大模型二次验证

```powershell
$env:OPENAI_API_KEY="sk-your-api-key"
.\.train_venv\Scripts\python.exe .\detect_trees_opencv.py --source .\DJI_0108.JPG --out .\output_demo_llm --llm-verify --show-rejected --no-show-species
```

讲解点：

- 传统视觉负责高召回，先尽量把疑似树冠找出来。
- GPT-4o Vision 负责语义判断，过滤草斑、阴影、裸地等误检。
- 输出 Excel 会新增 `species`、`llm_confidence`、`llm_decision` 字段，方便人工复核。

### 3. 批量预标注

```powershell
.\.train_venv\Scripts\python.exe .\batch_llm_label.py --source .\raw_images --out .\labels_llm --llm-model gpt-4o --limit 5
```

讲解点：

- 适合批量处理航拍任务。
- 支持中断续跑和跳过已存在结果。
- 可以先处理小批量样本，再扩大到完整数据集。

### 4. 可视化大屏

```powershell
.\.train_venv\Scripts\python.exe .\prepare_fast_demo.py --source-dir .\raw_images --labels-dir .\labels_pseudo --out .\demo_half_labeled --limit 33
.\.train_venv\Scripts\python.exe .\visual_demo_app.py --port 8888 --demo-dir .\demo_half_labeled
```

浏览器打开：

```text
http://127.0.0.1:8888
```

讲解点：

- 左侧展示统计指标和数据集列表。
- 主视图区展示原图和树冠圈画结果。
- 可用于给非技术人员快速理解识别效果。

## 评审讲解词

这个项目解决的是林业和果园场景里人工数树、圈树、统计树冠面积效率低的问题。无人机一次飞行会产生大量高分辨率图片，传统人工标注成本很高。系统先用传统视觉算法生成高召回候选，再调用 GPT-4o Vision 进行二次判断和补漏，最后把结果导出为 Excel，并沉淀成 YOLO 训练数据。随着人工修正样本增加，本地模型会逐轮变得更稳定。

这个流程的 token 和图像额度需求比较高，因为一张航拍图可能包含数百到上千个候选树冠。每个候选都需要视觉模型看图判断，还要输出结构化 JSON、置信度和原因。批量处理多张图时，会产生大量图像 tile、候选网格、模型调用和结果修正任务。

## 推荐上传材料

- `docs/assets/tree-crown-detection.jpg`
- `docs/assets/llm-verified-result.jpg`
- `docs/assets/yolo-training-results.png`
- `docs/assets/yolo-val-prediction.jpg`
- `README.md`
- `docs/RESULTS.md`
