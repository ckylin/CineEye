# 影瞳 CineEye · 模型文件与使用指南

> 本文回答两个问题：**训练出来的模型在哪里？**，以及**怎么用它识别电影？**
>
> 所有路径都相对项目根目录 `/home/kylin/Workarea/CineEye/`，所有命令中的输出均来自本机实测。
> 当前模型是用 `/home/kylin/movie_train/` 下 6 部真实影片训练得到的结果。

---

## 1. 一句话结论

**模型主文件是 `outputs/checkpoints/best.pt`（约 42.7 MB）。** 它是一个 PyTorch checkpoint，里面包含完整的网络权重、类别映射和训练时的配置。用 `cineeye/predict.py` 里的 `Predictor` 类加载它，或者直接跑 `scripts/05_predict.py`，即可做单图、批量图片和整段视频识别。

---

## 2. 模型与相关文件都在哪里

### 2.1 核心文件（使用模型必须）

| 文件 | 大小 | 作用 |
|---|---|---|
| `outputs/checkpoints/best.pt` | 42.7 MB | **主模型**。ResNet18 主干 + 6 分类头，含权重与元数据 |
| `outputs/label_map.json` | 小 | 编号 ↔ 类别名的映射（`movie_01` → `0` 等），已内嵌在 `best.pt` 中 |
| `outputs/reports/movie_id_map.json` | 小 | **仅存于本机**：`movie_01~06` 对应的原始影片文件名与时长 |
| `outputs/checkpoints/actor_head.pt` | 小（< 1 MB） | **演员识别模型**：线性分类头 + 演员标签表 + 拒识阈值 + 编码器指纹。由 `scripts/16_train_actors.py` 产出，不需要与 `best.pt` 同时存在 |

> 注意：`best.pt` 内部已自带 `label_map` 和训练配置，所以**单独拷贝 `best.pt` 一个文件就能推理**，不需要 `label_map.json` 或配置文件。`movie_id_map.json` 只是给你人工对照片名用的，模型本身不读它。

### 2.2 附属产物（不需要用于推理，但可查阅）

| 文件 | 作用 |
|---|---|
| `outputs/checkpoints/classical_baseline.joblib` | scikit-learn 传统特征基线（HOG + 颜色 + 逻辑回归），用于对照，不是主模型 |
| `outputs/features/deep_{train,val,test}.npz` | 冻结主干提取并缓存的深度特征；重新训练分类头时可秒级复用 |
| `outputs/features/deep_*.meta.json` | 上述特征对应的路径、标签、分组等元信息 |
| `outputs/features/classical_*.npz` | sklearn 基线的特征缓存 |
| `outputs/features/actor_features.npz` | 演员照片的特征缓存（带签名，照片/编码器变化自动失效） |
| `outputs/actors/embeddings.npy` + `names.json` + `encoder.json` | 演员检索库向量、逐行对应的演员名、编码器指纹 |
| `outputs/reports/train_report.json` | 训练摘要（最佳 epoch、验证集指标、样本数） |
| `outputs/reports/eval_test.json` | 测试集帧级 / 视频级指标 |
| `outputs/reports/classification_report_test.txt` | 每类的精确率、召回率、F1 |
| `outputs/reports/confusion_matrix_test.{png,csv}` | 混淆矩阵 |
| `outputs/reports/predictions_test.csv` | 测试集逐帧预测明细 |
| `outputs/reports/history.csv` | 每轮训练的 loss / top1 曲线数据 |

查看全部文件：

```bash
cd /home/kylin/Workarea/CineEye
find outputs -type f -printf '%p  (%k KB)\n' | sort
```

---

## 3. `best.pt` 里面装了什么

用以下命令可以随时查看，不需要训练代码：

```bash
./.venv/bin/python -c "
import torch
ck = torch.load('outputs/checkpoints/best.pt', map_location='cpu', weights_only=False)
for k, v in ck.items():
    print(f'{k:16s} : {type(v).__name__}', getattr(v, 'shape', ''))
print('label_map:', ck['label_map'])
print('metrics  :', ck['metrics'])
"
```

当前实际内容：

| 字段 | 值 / 类型 | 说明 |
|---|---|---|
| `version` | `1` | checkpoint 格式版本 |
| `mode` | `"feature"` | 训练模式（冻结主干 + 分类头） |
| `backbone` | `"resnet18"` | 主干网络名 |
| `feature_dim` | `512` | 主干输出特征维度 |
| `num_classes` | `6` | 类别数（6 部电影） |
| `model_state` | `dict`（权重张量） | 完整网络参数，键形如 `backbone.conv1.weight`、`head.0.weight` |
| `label_map` | `dict` | `{'movie_01': 0, ..., 'movie_06': 5}` |
| `config` | `dict` | 训练时的完整配置（增强、分辨率、推理参数等），加载时优先于当前 yaml |
| `metrics` | `dict` | 最佳 epoch、验证集 top1/top5、训练样本数等 |
| `device_info` | `dict` | 训练机器的 Python / torch / 线程数等环境信息 |

**关键点**：推理代码用 `pretrained=False` 构建网络，权重全部从 `model_state` 加载，所以推理时**不需要联网、不需要下载 ImageNet 预训练权重**。

---

## 4. 标签体系怎么理解

模型输出的是中性编号，不是影片原名：

```
movie_01, movie_02, movie_03, movie_04, movie_05, movie_06
```

这样做是为了让训练产物不直接暴露片名。编号与原始文件的对应关系保存在：

```
outputs/reports/movie_id_map.json
```

内容形如（此处省略具体文件名，请直接查看该文件）：

```json
{
  "movie_01": {"source_file": "<原始文件名>", "duration_sec": 5513.1, "segments": 8, "frames": 320},
  "...":      {"...": "..."}
}
```

**对外提供接口时**，建议在服务端把编号翻译成片名（或业务方约定的 ID）后再返回；如果不想暴露片名，直接返回编号也可以。

---

## 5. 怎么用：三种方式

### 方式一：命令行（最快，验证用）

识别一张图片：

```bash
cd /home/kylin/Workarea/CineEye
TMPDIR=$PWD/.tmp ./.venv/bin/python scripts/05_predict.py \
  --image data/frames/movie_01/seg_01/000001.jpg
```

识别整段视频（多帧聚合，比单帧稳定得多）：

```bash
TMPDIR=$PWD/.tmp ./.venv/bin/python scripts/05_predict.py \
  --video "/home/kylin/movie_train/杏林春暖.1992.mp4" \
  --num-frames 16 --aggregation mean_prob
```

批量识别多张图片，并以 JSON 输出（便于程序解析）：

```bash
TMPDIR=$PWD/.tmp ./.venv/bin/python scripts/05_predict.py \
  --image 图片1.jpg 图片2.jpg 图片3.jpg --json
```

单图 JSON 实测输出：

```json
[
  {
    "label": "movie_01",
    "label_idx": 0,
    "confidence": 0.912992,
    "unknown": false,
    "top_k": [
      {"label": "movie_01", "confidence": 0.912992},
      {"label": "movie_04", "confidence": 0.066356},
      {"label": "movie_06", "confidence": 0.018229},
      {"label": "movie_02", "confidence": 0.001851},
      {"label": "movie_05", "confidence": 0.000379}
    ],
    "source": "data/frames/movie_01/seg_01/000001.jpg"
  }
]
```

文本模式输出示例：

```
来源: /home/kylin/movie_train/杏林春暖.1992.mp4
识别结果: movie_04   置信度: 0.6250
聚合方式: mean_prob | 采样帧数: 16 | 帧一致性: 75.00%
Top-K: movie_04 0.625, movie_01 0.181, ...
```

**参数说明**：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--image` | 无 | 一张或多张图片路径，相对路径会先按项目根目录解析 |
| `--video` | 无 | 视频路径，均匀抽帧后聚合 |
| `--checkpoint` | `outputs/checkpoints/best.pt` | 指定另一个模型文件 |
| `--num-frames` | `16` | 视频抽帧数；片段短或 CPU 慢可减小 |
| `--aggregation` | `mean_prob` | 聚合方式：`mean_prob` / `log_prob` / `majority` |
| `--json` | 关 | 以 JSON 输出，适合接入其他程序 |
| `--set inference.top_k=3` | `5` | 返回 Top-K 个数 |
| `--set inference.unknown_threshold=0.5` | `0.0` | 低于该置信度标记为未知影片 |

> 所有脚本都支持 `--set 配置项=值` 覆盖配置，配置层级见 `configs/default.yaml`。

### 方式二：Python API（集成到你自己的程序）

`Predictor` 类只加载一次模型，之后可反复调用，适合批处理或封装成服务。

```python
import sys
sys.path.insert(0, "/home/kylin/Workarea/CineEye")

from cineeye.predict import Predictor

# 加载模型（默认读 outputs/checkpoints/best.pt；可传 checkpoint="路径"）
predictor = Predictor()

# 1) 单张图片
r = predictor.predict_image("/path/to/frame.jpg")
print(r["label"], r["confidence"], r["top_k"])

# 2) 批量图片（一次前向，效率高）
results = predictor.predict_images(["a.jpg", "b.jpg", "c.jpg"])
for item in results:
    print(item["source"], "->", item["label"], round(item["confidence"], 4))

# 3) 整段视频（16 帧均匀采样 + 概率平均）
v = predictor.predict_video("/path/to/movie.mp4", num_frames=16, aggregation="mean_prob")
print(v["label"], v["confidence"], "帧一致性", v["frame_agreement"])
print("逐帧投票:", v["frame_votes"])

# 4) 已经在内存里的图像帧（BGR ndarray，来自 cv2）
import cv2
frames = [cv2.imread(f) for f in ["f1.jpg", "f2.jpg"]]
r2 = predictor.predict_frames(frames, aggregation="majority")
```

返回值字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `label` | `str` | 识别出的类别编号，如 `movie_04` |
| `label_idx` | `int` | 类别下标 |
| `confidence` | `float` | 置信度 0~1 |
| `unknown` | `bool` | 是否低于拒识阈值（默认阈值 0，即不拒识） |
| `top_k` | `list[dict]` | 前 K 个候选及置信度 |
| `source` | `str` | 输入来源路径 |
| `num_frames` / `aggregation` / `frame_agreement` / `frame_votes` | — | 仅视频/多帧结果包含 |

实测加载信息（可用来自检）：

```
模型已加载：best.pt | 主干=resnet18 | 类别=6 | 模式=feature | 设备=cpu
label_map = {'movie_01': 0, 'movie_02': 1, 'movie_03': 2, 'movie_04': 3, 'movie_05': 4, 'movie_06': 5}
```

### 方式三：HTTP 接口（对外提供服务）

项目已内置基于 Python 标准库的推理服务，无需安装 FastAPI 等新依赖。

启动：

```bash
cd /home/kylin/Workarea/CineEye
TMPDIR=$PWD/.tmp ./.venv/bin/python scripts/06_serve.py --host 0.0.0.0 --port 8080
```

> 生产环境要保活，请用 systemd / supervisor / `nohup`，细节见 `docs/API_GUIDE.md`。

调用示例：

```bash
# 健康检查
curl --noproxy '*' http://127.0.0.1:8080/health

# 类别列表
curl --noproxy '*' http://127.0.0.1:8080/labels

# 单图识别（本地路径）
curl --noproxy '*' -X POST http://127.0.0.1:8080/predict \
  -H 'Content-Type: application/json' \
  -d '{"image":"data/frames/movie_01/seg_01/000001.jpg"}'

# 视频识别
curl --noproxy '*' -X POST http://127.0.0.1:8080/predict/video \
  -H 'Content-Type: application/json' \
  -d '{"video":"/home/kylin/movie_train/杏林春暖.1992.mp4","num_frames":16,"aggregation":"mean_prob"}'
```

接口完整说明（含 base64、批量、错误码、CORS、部署注意）见 **`docs/API_GUIDE.md`**。

---

## 6. 把模型部署到另一台机器

### 6.1 最小可运行清单

只需带这几样，不需要训练数据：

```
项目代码/                      # 整个 CineEye 目录（或至少 cineeye/ 与 configs/）
outputs/checkpoints/best.pt    # 主模型
.venv/                         # 已装好依赖的环境；或在新机按 requirements.txt 安装
```

推理不需要 `data/frames/`、`data/videos/`、特征缓存、报告文件。

### 6.2 新机器安装步骤

CPU 机器（或只需 CPU 推理）：

```bash
# 1) 安装依赖（CPU 版 torch）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 2) 确认环境
python scripts/00_check_env.py

# 3) 指定模型路径推理（若模型不在默认位置）
python scripts/05_predict.py --image /path/to/frame.jpg \
  --checkpoint /path/to/best.pt
```

NVIDIA GPU 机器（如 RTX 5070 12G）：装 CUDA 版 wheel 即可，RTX 50 系需 **cu128 + torch≥2.7**：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python scripts/00_check_env.py     # 确认 GPU 型号/显存/算力
```

checkpoint 是纯权重、跨设备通用，无需改模型：`Predictor` 用 `device: auto` 自动检测 CUDA 并切换。也可强制：

```python
predictor = Predictor(device="cuda")   # 或 "auto" / "cpu" / "cuda:1"
```

完整 GPU 安装/调优/排错见 **[GPU 使用指南](GPU_GUIDE.md)**。

### 6.3 迁移时的三个注意点

1. **路径**：`--image` / `--video` 传绝对路径最稳妥；相对路径会按项目根目录解析。
2. **依赖版本**：`torch`、`torchvision`、`opencv-python-headless<5` 要与训练环境兼容；推理代码有旧版 `torch.load` 的兼容处理。
3. **临时目录**：本机 `/tmp` 只有 10 MB，所有命令都加了 `TMPDIR=$PWD/.tmp`；换机器后若 `/tmp` 正常可省略。

---

## 7. 常用参数与调优建议

| 想调整什么 | 怎么调 |
|---|---|
| 视频识别更稳 | 增大 `--num-frames`（16 → 32），或改用 `aggregation=log_prob` |
| 单帧总被误判 | 用视频级聚合；单帧模型受相似镜头影响较大（测试集帧级 0.858，视频级 1.000） |
| 低置信度想拒识 | `--set inference.unknown_threshold=0.5`，结果中 `unknown=true` |
| 只要 Top-1 | `--set inference.top_k=1` |
| 换用另一个模型 | `--checkpoint 路径`，或 Python 里 `Predictor(checkpoint="路径")` |
| 强制 CPU/GPU | Python 里 `Predictor(device="cpu")` / `device="cuda"` |

---

## 8. 常见问题

**Q：提示"找不到模型：outputs/checkpoints/best.pt"？**
A：说明工作目录不对。先 `cd /home/kylin/Workarea/CineEye`，或用 `--checkpoint` 传绝对路径。

**Q：能不能只拷走 `best.pt`，不要整个项目？**
A：仅靠 `.pt` 文件无法直接推理，还需要 `cineeye/` 的加载代码和依赖环境。但**不需要**训练数据、特征缓存和其他报告文件。

**Q：识别结果为什么是 `movie_04` 而不是片名？**
A：刻意使用中性编号。对照表在 `outputs/reports/movie_id_map.json`，对外接口建议在服务端翻译。

**Q：置信度阈值设多少合适？**
A：当前帧级测试 top1 为 0.8583，正确样本置信度多在 0.6 以上。可先试 `0.5`，再按你的业务误报容忍度调整。

**Q：为什么推理不需要联网下载预训练权重？**
A：`Predictor` 用 `pretrained=False` 构建网络再从 checkpoint 加载全部权重；ResNet18 权重已固化在 `best.pt` 里。

**Q：想重新训练出一个新模型？**
A：见 `docs/TRAINING_GUIDE.md`。新模型会覆盖 `outputs/checkpoints/best.pt`，建议先备份旧文件。

---

## 9. 相关文档

- `docs/GPU_GUIDE.md` — NVIDIA GPU 安装/训练/推理/调优（RTX 5070 等）
- `docs/EXPANSION_GUIDE.md` — 扩展功能：海报/演员照/花絮检索、多片名、演员档案
- `README.md` — 项目总览与配置说明
- `docs/TRAINING_GUIDE.md` — 从视频到模型的完整训练与验证流程
- `docs/API_GUIDE.md` — HTTP 接口的请求/响应格式、错误码与部署
