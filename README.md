# 影瞳 CineEye

> 用画面认出电影 —— 基于深度学习监督学习的「视频帧 → 电影」识别框架。

影瞳是一个面向 **图片 / 视频识别电影** 的轻量级研究与工程框架。它把视频按时间抽成帧，用 **监督学习** 训练一个「帧 → 电影类别」分类器，再对整段视频做多帧聚合投票，输出影片名与置信度。项目同时保留 **scikit-learn 传统特征基线** 作为对照标尺，用数据回答「深度特征到底有没有带来提升」。

项目在 **纯 CPU 环境** 做了专门优化（本机无独显也能开箱即用、离线训练），同时**完整支持 NVIDIA GPU**：默认 `device: auto` 自动探测，在任何脚本加 `--device cuda` 即可用显卡，训练/推理/图库/演员识别全链路通用。

> **进阶文档**
> - **想要带界面的操作：用户前台识别 + 管理后台上传/打标/触发训练** → [Web 界面指南](docs/WEB_GUIDE.md) ｜ [完整设计](docs/WEB_UI_DESIGN.md)
> - 用 NVIDIA GPU（如 RTX 5070）加速训练与推理 → [GPU 使用指南](docs/GPU_GUIDE.md)
> - 想识别海报/演员照/花絮，返回多片名与演员档案 → [扩展功能使用指南](docs/EXPANSION_GUIDE.md) ｜ [架构设计](docs/EXPANSION_DESIGN.md)
> - **想上传演员照片、打标签训练、让系统认出演员** → [演员识别指南](docs/ACTOR_GUIDE.md)
> - 想知道模型文件在哪里、怎么加载和调用 → [模型文件与使用指南](docs/MODEL_GUIDE.md)
> - 想自己复现训练、换数据重训、验证模型好坏 → [训练与验证实操指南](docs/TRAINING_GUIDE.md)
> - 想把模型暴露成 HTTP 接口供外部系统调用 → [推理接口文档](docs/API_GUIDE.md)

---

## 目录

- [1. 为什么这样选型](#1-为什么这样选型)
- [2. 运行环境要求](#2-运行环境要求)
- [3. 安装](#3-安装)
- [4. 快速开始](#4-快速开始)
- [5. 数据组织](#5-数据组织)
- [6. 完整工作流](#6-完整工作流)
- [7. 配置系统](#7-配置系统)
- [8. 核心设计](#8-核心设计)
- [9. 输出产物](#9-输出产物)
- [10. 项目结构](#10-项目结构)
- [11. 推理 API 与接入](#11-推理-api-与接入)
- [12. 测试与验证](#12-测试与验证)
- [13. 常见问题](#13-常见问题)
- [14. 面向真实数据的建议](#14-面向真实数据的建议)
- [15. 当前边界与路线图](#15-当前边界与路线图)

---

## 1. 为什么这样选型

### 主框架：PyTorch

电影帧是高维图像，手工特征（颜色直方图、HOG）会丢失大量语义信息。因此选用 **PyTorch** 作为主框架，用 **ImageNet 预训练主干网络做迁移学习**。

关键取舍是训练模式：

| 模式 | 做法 | CPU 开销 | 适用 |
|---|---|---|---|
| `feature`（默认） | 冻结主干网络，只提取深度特征并训练一个轻量分类头 | 与 sklearn 基线同一量级 | **本机默认**，数据探索、快速迭代 |
| `finetune` | 端到端微调全部参数 | 极慢 | 有 NVIDIA GPU 时，或小数据后期精调 |

`feature` 模式是本项目的核心设计：CNN 主干只前向传播一次，特征被缓存到磁盘；后续每次调参只训练线性/浅层分类头，秒级完成。这让纯 CPU 机器也能高效做深度学习实验。

### 对照标尺：scikit-learn

训练时会同时跑一条 `HOG + 颜色直方图 + LogisticRegression` 基线（也支持 SVM / 随机森林），用于验证深度特征是否真的优于手工特征。它不是主模型，而是一面「照妖镜」。

### 硬件实测结论（开发机）

| 项目 | 实测 | 判断 |
|---|---|---|
| CPU | Intel Core Ultra 5 125H，18 逻辑核 | 够用 |
| 内存 / 磁盘 | 30 GiB（可用约 21 GiB）/ 数百 GiB 可用 | 充足 |
| GPU | 仅 Intel Arc 核显，无 `nvidia-smi`、无 `/dev/dri` | 本机走 CPU |

**结论：本机走 CPU 迁移学习路线；代码完整支持 NVIDIA GPU。** 框架通过 `device: auto` 自动探测设备——接上 NVIDIA 显卡（如 RTX 5070 12G）并装好 CUDA 版 torch 后会自动切 CUDA，可开启 `finetune + AMP` 大幅加速。任意脚本加 `--device cuda` 即可显式启用，详见 [GPU 使用指南](docs/GPU_GUIDE.md)。

---

## 2. 运行环境要求

- **操作系统**：Linux（已在 WSL2 / 容器环境验证）；macOS / Windows 理论可用（GPU 版 torch 在 Windows 同样支持）。
- **Python**：3.10+（本项目在 **Python 3.13** 上验证通过）。
- **依赖**：见 `requirements.txt`，核心为 `torch / torchvision / scikit-learn / opencv-python-headless<5 / pandas / numpy / PyYAML / Pillow` 等。
- **GPU（可选，强烈推荐）**：任意 NVIDIA 显卡可显著加速；RTX 30/40 系用 cu128 wheel，RTX 50 系（如 5070 / sm_120）需 **cu128 + torch≥2.7**。安装与验证见 [GPU 使用指南](docs/GPU_GUIDE.md)。无 GPU 时自动走 CPU，功能不受影响。

> 注意：`opencv-python-headless` 必须 `< 5`。OpenCV 5 移除了 `HOGDescriptor`，会导致 sklearn 基线无法导入（代码已加纯 NumPy HOG 兜底，但仍建议锁 `< 5`）。

运行环境自检脚本可随时回答「这台机器能不能跑」：

```bash
./.venv/bin/python scripts/00_check_env.py            # 人类可读输出
./.venv/bin/python scripts/00_check_env.py --json     # JSON 输出，便于脚本消费
```

它检查依赖版本、CPU / 内存 / 磁盘 / 共享内存、GPU 后端、预训练权重缓存，并给出针对本机硬件的推荐配置，**不修改任何文件、不下载模型**。

---

## 3. 安装

### 3.1 使用项目自带虚拟环境（推荐，已装齐）

项目根目录已有 `.venv`，依赖与 CPU 版 PyTorch 均已安装，可直接跳到 [快速开始](#4-快速开始)。

```bash
cd /home/kylin/Workarea/CineEye
./.venv/bin/python scripts/00_check_env.py   # 看到 ready=True 即可
```

### 3.2 从零创建环境

```bash
cd /home/kylin/Workarea/CineEye
python3.13 -m venv .venv
source .venv/bin/activate
```

**国内网络（实测官方源约 30 kB/s，阿里镜像 1~2 MB/s，强烈推荐）：**

```bash
# 若系统 /tmp 很小（如仅 10MB），先把临时目录指到数据盘，否则安装大 wheel 会 ENOSPC
export TMPDIR=$PWD/.tmp/pip-build PIP_CACHE_DIR=$PWD/.cache/pip
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"

pip install -i https://mirrors.aliyun.com/pypi/simple \
    numpy pandas Pillow PyYAML tqdm scikit-learn "opencv-python-headless>=4.9,<5" \
    matplotlib seaborn joblib pytest
pip install --index-url https://mirrors.aliyun.com/pytorch-wheels/cpu/ \
    --extra-index-url https://mirrors.aliyun.com/pypi/simple \
    "torch==2.9.1+cpu" "torchvision==0.24.1+cpu"
```

**境外网络：**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**NVIDIA GPU 台式机（如 RTX 5070 12G）：** 不要装 CPU 版，改用 CUDA 版 wheel。
RTX 50 系（Blackwell sm_120）必须 `cu128 + torch≥2.7`，否则报 `no kernel image`：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python scripts/00_check_env.py        # 确认 GPU 型号/显存/算力与 torch 匹配
```

完整说明（含 BF16、多卡、OOM 调优、错误排查）见 **[GPU 使用指南](docs/GPU_GUIDE.md)**。

### 3.3 离线预训练权重

`model.pretrained=true` 是 CPU 场景下效果的关键。项目已把 `resnet18-f37072fd.pth` 预置在 `.cache/torch/hub/checkpoints/`，**离线也能训练**。若需其他主干，把对应的 `.pth` 放进该目录即可，包导入时会自动改用本地权重，避免联网下载。

---

## 4. 快速开始

### 一键跑通（用内置合成数据）

```bash
./.venv/bin/python scripts/run_all.py --epochs 20
```

这条命令会依次执行：生成合成视频 → 抽帧 → 构建标注清单 → 训练 → 评估 → 单图/视频推理演示。约一两分钟内完成，产物在 `outputs/`。

### 用你自己的数据跑

```bash
# 把视频按 data/videos/<电影名>/<片段>.mp4 放好（见第 5 节）
./.venv/bin/python scripts/run_all.py --skip-demo
```

### 识别一张图 / 一段视频

```bash
./.venv/bin/python scripts/05_predict.py --image data/frames/星际回响/clip_01/000010.jpg
./.venv/bin/python scripts/05_predict.py --video data/videos/星际回响/clip_03.mp4
./.venv/bin/python scripts/05_predict.py --video 某片.mp4 --json    # 供程序消费
```

示例输出：

```
来源: data/videos/星际回响/clip_03.mp4
识别结果: 星际回响   置信度: 0.9303
聚合方式: mean_prob | 采样帧数: 16 | 帧一致性: 100.00%
Top-K: 星际回响 0.930, 雨夜列车 0.041, 赤色荒漠 0.019, ...
```

---

## 5. 数据组织

### 5.1 标签从目录名自动生成，无需手工标注

```text
data/
├── videos/                         # 原始视频（按电影名分目录）
│   ├── 星际回响/
│   │   ├── clip_01.mp4             # 一部电影建议 ≥3 段不同片段
│   │   ├── clip_02.mp4
│   │   └── clip_03.mkv
│   └── 雨夜列车/
│       └── clip_01.mp4
└── frames/                         # 抽帧结果（自动生成）
    └── 星际回响/
        └── clip_01/
            ├── 000000.jpg
            ├── 000001.jpg
            └── ...
```

- **类别（label）** = 电影名 = `videos/` 下的目录名。
- **video_id** = `<电影名>/<片段名>`，是防止数据泄漏的关键分组单位。
- 支持扩展名：`.mp4 .mkv .avi .mov .webm .ts .m4v`（见 `extract.video_extensions`）。

### 5.2 标注清单 `data/manifest.csv`

构建后每一行是一张帧，字段如下：

| 字段 | 说明 |
|---|---|
| `frame_path` | 帧图片路径（相对项目根） |
| `label` | 电影名（文本标签） |
| `video_id` | 所属视频分组，如 `星际回响/clip_03` |
| `frame_index` | 帧序号 |
| `label_idx` | 类别数字编号 |
| `split` | `train` / `val` / `test` |

### 5.3 合成演示数据（无真实片源时）

```bash
./.venv/bin/python scripts/00_make_demo_data.py                       # 默认 4 部 × 5 段
./.venv/bin/python scripts/00_make_demo_data.py --movies 6 --clips 6 --frames 100
```

每部「电影」有固定的色调与几何母题，类别可分、帧间有差异，且各 split 有独立片段。

> 警告：合成数据的识别指标只证明链路正确，**不代表真实电影识别精度**。

---

## 6. 完整工作流

五个脚本可单独执行，也可由 `run_all.py` 一键串联。所有脚本都支持公共参数 `--config` 与可重复的 `--set a.b.c=value`。

### 步骤 0 · 环境自检

```bash
./.venv/bin/python scripts/00_check_env.py
```

### 步骤 1 · 视频抽帧

```bash
./.venv/bin/python scripts/01_extract_frames.py
# 自定义：每秒 2 帧、单视频最多 400 帧
./.venv/bin/python scripts/01_extract_frames.py --set extract.fps=2 --set extract.max_frames=400
```

按时间戳采样（兼容可变帧率），自动跳过黑屏帧与相似帧（静态镜头）。

### 步骤 2 · 构建标注清单

```bash
./.venv/bin/python scripts/02_build_manifest.py
```

**按 `video_id` 分组划分 train/val/test**，保证同一段视频的帧绝不会同时出现在训练集和验证/测试集，避免准确率虚高。构建时会做泄漏自检。

### 步骤 3 · 训练

```bash
./.venv/bin/python scripts/03_train.py                                  # 默认 feature 模式
./.venv/bin/python scripts/03_train.py --mode finetune --set train.epochs=5   # 端到端微调（CPU 很慢）
./.venv/bin/python scripts/03_train.py --set model.backbone=mobilenet_v3_small
./.venv/bin/python scripts/03_train.py --set train.epochs=10 --set model.dropout=0.3
```

训练含类别加权 / 重采样、label smoothing、早停、学习率调度、最佳权重回滚，并同时跑 sklearn 基线。耗时主要在第一次提特征（后续命中缓存则秒级）。

### 步骤 4 · 评估

```bash
./.venv/bin/python scripts/04_evaluate.py                  # 默认 test 集
./.venv/bin/python scripts/04_evaluate.py --split val
./.venv/bin/python scripts/04_evaluate.py --checkpoint outputs/checkpoints/best.pt
```

同时给出**帧级**与**视频级**的 top1 / top5 / macro-F1 / balanced-accuracy，并产出混淆矩阵（PNG + CSV）、分类报告与逐帧预测明细。

### 步骤 5 · 推理

```bash
./.venv/bin/python scripts/05_predict.py --image a.jpg b.jpg            # 多张图片
./.venv/bin/python scripts/05_predict.py --video clip.mp4 \
    --set inference.num_frames=32 --set inference.aggregation=majority
./.venv/bin/python scripts/05_predict.py --video clip.mp4 \
    --set inference.unknown_threshold=0.5                               # 启用低置信度拒识
```

### 一键串联

```bash
./.venv/bin/python scripts/run_all.py --epochs 20
./.venv/bin/python scripts/run_all.py --skip-demo --set model.backbone=resnet18
```

`--set` 的覆盖会透传给每个子步骤，因此一条命令即可统一改超参。

---

## 7. 配置系统

所有超参集中在 `configs/default.yaml`，通过 `--set a.b.c=value` 覆盖任意一项（可重复）。配置会被快照保存，便于复现实验。

### 数据 `data.*`

| 键 | 默认 | 说明 |
|---|---|---|
| `root` / `videos_dir` / `frames_dir` | `data` / `data/videos` / `data/frames` | 数据根目录 |
| `manifest` / `label_map` | `data/manifest.csv` / `outputs/label_map.json` | 清单与类别映射 |
| `image_size` | `224` | 模型输入分辨率 |
| `split.train/val/test` | `0.70 / 0.15 / 0.15` | 划分比例 |
| `split.group_by` | `video_id` | 分组字段（防泄漏，**勿改**） |
| `augmentation.*` | 见文件 | 增强策略；默认关闭水平翻转（避免镜像字幕） |

### 抽帧 `extract.*`

| 键 | 默认 | 说明 |
|---|---|---|
| `fps` | `1.0` | 每秒采样帧数 |
| `max_frames` | `200` | 单视频最多帧数 |
| `resize_long_edge` | `360` | 落盘时长边缩放，`0` 为原尺寸 |
| `jpeg_quality` | `90` | JPEG 质量 |
| `skip_similar` / `skip_similar_threshold` | `true` / `0.985` | 跳过相似帧及其相关系数阈值 |
| `skip_black` / `black_mean_threshold` | `true` / `12.0` | 跳过黑屏帧及其亮度阈值 |

### 模型与训练 `model.*` / `train.*`

| 键 | 默认 | 说明 |
|---|---|---|
| `model.backbone` | `resnet18` | 可选 `resnet18/34/50`、`mobilenet_v3_small`、`efficientnet_b0` |
| `model.pretrained` | `true` | ImageNet 预训练（CPU 效果关键） |
| `model.mode` | `feature` | `feature` / `finetune` |
| `model.dropout` / `model.head_hidden` | `0.2` / `0` | 分类头结构 |
| `train.device` | `auto` | `auto/cpu/cuda/cuda:0/mps/xpu`，自动探测；CLI `--device` 可覆盖（[GPU 指南](docs/GPU_GUIDE.md)） |
| `train.epochs` / `batch_size` / `lr` | `60` / `64` / `0.001` | 训练超参；GPU 显存充足时 `batch_size` 可提到 128~256 |
| `train.num_workers` | `0` | DataLoader 进程数（`/dev/shm` 小时必须为 0，代码会自动降级）；物理机 GPU 可设 2~4 |
| `train.cpu_threads` | `8` | torch CPU 线程数（仅 CPU 模式生效） |
| `train.patience` / `min_delta` | `10` / `0.0005` | 早停 |
| `train.balanced_sampler` / `label_smoothing` | `true` / `0.05` | 类别均衡与正则 |
| `train.amp` / `amp_dtype` | `false` / `fp16` | 混合精度（仅 CUDA）：`fp16`（通用）/ `bf16`（RTX 30/40/50 系更稳） |
| `train.grad_clip` | `5.0` | 梯度裁剪 |

### 推理 `inference.*`

| 键 | 默认 | 说明 |
|---|---|---|
| `num_frames` | `16` | 视频均匀采样帧数 |
| `aggregation` | `mean_prob` | `mean_prob` / `log_prob` / `majority` |
| `top_k` | `5` | 返回 Top-K |
| `unknown_threshold` | `0.0` | `>0` 启用拒识：最高概率低于阈值输出「未知影片」 |
| `checkpoint` | `outputs/checkpoints/best.pt` | 权重路径 |

### 特征缓存与基线 `features.*` / `classical.*`

`features.cache_dir=outputs/features`；`classical.enabled=true` 时训练会附带 sklearn 基线，`estimator` 可选 `logreg/svm/rf`。

---

## 8. 核心设计

- **防数据泄漏**：按 `video_id` 分组划分数据集。相邻帧来自同一镜头、高度相似，若随机按帧划分会让验证集严重虚高。构建清单时会自动做泄漏自检。
- **冻结主干 + 特征缓存**（CPU 友好的关键）：`feature` 模式下主干只前向传播一次，特征按清单内容签名缓存到 `outputs/features/`；重复实验直接命中缓存，无需重跑 CNN。
- **三 split 共享主干**：train/val/test 的特征由同一个主干网络提取，保证分布一致。
- **视频级聚合**：单帧可能恰好是黑屏或通用镜头，对整段视频均匀采样多帧、逐帧预测后聚合（平均概率 / 对数概率 / 多数投票），显著优于单帧。
- **未知影片拒识**：`unknown_threshold > 0` 时，若最高概率低于阈值则输出「未知影片」，避免闭集模型强行归类。
- **传统基线对照**：每条训练都附一条 HOG + 颜色直方图基线，量化深度特征相对手工特征的提升。
- **设备自适应**：`device: auto` 自动探测 CUDA → MPS → XPU → CPU，并有 CUDA 内核级预检（防止 RTX 50 系装错 wheel 时"检测到 GPU 却一跑就崩"）；训练/评估/推理/图库/演员识别全链路共用同一设备选择，见 [GPU 使用指南](docs/GPU_GUIDE.md)。
- **环境自适应**：包导入时自动把临时目录指向项目 `.tmp`（规避 `/tmp` 仅 10MB 的问题）；DataLoader 检测到 `/dev/shm` 不足会自动降为单进程（规避 Bus error）；中文字体自动选择 Noto Sans CJK。

---

## 9. 输出产物

```text
outputs/
├── label_map.json                          # 电影名 → 类别编号
├── checkpoints/
│   ├── best.pt                             # 最佳深度模型权重（含 label_map / 配置）
│   └── classical_baseline.joblib           # sklearn 基线模型
├── features/
│   ├── deep_{train,val,test}.npz           # 缓存的深度特征
│   ├── deep_{train,val,test}.meta.json     # 特征缓存签名
│   └── classical_{train,val}.npz           # 缓存的手工特征
├── reports/
│   ├── train_report.json                   # 训练摘要（模式/设备/指标/基线对比）
│   ├── history.csv                         # 每轮 loss / top1 / top5 / lr
│   ├── eval_test.json                      # 帧级 + 视频级 + 分类报告
│   ├── confusion_matrix_test.png/.csv      # 混淆矩阵
│   ├── classification_report_test.txt      # 各类精确率/召回率/F1
│   ├── predictions_test.csv                # 逐帧预测明细
│   └── movie_id_map.json                   # 中性编号 → 原始影片文件名（仅本机）
├── cineeye.db                              # 扩展：电影档案 SQLite 库（电影/别名/演员/图片索引）
├── gallery/                                # 扩展：开放集图库向量索引
│   ├── embeddings.npy                      # (N, 512) L2 归一化向量
│   ├── meta.json                           # 每行的 movie_id / kind / path / source
│   └── index.meta.json                     # 维度、样本数、编码器指纹、更新时间
├── metadata/
│   └── movies.csv.example                  # 元数据导入模板（复制后填写真实片名/演员）
└── actors/                                 # 扩展：演员参考图库（放入 <演员名>/*.jpg 后构建）
```

其中 `best.pt` 是一个字典，至少包含模型状态、`label_map`、`mode`、`feature_dim`、`num_classes` 等，被评估与推理模块直接复用。扩展产物（`cineeye.db`、`gallery/`、`actors/`）由 `scripts/11~13` 生成，与训练产物相互独立。

---

## 10. 项目结构

```text
CineEye/
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml                # 全部超参
├── cineeye/                        # 核心包
│   ├── config.py                   # YAML 加载、点路径读写、配置快照
│   ├── assets.py                   # 素材资产层：五层标签、多电影/演员关联、派生写回
│   ├── jobs.py                     # 异步训练任务：线程池 + 子进程 + 进度轮询
│   ├── device.py                   # 设备探测（cuda/xpu/mps/cpu）与线程设置
│   ├── utils.py                    # 日志、随机种子、JSON、文件签名、计时
│   ├── cli.py                      # 公共命令行参数（--config / --set）
│   ├── data/
│   │   ├── extract_frames.py       # 视频抽帧、黑屏/相似帧过滤、推理采样
│   │   ├── manifest.py             # 扫描帧、按 video_id 分组划分、泄漏自检
│   │   └── dataset.py              # FrameDataset、数据增强、DataLoader 自适应
│   ├── models/
│   │   ├── cnn.py                  # 5 种预训练主干 + 分类头（feature/finetune）
│   │   └── classical.py            # HOG + 颜色直方图 + sklearn 基线
│   ├── features.py                 # 深度特征提取与缓存（内容签名判定）
│   ├── train.py                    # feature / finetune 双模式训练循环
│   ├── evaluate.py                 # 帧级/视频级评估、混淆矩阵、报告
│   ├── predict.py                  # 单图/视频推理、多帧聚合、拒识（封闭集）
│   ├── serve.py                    # HTTP 推理服务（基础 + 扩展接口）
│   ├── identify.py                 # 统一识别：检索 + 分类器融合 + 元数据丰富
│   ├── gallery/                    # 开放集图库：向量编码、增量索引、类型过滤
│   │   ├── encoder.py              # ImageEncoder：复用 best.pt 主干提 512 维向量
│   │   └── index.py                # GalleryIndex：余弦检索、多图聚合、增量构建
│   ├── metadata/                   # 电影档案元数据库（SQLite）
│   │   ├── store.py                # MetaStore：电影/别名/演员/图片增删改查
│   │   ├── normalize.py            # 片名归一化（繁简/大小写/标点）
│   │   └── providers/              # 可插拔数据源：local_table / tmdb / douban
│   └── actors/                     # 演员识别（照片打标签 → 训练 → 识别）
│       ├── face.py                 # Haar 人脸检测与裁剪
│       ├── dataset.py              # 照片导入/打标签/去重/分层划分/manifest
│       ├── encoder.py              # ActorEncoder：ImageNet 或 best.pt 主干提特征
│       ├── train.py                # 特征缓存 + 线性分类头训练/评估/阈值校准
│       └── recognizer.py           # 分类头+检索融合、多脸识别、unknown 拒识
├── scripts/
│   ├── 00_check_env.py             # 环境自检
│   ├── 00_make_demo_data.py        # 合成演示数据
│   ├── 01_extract_frames.py        # 步骤 1
│   ├── 02_build_manifest.py        # 步骤 2
│   ├── 03_train.py                 # 步骤 3
│   ├── 04_evaluate.py              # 步骤 4
│   ├── 05_predict.py               # 步骤 5
│   ├── 06_serve.py                 # 启动 HTTP 服务
│   ├── 10_prepare_real_movies.py   # 真实长片切段抽帧
│   ├── 11_import_metadata.py       # 导入电影元数据（4 种来源）
│   ├── 12_build_gallery.py         # 构建/更新图库索引
│   ├── 13_actor_reference.py       # 演员参考图库管理
│   ├── 14_identify.py              # 统一电影档案识别（--actor-image 可只认演员）
│   ├── 15_import_actors.py         # 演员照片导入与打标签（目录/CSV）
│   ├── 16_train_actors.py          # 训练演员识别模型（分类头+检索库）
│   ├── run_all.py                  # 一键串联
├── web/                            # React + Vite 前端源码（用户前台 / 管理后台）
│   └── src/
│       ├── UserApp.jsx             # 前台：上传识别 → 结果卡片
│       ├── AdminApp.jsx            # 后台壳：侧边导航
│       ├── pages/                  # 素材库/上传/训练中心/状态中心
│       └── components/             # TagForm（按标签契约动态生成表单）
├── static/                         # 前端构建产物（npm run build 生成，随项目走）
├── configs/tag_schema.json         # 五层标签契约（前后端共用，一处定义）
├── tests/
│   ├── conftest.py
│   ├── test_pipeline.py            # 6 个基础流程测试
│   ├── test_expansion.py           # 6 个扩展模块测试
│   ├── test_actors.py              # 5 个演员识别测试（导入/上传/训练/识别/拒识）
│   └── test_web.py                 # 6 个 Web 资产层测试（契约/校验/去重/派生/任务）
├── data/                           # 视频、帧、清单（gitignore）
└── outputs/                        # 权重、特征、报告、元数据库、图库（gitignore）
```

核心包约 3900 行（含脚本约 4900 行），分层清晰：`data` 负责数据、`models` 负责模型、`features/train/evaluate/predict` 构成封闭集训练推理闭环；`gallery/metadata/actors/identify` 在其之上叠加开放集电影档案检索。

---

## 11. 推理 API 与接入

除命令行外，可直接在 Python 中调用 `Predictor`：

```python
from cineeye.config import load_config
from cineeye.predict import Predictor

cfg = load_config()
predictor = Predictor("outputs/checkpoints/best.pt", cfg=cfg)

# 单张图片
r = predictor.predict_image("data/frames/星际回响/clip_01/000010.jpg")
print(r["label"], r["confidence"], r["top_k"])

# 整段视频（多帧聚合）
r = predictor.predict_video("data/videos/星际回响/clip_03.mp4",
                            num_frames=32, aggregation="mean_prob")
print(r["label"], r["confidence"], r["frame_agreement"])
```

`Predictor` 主要方法：

| 方法 | 用途 |
|---|---|
| `predict_image(path)` / `predict_images(paths)` | 单张 / 批量图片 |
| `predict_video(path, num_frames, aggregation)` | 整段视频均匀采样 + 聚合 |
| `predict_frames(frames, aggregation)` | 直接对内存中的帧列表推理 |
| `predict_proba(tensors)` | 底层批量概率输出 |

返回字典含 `label`、`label_idx`、`confidence`、`unknown`、`top_k`、`source`；视频结果额外含 `aggregation`、`num_frames`、`frame_agreement`。配合命令行的 `--json` 即可无缝接入 Web 服务或批处理系统。

### 开放集识别：Identifier

`Predictor` 只做封闭集帧分类；若要识别海报、演员照、花絮并返回**多片名 / 演员档案**，用 `Identifier`（需先跑 `scripts/11~13` 导入元数据并建库）：

```python
from cineeye.identify import Identifier

ident = Identifier()
r = ident.identify_image("海报.jpg", kind="poster", top_k=3)  # 单图
for m in r["matches"]:
    print(m["movie_id"], m["title"], m["aliases"], m["cast"])
r = ident.identify_video("片段.mp4", num_frames=16)           # 视频
ident.search_movies("Example Movie")                          # 按片名/别名搜索（多语言）
```

`Identifier` 主要方法：`identify_image` / `identify_images`、`identify_video`、`search_movies`、`get_movie`、`list_movies`、`get_actor`、`status`。详见 `docs/EXPANSION_GUIDE.md`，HTTP 接口见 `docs/API_GUIDE.md`。

---

## 12. 测试与验证

```bash
./.venv/bin/python -m pytest tests -q       # 23 passed
```

测试覆盖：数据划分无泄漏、帧提取、配置覆盖、transforms 形状、模型构建、特征缓存（以上基础 6 项），元数据归一化、SQLite 存取、图库检索、演员检测（扩展 6 项），以及演员照片导入去重、CSV 打标签、分类头训练、多脸识别与 unknown 拒识（演员 5 项）。

项目交付前已在合成数据上端到端验证：

- `00_check_env.py`：`ready=True`，device=cpu，8 线程，权重已缓存。
- `run_all.py`：9 段合成视频 → 90 帧 / 3 类，泄漏自检通过；test 帧级与视频级 top1 均 `1.000`，sklearn 基线 `0.800`。
- `feature` 与 `finetune` 两种训练模式均跑通；单图与整段视频推理均命中正确影片。

> 上述高指标仅证明链路正确（合成数据色调/几何差异极大），**不代表真实电影识别精度**。

---

## 13. 常见问题

**Q：安装时提示 `ENOSPC` / `No space left on device`？**
系统 `/tmp` 可能是仅 10MB 的 tmpfs。安装前先 `export TMPDIR=$PWD/.tmp/pip-build PIP_CACHE_DIR=$PWD/.cache/pip`。项目代码导入时也已自动兜底临时目录。

**Q：训练时 DataLoader 报 `Bus error`？**
`/dev/shm` 太小（本机 10MB）。`train.num_workers` 必须为 `0`；`dataset.py` 已检测到共享内存 < 256MB 时自动降级为单进程。物理机内存充足时可设 `2~4`。

**Q：sklearn 基线导入失败 / 找不到 `HOGDescriptor`？**
OpenCV 版本 ≥ 5。降级到 `< 5`：`pip install "opencv-python-headless>=4.9,<5"`。代码也已内置纯 NumPy HOG 兜底。

**Q：没有网络，能用预训练权重吗？**
能。项目已把 `resnet18` 权重放在 `.cache/torch/hub/checkpoints/`。换主干时把对应 `.pth` 放进该目录即可。运行 `00_check_env.py` 可查看已缓存权重。

**Q：CPU 训练太慢怎么办？**
保持 `model.mode=feature`（默认）、`extract.fps=1` 控制帧量、主干选 `resnet18` 或 `mobilenet_v3_small`、命中特征缓存后只训分类头。`finetune` 在纯 CPU 上很慢，不建议；有 NVIDIA GPU 时用 `--device cuda` + `finetune + AMP`（见 [GPU 指南](docs/GPU_GUIDE.md)）。

**Q：有 NVIDIA 显卡，代码怎么切换到 GPU？**
默认 `device: auto` 会自动用 GPU；也可显式 `--device cuda`。RTX 5070 等 50 系需装 cu128 + torch≥2.7，详见 [GPU 使用指南](docs/GPU_GUIDE.md)。图库/演员识别同样自动走 GPU。

**Q：装了 CUDA 版 torch 但还是用 CPU / 报 `no kernel image`？**
多半是 wheel 与 GPU 架构不匹配（尤其 RTX 50 系 sm_120）。跑 `scripts/00_check_env.py` 诊断，按提示重装 cu128 wheel。项目有内核级预检，不匹配时会自动回退 CPU 并给出提示，不会直接崩。

**Q：验证集准确率虚高？**
检查清单是否按 `video_id` 分组划分（`data.split.group_by`），绝不能按帧随机划分。`02_build_manifest.py` 已内置自检。

**Q：界面/图表中文显示成方框？**
评估模块会自动搜索并启用 Noto Sans CJK 等中文字体。若无中文字体，可安装 `fonts-noto-cjk`。

---

## 14. 面向真实数据的建议

- **数据量**：每部电影至少 **3 段不同片段**，每段 ≥ 100 帧；类别间样本量尽量均衡（代码已有类别加权 / 重采样兜底）。
- **片段多样性**：同一部电影选不同场景、不同时间点，避免只取预告片或片头字幕。
- **字幕与水印**：默认关闭水平翻转以避免镜像文字；必要时可调 `augmentation.color_jitter`。
- **调参顺序**：先固定 `feature` 模式跑通基线与深度特征对比，再决定是否换更大主干（`resnet50` / `efficientnet_b0`），最后在有 GPU 时尝试 `finetune`。
- **减少冗余**：真实视频帧间高度相似，`extract.skip_similar=true` 配合较低 `extract.fps` 可显著减少无效帧、加快训练。
- **拒识阈值**：闭集模型对集外影片会强行归类，部署时用 `inference.unknown_threshold` 配合验证集调出合适阈值。

---

## 15. 当前边界与路线图

已实现并验证的部分构成一个**完整可用的框架**（含真实电影训练、HTTP 服务、开放集电影档案检索）。以下为尚未包含、可按需求推进的方向：

- **真实片名 / 演员数据填充**：当前元数据库是中性编号占位档案，真实片名与演员需通过本地表格 / TMDB / 豆瓣显式导入（见 `docs/EXPANSION_GUIDE.md`）。
- 更强的人脸特征：当前演员识别用 ResNet18 通用特征 + 线性分类头（已完成打标签训练闭环）；样本量上来后可替换为 InsightFace / ArcFace（`FaceEncoderStub` 已预留接口）。
- 模型导出（ONNX / TorchScript）与移动端部署。
- 时序聚合模型（如帧序列 LSTM / Transformer）替代当前的概率平均投票。
- 困难样本挖掘、主动学习标注。
- 向量规模上万时启用 faiss、CI 流水线与更完整的单元测试覆盖。

> 已完成：真实 6 部电影端到端训练与评估（帧级 top1 0.858 / 视频级 top1 1.000）、HTTP 接口服务、海报/演员照/花絮检索、多语言片名与演员档案、加新片免重训、演员照片打标签训练与多脸识别（`/actors/recognize`）。

---

## 许可与致谢

本项目为「影瞳 CineEye」学习与工程实践项目。依赖 PyTorch、torchvision、scikit-learn、OpenCV 等开源生态，特此致谢。

> 影瞳 —— 让机器用画面认出电影。
