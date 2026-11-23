# 影瞳 CineEye · 训练与验证实操指南

> 这份指南记录「影瞳」在真实电影数据上从零训练、评估、验证的完整流程，基于本机（纯 CPU）实际跑通的一次训练。
> 按本文顺序执行，你可以完全自己复现、换数据重训、并判断模型好坏。
> 有 NVIDIA GPU 的机器可直接 `--device cuda` 加速（见第 5 节 GPU 训练与 [GPU_GUIDE.md](GPU_GUIDE.md)）。
> 对外提供接口见 [API_GUIDE.md](API_GUIDE.md)。

---

## 目录

- [0. 整体流程一览](#0-整体流程一览)
- [1. 环境与前置检查](#1-环境与前置检查)
- [2. 准备你的视频数据](#2-准备你的视频数据)
- [3. 切段抽帧（关键步骤）](#3-切段抽帧关键步骤)
- [4. 构建标注清单](#4-构建标注清单)
- [5. 训练模型](#5-训练模型)
- [6. 验证与评估](#6-验证与评估)
- [7. 推理验证](#7-推理验证)
- [8. 如何判断模型好坏](#8-如何判断模型好坏)
- [9. 调参与常见问题](#9-调参与常见问题)
- [10. 完整命令速查](#10-完整命令速查)

---

## 0. 整体流程一览

```text
原始视频（每部 1 个大文件）
   │  ① 10_prepare_real_movies.py：按时间切段 + 每段抽帧
   ▼
data/frames/movie_XX/seg_YY/*.jpg        ← 帧级图像 + 中性类别编号
   │  ② 02_build_manifest.py：按 video_id 分组划分 train/val/test
   ▼
data/manifest.csv                        ← 帧级标注清单（含 split）
   │  ③ 03_train.py：ResNet18 迁移学习（CPU feature 模式）
   ▼
outputs/checkpoints/best.pt              ← 训练好的模型
   │  ④ 04_evaluate.py：帧级 + 视频级指标
   │  ⑤ 05_predict.py：单图 / 视频推理
   ▼
对外接口：scripts/06_serve.py（见 API_GUIDE.md）
```

**核心原则**：先切段、后分组、按段划分。这是避免数据泄漏、拿到可信指标的前提（详见第 3、4 节）。

---

## 1. 环境与前置检查

项目根目录已有 `.venv`（依赖齐全），所有命令都在项目根目录执行。

```bash
cd /home/kylin/Workarea/CineEye

# 1) 环境自检：确认 ready=True、依赖齐全、权重已缓存
./.venv/bin/python scripts/00_check_env.py

# 2) （可选）单元测试，确认代码本身没坏
./.venv/bin/python -m pytest tests -q        # 期望 12 passed（基础 6 + 扩展 6）
```

自检若报缺依赖或权重未缓存，参考根目录 `README.md` 第 3 节安装。

> 本机特性：`/tmp` 与 `/dev/shm` 都只有约 10MB。因此所有长耗时命令都带 `TMPDIR=项目/.tmp`，避免临时文件写爆系统盘。下文命令已包含该设置。

---

## 2. 准备你的视频数据

把视频放进一个目录（本文用 `/home/kylin/movie_train/` 举例），**每个视频文件代表一部电影**：

```text
/home/kylin/movie_train/
├── 电影A.mp4
├── 电影B.mkv
└── ...
```

要求：

- 支持 `.mp4 .mkv .avi .mov .webm .ts .m4v`。
- 每部片一个文件即可（哪怕是完整长片）。
- 类别数 ≥ 2（本次是 6 部）。

脚本会按文件名字典序给每部分配中性编号 `movie_01`、`movie_02`…，原始片名只写进映射表 `outputs/reports/movie_id_map.json`，不进入训练产物，便于对接方用编号交流。

> 如果你想用「每部电影多个视频文件」的方式组织，也可以跳过切段脚本，直接按 `data/videos/<电影名>/<片段>.mp4` 放好，再用 `scripts/01_extract_frames.py` 抽帧。但长片场景推荐下面的切段脚本。

---

## 3. 切段抽帧（关键步骤）

### 为什么必须切段

项目按 `video_id` 分组划分数据集：同一段视频的帧要么全在训练集，要么全在验证/测试集。如果每部电影只有 1 个视频文件，就只有 1 个 `video_id`，无法在三个集合都留出独立视频，指标会失真。

解决办法：把每部长片按时间切成 N 段，每段成为一个独立 `video_id`。本实用 8 段 × 40 帧 = 每部 320 帧。

### 执行命令

```bash
TMPDIR=/home/kylin/Workarea/CineEye/.tmp \
./.venv/bin/python scripts/10_prepare_real_movies.py \
    --src /home/kylin/movie_train \
    --segments 8 \
    --frames-per-seg 40
```

参数说明：

| 参数 | 默认 | 含义 | 建议 |
|---|---|---|---|
| `--src` | `/home/kylin/movie_train` | 原始视频目录 | 改成你自己的目录 |
| `--segments` | `8` | 每部分成几段 | ≥3，长片建议 6–10 |
| `--frames-per-seg` | `40` | 每段抽多少帧 | 30–50，越多越慢 |
| `--long-edge` | `256` | 图片落盘长边 | 224–320 足够 |

### 产物与耗时

```text
data/frames/movie_01/seg_00/000001.jpg ...   # 1920 张帧（6 部 × 8 段 × 40）
outputs/reports/movie_id_map.json            # 编号 → 原始片名映射
```

本机实测：6 部长片（50–90 分钟）切段抽帧约 **28 分钟**，主要耗在 ffmpeg 对大文件按时间定位。可用 `tail -f .tmp/prepare.log` 观察进度。

> 脚本会先把旧的演示帧和训练产物移到 `.tmp/backup_demo/` 备份（不删除）。若是想在同一批数据上重跑，加 `--no-backup` 跳过备份。

### 抽帧质量自检

```bash
./.venv/bin/python -c "
import cv2, glob
fs = sorted(glob.glob('data/frames/*/*/*.jpg'))
print('总帧数:', len(fs))
im = cv2.imread(fs[0]); print('样本尺寸:', None if im is None else im.shape)
"
```

确认总数符合预期、图片可正常读取（非 `None`）即可。

---

## 4. 构建标注清单

扫描帧目录，按 `video_id`（即 `movie_XX/seg_YY`）分组划分 train/val/test：

```bash
TMPDIR=/home/kylin/Workarea/CineEye/.tmp \
./.venv/bin/python scripts/02_build_manifest.py
```

期望输出（本次真实结果）：

```text
类别数=6 | 帧数=1920 | 视频数=48
  train: 1440 帧 / 36 段视频 / 6 类
  val  :  240 帧 /  6 段视频 / 6 类
  test :  240 帧 /  6 段视频 / 6 类
数据泄漏自检通过 (OK)
```

检查每类在三集合的分布是否均衡：

```bash
./.venv/bin/python -c "
import pandas as pd
df = pd.read_csv('data/manifest.csv')
print(pd.crosstab(df['label'], df['split']))
"
```

本次结果每类都是 `train 240 / val 40 / test 40`，完全均衡。**如果某一类在 val 或 test 是 0，说明段数太少，回到第 3 节增大 `--segments`。**

> 划分比例在 `configs/default.yaml` 的 `data.split`（默认 0.7/0.15/0.15），一般不用改。

---

## 5. 训练模型

### 默认训练（推荐，CPU 友好）

```bash
TMPDIR=/home/kylin/Workarea/CineEye/.tmp \
./.venv/bin/python scripts/03_train.py --set train.epochs=40
```

这条命令做了两件事：

1. **主模型**：ResNet18 ImageNet 预训练 → 冻结主干提特征 → 训练轻量分类头（feature 模式）。
2. **对照基线**：HOG + 颜色直方图 + 逻辑回归（sklearn），用来量化深度特征是否真的更好。

本次真实结果（约 1 分钟完成）：

```text
深度特征 train: (1440, 512) / val: (240, 512) / test: (240, 512)
val top1=0.7875  top5=1.0000
sklearn 基线 val_top1=0.6000
深度特征 vs sklearn 基线：+0.1875
checkpoint → outputs/checkpoints/best.pt
```

### 关键参数

通过 `--set a.b.c=value` 覆盖，可重复：

| 参数 | 默认 | 说明 |
|---|---|---|
| `train.device` | auto | `auto/cpu/cuda/cuda:N/mps/xpu`；有 NVIDIA GPU 自动用，CLI `--device cuda` 可覆盖 |
| `train.epochs` | 60 | 训练轮数，本次用 40 |
| `model.backbone` | resnet18 | `resnet18/34/50`、`mobilenet_v3_small`、`efficientnet_b0` |
| `model.mode` | feature | `feature`（CPU 快）/ `finetune`（端到端，GPU 推荐、CPU 很慢） |
| `train.lr` | 0.001 | 学习率 |
| `train.batch_size` | 64 | 批大小；GPU 显存充足可提到 128~256 |
| `train.patience` | 10 | 早停轮数 |
| `train.amp` / `train.amp_dtype` | false / fp16 | 混合精度（仅 CUDA）；GPU 微调建议 `amp=true` + `amp_dtype=bf16` |
| `classical.enabled` | true | 是否跑 sklearn 基线 |

示例：换主干、改轮数、关基线：

```bash
./.venv/bin/python scripts/03_train.py \
    --set model.backbone=mobilenet_v3_small \
    --set train.epochs=30 \
    --set classical.enabled=false
```

### 特征缓存机制

`feature` 模式会把主干网络提取的特征缓存到 `outputs/features/`。只要清单内容和主干不变，再次训练会直接命中缓存、跳过提特征，秒级完成。若改了数据或主干，缓存自动失效重提。

### 用 GPU 训练（有 NVIDIA 显卡时）

默认 `device: auto` 检测到 CUDA 就会自动用 GPU。端到端微调在 GPU 上可行且快，配合 AMP 进一步提速：

```bash
./.venv/bin/python scripts/03_train.py \
    --device cuda \
    --set model.mode=finetune \
    --set train.amp=true --set train.amp_dtype=bf16 \
    --set train.batch_size=128 --set train.num_workers=4
```

- RTX 50 系（如 5070，sm_120）需装 **cu128 + torch≥2.7**，否则报 `no kernel image`；
- `amp_dtype=bf16` 在 RTX 30/40/50 系数值更稳、无需 GradScaler；
- 显存不足就调小 `batch_size`；多卡机器用 `--device cuda:1` 指定。

完整安装/调优/排错见 **[GPU 使用指南](GPU_GUIDE.md)**。纯 CPU 请勿用 `finetune`（会很慢）。

---

## 6. 验证与评估

在测试集上评估，输出帧级与视频级两类指标：

```bash
TMPDIR=/home/kylin/Workarea/CineEye/.tmp \
./.venv/bin/python scripts/04_evaluate.py --split test
```

本次真实结果：

```text
帧级  top1=0.8583  top5=1.0000  macro_f1=0.8582  (240 帧)
视频级 top1=1.0000  top5=1.0000  macro_f1=1.0000  (6 段)
```

产出文件（都在 `outputs/reports/`）：

| 文件 | 内容 |
|---|---|
| `eval_test.json` | 帧级 + 视频级全部指标、各类别精确率/召回率/F1 |
| `confusion_matrix_test.png` / `.csv` | 混淆矩阵 |
| `classification_report_test.txt` | 分类报告 |
| `predictions_test.csv` | 逐帧预测明细（真值、预测、置信度、是否正确） |
| `history.csv` | 训练过程的 loss/top1/lr 曲线数据 |

也可以评估验证集：`--split val`。

---

## 7. 推理验证

### 命令行推理

```bash
# 单张图片
TMPDIR=/home/kylin/Workarea/CineEye/.tmp \
./.venv/bin/python scripts/05_predict.py \
    --image data/frames/movie_01/seg_01/000001.jpg

# 整段视频（多帧聚合）
TMPDIR=/home/kylin/Workarea/CineEye/.tmp \
./.venv/bin/python scripts/05_predict.py \
    --video "/home/kylin/movie_train/某片.mp4" \
    --num-frames 16 --aggregation mean_prob

# JSON 输出，便于程序解析
./.venv/bin/python scripts/05_predict.py --video "某片.mp4" --json
```

本次真实结果：单帧识别 `movie_01` 置信度 0.913；整段视频聚合识别正确，置信度 0.625、帧一致率 75%。

### 用 HTTP 接口验证

启动服务后用 curl 调（详见 [API_GUIDE.md](API_GUIDE.md)）：

```bash
./.venv/bin/python scripts/06_serve.py --port 8080
curl --noproxy '*' http://127.0.0.1:8080/health
```

---

## 8. 如何判断模型好坏

按重要性排序看这几个指标：

1. **视频级 top1**：一段视频聚合后的识别准确率。这是最终业务指标（用户通常传整段视频），应优先看它。本次 1.000。
2. **帧级 top1 / macro_f1**：单帧识别能力，反映模型对困难帧的表现。本次 0.858。帧级低于视频级是正常的——单帧可能是转场、黑屏、通用镜头。
3. **深度特征 vs sklearn 基线的差值**：若差值接近 0，说明深度模型没带来价值，需检查数据或换主干。本次 +0.19，健康。
4. **混淆矩阵**：看哪两类容易混。若某两类长期互混，可能是内容本身相似，考虑增加这两类的训练帧或菌选更有区分度的片段。

判断训练是否正常：

- 训练 loss 稳步下降、val loss 趋稳 → 正常。
- val top1 远低于 train top1（如 0.99 vs 0.60）→ 过拟合，减少 epoch 或加大 `model.dropout`。
- val loss 早早反弹、早停触发在很靠前的 epoch → 学习率可能过大，试 `--set train.lr=0.0005`。

---

## 9. 调参与常见问题

### 想进一步提升精度

1. **增数据**：每段多抽帧（`--frames-per-seg 60`）、每部多切段（`--segments 12`）。这是最有效的手段。
2. **换主干**：`--set model.backbone=efficientnet_b0` 或 `resnet50`（CPU 上提特征会更慢，但 feature 模式仍可接受）。
3. **调分类头**：`--set model.head_hidden=128` 增加一层隐藏层。
4. **调正则**：`--set model.dropout=0.3`、`--set train.weight_decay=0.001`。

### 常见问题

| 现象 | 原因 | 解决 |
|---|---|---|
| 某类 val/test 帧数为 0 | 段数太少 | 增大 `--segments`（≥3） |
| `Bus error` / DataLoader 报错 | `/dev/shm` 太小 | 保持 `train.num_workers=0`（默认，代码自动降级） |
| `ENOSPC` / 磁盘满 | `/tmp` 仅 10MB | 所有命令带 `TMPDIR=项目/.tmp` |
| 提特征很慢 | 首次提特征无缓存 | 正常，后续训练命中缓存会秒级 |
| 识别结果都是同一类 | 类别极不均衡或标签搞错 | 检查清单 `pd.crosstab` 分布、确认映射表 |
| 换数据后指标没变 | 命中了旧特征缓存 | 缓存按清单内容签名，换数据会自动失效；若可疑可删 `outputs/features/deep_*` 重训 |

---

## 10. 完整命令速查

```bash
cd /home/kylin/Workarea/CineEye
export TMPDIR=/home/kylin/Workarea/CineEye/.tmp

# ① 环境自检
./.venv/bin/python scripts/00_check_env.py

# ② 切段抽帧（换数据时改 --src）
./.venv/bin/python scripts/10_prepare_real_movies.py \
    --src /home/kylin/movie_train --segments 8 --frames-per-seg 40

# ③ 构建清单
./.venv/bin/python scripts/02_build_manifest.py
./.venv/bin/python -c "import pandas as pd; print(pd.crosstab(pd.read_csv('data/manifest.csv')['label'], pd.read_csv('data/manifest.csv')['split']))"

# ④ 训练
./.venv/bin/python scripts/03_train.py --set train.epochs=40

# ⑤ 评估
./.venv/bin/python scripts/04_evaluate.py --split test

# ⑥ 推理
./.venv/bin/python scripts/05_predict.py --image data/frames/movie_01/seg_01/000001.jpg
./.venv/bin/python scripts/05_predict.py --video "/home/kylin/movie_train/某片.mp4"

# ⑦ 对外提供接口
./.venv/bin/python scripts/06_serve.py --port 8080     # 见 API_GUIDE.md
```

---

写好这份指南后，记得：换数据重训时，编号 `movie_XX` 对应的原始片名见 `outputs/reports/movie_id_map.json`。对外接口返回的是中性编号，对接方需用该映射表翻译成片名（或你在接口层做转换）。
