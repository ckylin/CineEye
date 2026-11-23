# 影瞳 CineEye · GPU（NVIDIA 显卡）使用指南

> 同一份代码在 **CPU 笔记本** 和 **NVIDIA GPU 台式机** 上都能跑，无需改业务代码。
> 默认情况下 `device: auto` 会自动探测：有可用 CUDA GPU 就用 GPU，否则回退 CPU。
> 本文以 **RTX 5070 12GB（Blackwell 架构 / sm_120）** 为例，也适用于 RTX 30/40/50 及其他 NVIDIA 显卡。

---

## 目录

- [1. 一句话结论](#1-一句话结论)
- [2. 安装 CUDA 版 PyTorch（关键）](#2-安装-cuda-版-pytorch关键)
- [3. 验证 GPU 是否真的可用](#3-验证-gpu-是否真的可用)
- [4. 用 GPU 推理](#4-用-gpu-推理)
- [5. 用 GPU 训练](#5-用-gpu-训练)
- [6. 图库 / 演员识别也走 GPU](#6-图库--演员识别也走-gpu)
- [7. 显存与速度调优](#7-显存与速度调优)
- [8. 常见错误排查](#8-常见错误排查)

---

## 1. 一句话结论

1. **代码不用改**：所有训练/评估/推理/图库/演员模块都通过统一的 `resolve_device()` 选设备，`auto` 模式见 GPU 即用。
2. **关键在安装**：RTX 5070 是 Blackwell（sm_120），必须装 **CUDA 12.8 版 wheel（cu128）+ torch ≥ 2.7**，否则报 `no kernel image is available`。
3. **快捷切换**：任何脚本加 `--device cuda`（或 `auto` / `cpu` / `cuda:1`）即可；也可用 `--set train.device=cuda`。
4. **训练提速**：GPU 上建议 `model.mode=finetune` + `train.amp=true` + `train.amp_dtype=bf16`，并把 `batch_size` 调大。

---

## 2. 安装 CUDA 版 PyTorch（关键）

### 2.1 RTX 50 系（含 5070，必须 cu128）

Blackwell 的 sm_120 内核从 **PyTorch 2.7.0** 起才进入稳定版，且需要 **CUDA 12.8** 的 wheel。
先卸载 CPU 版，再从 PyTorch 官方源安装（`--index-url` 而非 `--extra-index-url`，避免 pip 回退到 CPU 版）：

```bash
# Windows PowerShell / Linux 均可（在项目的 .venv 中）
./.venv/bin/python -m pip uninstall -y torch torchvision torchaudio   # Linux/macOS
# Windows: .\.venv\Scripts\python.exe -m pip uninstall -y torch torchvision torchaudio

# 安装 CUDA 12.8 版（wheel 自带 CUDA 运行时，无需本机单独装 CUDA Toolkit）
./.venv/bin/python -m pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu128
```

> 下载约 2.5GB+ 属正常（内含 CUDA 运行时 / cuDNN / NCCL）。国内网络慢时可用镜像版 cu128：
> `https://mirrors.aliyun.com/pytorch-wheels/cu128/`（用 `--index-url` 指向它）。

### 2.2 RTX 30 / 40 系（Ampere / Ada）

cu124 / cu126 / cu128 均可，直接：

```bash
./.venv/bin/python -m pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu128
```

### 2.3 其余依赖

```bash
./.venv/bin/python -m pip install -r requirements.txt
```

> `requirements.txt` 中 `torch>=2.6 / torchvision>=0.21` 只是依赖声明；CUDA 版已装好后 pip 不会覆盖。

---

## 3. 验证 GPU 是否真的可用

```bash
# 项目自检（会显示 GPU 型号、显存、算力、驱动，以及 torch 版本是否匹配 sm_120）
./.venv/bin/python scripts/00_check_env.py

# 直接用 torch 验证内核（RTX 5070 应输出 cuda:0 / (12, 0) / True）
./.venv/bin/python -c "
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device:', torch.cuda.get_device_name(0))
    print('compute cap:', torch.cuda.get_device_capability(0))  # RTX 5070 -> (12, 0)
    x = torch.ones(4,4, device='cuda'); print('kernel ok:', (x@x).sum().item() == 64)
"
```

项目还提供**内核级预检**（`cineeye.device.cuda_kernel_ok()`）：即使 `is_available()` 返回 True，
也会先跑一次真实运算确认内核能执行，避免"检测到 GPU 却一推理就崩"。不匹配时自动回退 CPU 并给出 cu128 安装提示。

---

## 4. 用 GPU 推理

```bash
# 默认 auto：有 GPU 自动用，无需加参数
./.venv/bin/python scripts/05_predict.py --image frame.jpg --device auto

# 显式指定 GPU
./.venv/bin/python scripts/05_predict.py --video clip.mp4 --device cuda

# 多卡机器指定第 2 块
./.venv/bin/python scripts/05_predict.py --image frame.jpg --device cuda:1

# 强制 CPU（对比/排错）
./.venv/bin/python scripts/05_predict.py --image frame.jpg --device cpu
```

启动日志会打印实际设备，例如：

```text
模型已加载：best.pt | 主干=resnet18 | 类别=6 | 模式=feature | 设备=cuda
```

Python API 同样支持：

```python
from cineeye.predict import Predictor
p = Predictor(device="cuda")      # 或 "auto" / "cpu" / torch.device("cuda:0")
r = p.predict_image("frame.jpg")
```

HTTP 服务：

```bash
./.venv/bin/python scripts/06_serve.py --port 8080 --device cuda --with-identifier
# /health 的 data.identifier.device 会显示 cuda，便于确认扩展服务也在 GPU 上
```

---

## 5. 用 GPU 训练

### 5.1 端到端微调（GPU 推荐）

```bash
./.venv/bin/python scripts/03_train.py \
    --device cuda \
    --set model.mode=finetune \
    --set train.amp=true \
    --set train.amp_dtype=bf16 \
    --set train.batch_size=128 \
    --set train.num_workers=4 \
    --set train.epochs=30
```

### 5.2 冻结主干 feature 模式（更快）

```bash
./.venv/bin/python scripts/03_train.py \
    --device cuda \
    --set model.mode=feature \
    --set train.batch_size=256
```

> feature 模式提特征那一步也在 GPU 上跑（CPU 版是秒级缓存，GPU 更快）；分类头训练本身很轻。

### 5.3 AMP 说明

| `train.amp_dtype` | 适用 | 特点 |
|---|---|---|
| `fp16`（默认） | 所有 NVIDIA GPU | 需 GradScaler，兼容性最广 |
| `bf16` | Ampere+（RTX 30/40/50 系、A100/H100） | 数值更稳、无需 GradScaler，**RTX 5070 推荐** |

AMP 仅在 `device` 为 CUDA 且 `train.amp=true` 时生效；CPU 下自动忽略。

---

## 6. 图库 / 演员识别也走 GPU

开放集检索、演员识别原先在 GPU 机器上仍走 CPU（编码器设备写死），现已修复并统一：

- `Identifier` 只解析一次设备，图库编码器、分类器共用同一设备；
- 构建图库、演员参考库时也自动用 GPU；
- HTTP 服务的 `--device` 现在同样作用于扩展服务。

```bash
# 建库（自动用 GPU）
./.venv/bin/python scripts/12_build_gallery.py --rebuild
./.venv/bin/python scripts/13_actor_reference.py --build

# 统一识别（可指定设备）
./.venv/bin/python scripts/14_identify.py --image poster.jpg --kind poster --device cuda

# 确认扩展服务设备
./.venv/bin/python scripts/14_identify.py --status   # 输出中的 device 字段
```

> 图库向量 `.npy` 存的是 CPU numpy 数组，与训练/推理设备无关；换 GPU 后**无需重建索引**即可用 GPU 检索。

---

## 7. 显存与速度调优

| 目标 | 做法 |
|---|---|
| 显存不够（OOM） | 调小 `train.batch_size`（128→64→32）；finetune 比 feature 更吃显存 |
| 想更快 | `train.amp=true` + `amp_dtype=bf16`；`num_workers=2~4`；batch 尽量大 |
| 数据加载慢 | `train.num_workers=4`（物理机 /dev/shm 足够时；容器会自动降级） |
| 多卡机器 | `--device cuda:0` 指定单卡；当前代码单卡推理/训练，未做多卡并行 |
| CPU↔GPU 切换 | 用 `--device` 或 `--set train.device`，不必改配置文件 |

---

## 8. 常见错误排查

| 现象 | 原因 | 解决 |
|---|---|---|
| `CUDA error: no kernel image is available for execution on the device` | torch wheel 不支持该 GPU 架构（RTX 5070=sm_120） | 装 cu128 + torch≥2.7（见第 2 节） |
| `torch.cuda.is_available()` 为 False | 装成了 CPU 版 torch | 卸载后从 `pytorch.org/whl/cu128` 重装；用 `--index-url` |
| 检测到 GPU 但代码仍走 CPU | 内核预检未通过，自动回退 | 看启动日志/__warning__，按提示升级 wheel；或 `00_check_env.py` 诊断 |
| `CUDA out of memory` | batch 太大 / 模型太大 | 调小 `train.batch_size`，feature 模式更省显存 |
| `module 'torch' has no attribute ...` | torch 版本过旧 | 升级到 ≥2.7（RTX 50 系） |
| 图库/演员识别没用上 GPU | 用了旧代码路径 | 确认已更新到本版本；`14_identify.py --status` 看 device |
| `pin_memory` 相关警告 | 无 CUDA 时无害 | 忽略；GPU 下自动开启锁页内存加速 |

> 相关文档：[训练与验证](TRAINING_GUIDE.md) ｜ [模型使用](MODEL_GUIDE.md) ｜ [接口](API_GUIDE.md) ｜ [扩展功能](EXPANSION_GUIDE.md)
