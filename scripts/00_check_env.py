"""影瞳 · 运行环境自检：回答「这台电脑能不能跑深度学习训练」。

用途：在训练前一次性体检——依赖是否装齐、CPU/内存/磁盘是否够、GPU 后端能否真正
执行 CUDA 内核、ImageNet 预训练权重是否已缓存，并按本机硬件给出推荐的训练配置。
不修改任何文件，也不下载模型，可以随时重复执行。

典型调用：
    python scripts/00_check_env.py
    python scripts/00_check_env.py --json

输入/输出：
    读取：/proc/meminfo、/dev/shm、$TMPDIR、nvidia-smi、
          $TORCH_HOME/hub/checkpoints/*.pth（默认 ~/.cache/torch/...）
    输出：仅标准输出（人类可读报告或 --json）；不写任何文件。

运维要点：
    1. 本机 /tmp 与 /dev/shm 只有 10MB —— 报告里的 tmp_free_mb / shm_free_mb
       就是给这个坑用的：装 wheel、ffmpeg 临时文件、DataLoader 多进程都会因此失败。
    2. RTX 50 系（Blackwell，sm_120 / compute_cap 12.0）必须用 cu128 的 torch>=2.7
       wheel，否则会看到 "no kernel image is available"；本脚本会显式提示。
    3. 本机没有 GPU 但代码支持 CUDA：脚本会分别给出 CPU 路线（feature 模式）
       与 GPU 路线（finetune + amp）的建议，改配置前请对照结论。
    4. 「ready=True」只代表依赖齐全且设备可探测，不代表训练一定快，
       数据量与划分方式（按 video_id 分组）仍是效果的决定因素。
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import re
import shutil
import sys
from pathlib import Path

# 脚本被直接 `python scripts/xx.py` 执行时，sys.path[0] 是 scripts/ 而非项目根，
# 必须先手工插入项目根才能 import cineeye（下面的 noqa: E402 就是这个原因）。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cineeye.config import ROOT as PROJECT_ROOT  # noqa: E402

# (import 名, pip 安装名) 对照表；两者不一致的必须写对，否则诊断信息会误导运维
REQUIRED = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("PIL", "Pillow"),
    ("yaml", "PyYAML"),
    ("cv2", "opencv-python-headless"),
    ("sklearn", "scikit-learn"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("matplotlib", "matplotlib"),
    ("joblib", "joblib"),
    ("tqdm", "tqdm"),
    ("pytest", "pytest"),
]


def check_packages() -> dict[str, str]:
    """逐个 import 依赖并取 __version__，缺失或导入失败时记成 "MISSING (异常类型)"。

    用 importlib 实际 import 而不是查 pip 元数据，是为了发现「装了却 import 崩」
    的情况（CUDA 运行库缺失、DLL 冲突），这是 GPU 环境最常见的故障形态。
    任何单个包失败都不抛异常，保证整份体检报告能出完。
    """
    out: dict[str, str] = {}
    for mod, pip_name in REQUIRED:
        try:
            m = importlib.import_module(mod)
            out[pip_name] = str(getattr(m, "__version__", "unknown"))
        except Exception as e:  # noqa: BLE001
            out[pip_name] = f"MISSING ({type(e).__name__})"
    return out


def check_hardware() -> dict:
    """采集 CPU / 内存 / 项目盘 / 临时目录 / /dev/shm 的空间信息。

    这些数值直接决定配置怎么调：num_workers 受 /dev/shm 限制，
    能否安装 wheel 受 $TMPDIR 剩余空间限制（本机 /tmp 只有 10MB）。
    """
    total_gb = None
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                # /proc/meminfo 单位是 kB：先 /1024 得 MB，再 /1024 得 GB
                total_gb = round(int(line.split()[1]) / 1024 / 1024, 1)
                break
    disk = shutil.disk_usage(PROJECT_ROOT)
    # TMPDIR 可能已被上层脚本指向项目 .tmp（本机 /tmp 只有 10MB），这里如实反映
    tmp = Path(os.environ.get("TMPDIR") or "/tmp")
    try:
        tmp_free_mb = round(shutil.disk_usage(tmp).free / 1024 ** 2, 1)
    except OSError:
        # 目录不存在/不可读时不中断体检，记 0 交给 recommend() 提示
        tmp_free_mb = 0.0
    shm = Path("/dev/shm")
    try:
        # 容器里可能没有 /dev/shm：缺省给 9999 表示「不受限」，避免误报 workers 必须为 0
        shm_free_mb = round(shutil.disk_usage(shm).free / 1024 ** 2, 1) if shm.exists() else 9999.0
    except OSError:
        shm_free_mb = 0.0
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_logical": os.cpu_count(),
        "memory_total_gb": total_gb,
        "project_disk_free_gb": round(disk.free / 1024 ** 3, 1),
        "tmp_dir": str(tmp),
        "tmp_free_mb": tmp_free_mb,
        "shm_free_mb": shm_free_mb,
    }


def _nvidia_smi() -> dict:
    """从 nvidia-smi 读取 GPU 信息；无驱动/无命令时安全返回空。

    为什么不直接用 torch.cuda.get_device_name：torch wheel 与架构不匹配时 torch 会直接
    报错，而 nvidia-smi 仍能给出真实型号与 compute_cap，正好用来区分
    「卡坏了」和「卡是好的、wheel 装错了」。
    """
    import subprocess

    out: dict = {}
    try:
        r = subprocess.run(
            # 只取第一块卡；compute_cap 是判断 sm_120（RTX 50 系）的关键字段
            ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8, check=False,
        )
        if r.returncode == 0 and r.stdout.strip():
            first = r.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in first.split(",")]
            if len(parts) >= 4:
                out = {
                    "nvidia_gpu": parts[0],
                    "vram_gb": round(int(float(parts[1])) / 1024, 1) if parts[1] else None,
                    "compute_cap": parts[2],
                    "driver": parts[3],
                }
    except (OSError, ValueError, subprocess.SubprocessError):
        # 没装 nvidia-smi / 输出格式变化 / 超时：一律视为「没有可用 NVIDIA GPU」
        pass
    return out


def check_device() -> dict:
    """探测 torch 可用后端并合入 nvidia-smi 的硬件真值。

    返回 dict 中最关键的两个字段：
      cuda_available  —— torch 认为「能不能用 CUDA」
      cuda_kernel_ok  —— 真正试算过一次矩阵乘，代表「内核能不能在卡上执行」
    很多 GPU 问题只满足前者、不满足后者（no kernel image），所以两者要一起看。
    """
    try:
        import torch

        from cineeye.device import describe_device, resolve_device

        dev = resolve_device("auto", min(8, os.cpu_count() or 4))
        info = describe_device(dev)
        info.update(_nvidia_smi())
        # RTX 50 系（sm_120 / compute_cap 12.0）需 torch>=2.7 + cu128，否则内核报错。
        # 这里只判 wheel 的本地版本号标签：只有 +cu12x 才是 CUDA 12 版轮子；
        # CPU 版（无本地版本号）与 cu121/cu124 等老轮子都会命中下面的告警分支。
        cap = str(info.get("compute_cap") or info.get("cuda_capability") or "")
        if cap.startswith("12."):
            ver = str(info.get("torch", ""))
            # 解析本地版本号里的 CUDA 主版本：+cu128 / +cu129 / +cu130 才算带 sm_120 内核。
            # 注意不能写成 "+cu12" in ver —— 那会把 cu121/cu124 也误判为可用（假阴性）。
            m = re.search(r"\+cu(\d+)", ver)
            cu = int(m.group(1)) if m else 0
            info["sm120_wheel_ok"] = cu >= 128
        return info
    except Exception as e:  # noqa: BLE001
        # torch 本身 import 失败也走这里：把错误文本塞进 dict，由 recommend() 汇总提示
        return {"error": f"{type(e).__name__}: {e}"}


def check_weights() -> dict:
    """检查 torchvision 预训练权重缓存，避免离线机器卡在联网下载。

    TORCH_HOME 优先（部署脚本可改到数据盘），否则默认 ~/.cache/torch。
    只列 *.pth：够用来判断首次训练需不需要联网。
    """
    home = Path(os.environ.get("TORCH_HOME") or (Path.home() / ".cache" / "torch"))
    ckpt_dir = home / "hub" / "checkpoints"
    files = sorted(p.name for p in ckpt_dir.glob("*.pth")) if ckpt_dir.is_dir() else []
    return {"torch_home": str(home), "cached_weights": files}


def recommend(hw: dict, dev: dict, pkgs: dict, weights: dict) -> tuple[bool, list[str]]:
    """根据体检结果生成人话建议，返回 (是否就绪, 建议列表)。

    建议按「缺依赖 → 设备不可用 → GPU 轮子不匹配 → 调参与数据」的顺序排列，
    对应运维排查的先后顺序；这些文案只负责告知，不做自动修复。
    """
    missing = [k for k, v in pkgs.items() if str(v).startswith("MISSING")]
    tips: list[str] = []
    has_gpu = bool(dev.get("cuda_available") or dev.get("xpu_available") or dev.get("mps_available"))

    if missing:
        tips.append("缺少依赖：" + ", ".join(missing) + " → 按 requirements.txt 安装")
    if "error" in dev:
        tips.append("设备探测失败：" + dev["error"])

    if has_gpu:
        gpu_name = dev.get("nvidia_gpu") or dev.get("cuda_device") or "GPU"
        cap = dev.get("compute_cap") or dev.get("cuda_capability") or "?"
        vram = dev.get("vram_gb") or dev.get("cuda_vram_gb")
        tips.append(f"检测到 GPU：{gpu_name}（算力 sm{cap}" +
                    (f" / 显存 {vram}GB" if vram else "") +
                    "）；推理/图库编码将自动跑在 GPU 上")
        if cap.startswith("12.") and dev.get("sm120_wheel_ok") is False:
            # RTX 50 系（Blackwell）专属坑：cu121/cu124 的 wheel 里没有 sm_120 的 kernel 镜像
            tips.append(f"⚠ 当前 torch={dev.get('torch')} 可能不支持 sm_120（RTX 50 系）："
                        "请装 cu128 wheel：pip install torch torchvision "
                        "--index-url https://download.pytorch.org/whl/cu128（需 torch>=2.7），"
                        "否则会出现 no kernel image 错误")
        if dev.get("cuda_kernel_ok") is False and cap:
            # 回退信号：torch 说有 CUDA 但试算失败，几乎必然是 wheel 与 GPU 架构不匹配
            tips.append("⚠ CUDA 设备存在但内核不可用：多半是 torch wheel 与 GPU 架构不匹配，见上一条")
        tips.append("GPU 推荐：model.mode=finetune（端到端微调）+ train.amp=true（混合精度）；"
                    "显存充足可把 train.batch_size 提到 128 以上、train.num_workers 设 2~4")
    else:
        # 无 GPU 也能训练，但必须走 feature 模式：主干只前向一次并缓存特征，CPU 才跑得动
        tips.append("无可用 GPU：可以训练，但必须走 CPU 迁移学习路线 "
                    "（model.pretrained=true + model.mode=feature：冻结主干、缓存特征、只训分类头）")
        # /dev/shm < 256MB 时 DataLoader 多进程会因共享内存不足 ENOSPC，必须单进程
        workers = 0 if hw["shm_free_mb"] < 256 else "2~4"
        tips.append(f"推荐 train.cpu_threads={min(8, hw['cpu_logical'] or 4)}、"
                    f"train.num_workers={workers}、train.batch_size=64；纯 CPU 别用 finetune 模式")
        if workers == 0:
            tips.append(f"/dev/shm 仅 {hw['shm_free_mb']:.0f} MiB，DataLoader 必须单进程"
                        "（dataset.py 会自动降级）；物理机内存充足时可设 2~4")
        tips.append("主干优先 resnet18 / mobilenet_v3_small；数据量大时可先降 extract.fps 减少冗余帧")
    if hw["tmp_free_mb"] < 512:
        # 本机 /tmp 只有 10MB：pip 解压 wheel 会直接 ENOSPC，必须先改 TMPDIR
        tips.append("系统临时目录空间不足（安装大 wheel 会 ENOSPC），"
                    "安装依赖前先 export TMPDIR 到数据盘；cineeye 包导入时已自动兜底")
    if not weights["cached_weights"]:
        tips.append("未发现本地 ImageNet 权重：首次训练会联网下载约 45MB；"
                    "离线机器请把 .pth 放进 " + str(Path(weights["torch_home"]) / "hub" / "checkpoints"))
    else:
        tips.append("已缓存预训练权重：" + ", ".join(weights["cached_weights"]))
    tips.append("数据建议：每部电影至少 3 段不同片段、每段 >=100 帧；"
                "清单必须按 video_id 分组划分，否则验证集准确率会虚高")

    # 「就绪」只代表能跑起来：依赖齐全 + 设备可探测；权重可联网下载，不构成阻塞项
    ok = not missing and "error" not in dev
    return ok, tips


def main() -> None:
    """跑完所有检查并打印报告；--json 输出机器可读版本，供 CI 或其他脚本消费。"""
    ap = argparse.ArgumentParser(description="影瞳 · 运行环境自检")
    ap.add_argument("--json", action="store_true", help="以 JSON 输出，便于脚本消费")
    args = ap.parse_args()

    hw, dev, pkgs, weights = check_hardware(), check_device(), check_packages(), check_weights()
    ok, tips = recommend(hw, dev, pkgs, weights)
    report = {"ready": ok, "hardware": hw, "packages": pkgs, "device": dev,
              "pretrained_weights": weights, "recommendations": tips}

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    # ---- 人类可读报告：分块打印，方便人工逐项核对 ----
    print("=" * 72)
    print("影瞳 CineEye · 运行环境自检")
    print("=" * 72)
    print(f"结论：{'可以运行监督训练' if ok else '环境未就绪，请先补齐依赖'}")
    print(f"系统    : {hw['platform']}")
    print(f"Python  : {hw['python']}")
    print(f"CPU     : {hw['cpu_logical']} 逻辑核 | 内存 {hw['memory_total_gb']} GiB | "
          f"项目盘可用 {hw['project_disk_free_gb']} GiB")
    print(f"临时目录: {hw['tmp_dir']} (可用 {hw['tmp_free_mb']} MiB) | "
          f"/dev/shm 可用 {hw['shm_free_mb']} MiB")
    if dev.get("nvidia_gpu"):
        print(f"GPU     : {dev['nvidia_gpu']} | 算力 sm{dev.get('compute_cap', '?')} | "
              f"显存 {dev.get('vram_gb', '?')}GB | 驱动 {dev.get('driver', '?')}")
    elif not dev.get("cuda_available"):
        print("GPU     : 未检测到 NVIDIA GPU（当前走 CPU；代码支持在 GPU 机器上自动切换）")
    print("-" * 72)
    for name, ver in pkgs.items():
        print(f"  {name:<26} {ver}")
    print("-" * 72)
    for key, val in dev.items():
        print(f"  {key:<26} {val}")
    print("-" * 72)
    for i, tip in enumerate(tips, 1):
        print(f"  {i}. {tip}")
    print("=" * 72)


if __name__ == "__main__":
    main()
