"""设备探测与线程/显存调优模块（CPU 与 NVIDIA GPU 通吃）。

用途：把"用哪个设备跑"这件事收敛到一处——resolve_device() 按回退链路
    选出真正能用的 torch.device，describe_device()/log_device() 输出环境摘要，
    供训练报告与 /health 接口使用。
被谁使用：cineeye/train.py、cineeye/evaluate.py、cineeye/predict.py（启动时
    resolve_device + log_device）；服务端 /health 端点（describe_device）。
依赖：torch（硬依赖，本模块在 import 时即需要）；cineeye.utils.get_logger。
运维要点：
    - 设计目标：同一份代码在本机纯 CPU（Intel Core Ultra / 无独显）与用户的
      NVIDIA 台式机（如 RTX 5070 12G，Blackwell sm_120）上都能自动选到最优设备，
      调用方无需改代码，只需 ``--device auto``（默认）或显式指定。

    - CPU：限制线程数，避免 PyTorch 开满逻辑核导致小 batch 推理反而变慢。
    - CUDA：检查驱动/内核是否真的可用（含 sm_120 等新架构的 "no kernel image" 陷阱）；
      GPU 可用时给出 AMP / finetune / 更大 batch 的建议。

    - _CUDA_PROBE_CACHE 是模块级全局缓存，探测结果跨调用共享；改 cuda_kernel_ok
      的逻辑时注意缓存可能掩盖新改动（必要时重启进程清缓存）。
    - torch wheel 与 GPU 架构必须匹配：RTX 50 系（sm_120）需要 cu128 wheel +
      torch>=2.7；装成 cu118/cu126 等旧 wheel 时 is_available() 仍为 True，
      但内核无法执行——本模块的探测就是为了提前暴露这种情况。
"""

from __future__ import annotations

import os
import platform
from typing import Any

import torch

from .utils import get_logger

log = get_logger()


def _safe(fn, default: Any = False) -> Any:
    """容错执行 fn，任何异常都吞掉并返回 default。

    目的：设备探测面对的是"环境千差万别"的现实（旧驱动、精简 wheel、
    后端未编译等），任何一次探测失败都不应让整个程序崩溃。

    Args:
        fn: 无参可调用对象。
        default: 异常时的返回值，默认 False。

    Returns:
        fn() 的结果，或 default。
    """
    try:
        return fn()
    except Exception:
        return default


# 模块级探测缓存：cuda_kernel_ok 只真正跑一次内核，之后直接查表；
# key 是设备索引字符串（探测代价高且结果在进程内不会变）
_CUDA_PROBE_CACHE: dict[str, Any] = {}


def cuda_kernel_ok(device_index: int = 0) -> bool:
    """真实探测 CUDA 内核能否在该 GPU 上执行，而不只看 is_available()。

    RTX 50 系（Blackwell sm_120）若装了旧 wheel，``is_available()`` 可能为 True，
    但第一次真正运算会抛 ``no kernel image is available for execution on the device``。
    这里跑一次极小的张量加法提前暴露，并缓存结果避免每次调用都探测。

    Args:
        device_index: 要探测的 CUDA 设备索引，默认 0。

    Returns:
        True 表示该 GPU 上内核确实可执行；False 表示不可用或探测失败。
        结果已写入模块级 _CUDA_PROBE_CACHE，进程内只探测一次。

    Raises:
        不抛异常——所有失败都转化为 False 并打 warning。
    """
    key = str(device_index)
    if key in _CUDA_PROBE_CACHE:
        return bool(_CUDA_PROBE_CACHE[key])
    if not _safe(torch.cuda.is_available):
        _CUDA_PROBE_CACHE[key] = False
        return False
    try:
        # 用一次 8x8 矩阵乘做最小内核探测：代价可忽略，但能触发内核加载
        x = torch.ones(8, 8, device=f"cuda:{device_index}")
        (x @ x).cpu()  # 触发内核加载；架构不匹配会在此抛错
        ok = True
    except Exception as exc:  # noqa: BLE001
        ok = False
        # 典型场景：RTX 5070（sm_120）装了 cu118/cu126 旧 wheel——
        # is_available() 为 True 但没有 sm_120 的内核镜像。给出精确的修复命令
        log.warning(
            "CUDA 报告可用但内核无法执行（%s）。若是 RTX 50 系（sm_120），"
            "请安装 CUDA 12.8 版 PyTorch：pip install torch torchvision "
            "--index-url https://download.pytorch.org/whl/cu128（需 torch>=2.7）", exc,
        )
    _CUDA_PROBE_CACHE[key] = ok
    return ok


def resolve_device(preference: str = "auto", cpu_threads: int = 8) -> torch.device:
    """按优先级挑选可用设备，永远保证返回一个能跑的设备。

    preference 取值：auto / cpu / cuda / cuda:N / mps / xpu。

    回退链路（关键运维知识）：
        - auto：cuda（内核探测通过）→ mps → xpu → cpu；
        - 显式请求 cuda/cuda:N：内核探测失败时回退 cpu 并告警；
        - mps / xpu：后端不可用时静默回退 cpu；
        - cpu：不回退，直接生效。

    Args:
        preference: 设备偏好字符串，大小写不敏感，默认 "auto"。
        cpu_threads: 最终落在 CPU 时设置的 torch 线程数上限。本机是混合架构
            CPU（P 核+E 核），开满全部逻辑核反而增加调度开销，所以默认只给 8。

    Returns:
        一个保证可用的 torch.device（不会返回探测失败的设备）。
    """
    pref = str(preference).strip().lower()
    if pref == "cpu":
        device = torch.device("cpu")
    elif pref.startswith("cuda"):
        # 支持 "cuda" / "cuda:0" / "cuda:1"
        # 注意：这里只探测 cuda:0；多卡机器请求 cuda:1 时若 0 号卡坏会误回退
        if cuda_kernel_ok(0):
            device = torch.device(pref)
        else:
            log.warning("请求了 %s 但 CUDA 内核不可用，已回退到 CPU", pref)
            device = torch.device("cpu")
    elif pref == "mps":
        device = torch.device("mps" if _safe(lambda: torch.backends.mps.is_available()) else "cpu")
    elif pref == "xpu":
        has_xpu = _safe(lambda: hasattr(torch, "xpu") and torch.xpu.is_available())
        device = torch.device("xpu" if has_xpu else "cpu")
    else:  # auto
        # auto 链路：CUDA 内核可用性优先于 is_available()——
        # 这正是为了拦截"报告可用但内核跑不了"的 sm_120 旧 wheel 场景
        if cuda_kernel_ok(0):
            device = torch.device("cuda")
        elif _safe(lambda: torch.backends.mps.is_available()):
            device = torch.device("mps")
        elif _safe(lambda: hasattr(torch, "xpu") and torch.xpu.is_available()):
            device = torch.device("xpu")
        else:
            device = torch.device("cpu")

    if device.type == "cpu":
        cores = os.cpu_count() or 4
        # 钳制到 [1, 核数范围内]：既防止配置填 0/负数，也防止超过总核数
        n = max(1, min(int(cpu_threads), cores))
        torch.set_num_threads(n)
        try:
            # interop 线程只允许设置一次；resolve_device 被多次调用时
            # 第二次起会抛 RuntimeError，静默忽略即可
            torch.set_num_interop_threads(1)
        except RuntimeError:  # 已初始化过会抛错，忽略
            pass
    return device


def describe_device(device: torch.device | None = None) -> dict:
    """返回一份可读的环境摘要，写入训练报告 / 终端输出 / HTTP /health。

    Args:
        device: 当前选定的设备；为 None 时摘要里 selected_device 显示 "n/a"。

    Returns:
        包含 Python/torch 版本、CPU 核数、CUDA 可用性与内核探测结果、
        GPU 名称/算力/显存等键的 dict。所有子探测都经 _safe 包裹，
        任何一项失败只是缺键，不影响整体返回。
    """
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        # 注意区分两个层次：cuda_available 只说明驱动层面可用，
        # cuda_kernel_ok 才说明内核真的能跑（见 cuda_kernel_ok 的 sm_120 说明）
        "cuda_available": bool(_safe(torch.cuda.is_available)),
        "cuda_kernel_ok": bool(_safe(cuda_kernel_ok)),
        "cuda_version": _safe(lambda: torch.version.cuda) or None,
        "mps_available": bool(_safe(lambda: torch.backends.mps.is_available())),
        "xpu_available": bool(_safe(lambda: hasattr(torch, "xpu") and torch.xpu.is_available())),
        "selected_device": str(device) if device else "n/a",
    }
    if info["cuda_available"]:
        try:
            info["cuda_device"] = torch.cuda.get_device_name(0)
            # 算力形如 "12.0"（即 sm_120），排查 wheel 匹配问题的关键字段
            info["cuda_capability"] = ".".join(
                str(x) for x in torch.cuda.get_device_capability(0))
            free, total = torch.cuda.mem_get_info(0)
            info["cuda_vram_gb"] = round(total / 1024 ** 3, 1)
            info["cuda_vram_free_gb"] = round(free / 1024 ** 3, 1)
        except Exception:  # noqa: BLE001
            # 仅供展示的信息，获取失败不值得让调用方失败
            pass
    return info


def log_device(device: torch.device, cpu_threads: int = 8) -> dict:
    """把设备摘要按设备类型格式化输出到日志，并返回摘要 dict。

    Args:
        device: resolve_device() 返回的设备。
        cpu_threads: 与 describe_device 摘要展示相关的线程数上下文（当前
            仅透传，实际线程数已在 resolve_device 中设置）。

    Returns:
        describe_device() 的结果，调用方可继续把它写进训练报告。

    Note:
        GPU 模式下会额外给出 finetune/AMP 建议（12G 显存对 RTX 5070 足够
        finetune）；CPU 且无 CUDA 时提示保持 model.mode=feature，避免用户
        在 CPU 上误开 finetune 拖慢数个量级。
    """
    info = describe_device(device)
    if device.type == "cuda":
        log.info(
            "设备=%s | GPU=%s 算力=sm%s 显存=%.1fGB | torch=%s+cuda%s",
            info["selected_device"], info.get("cuda_device", "?"),
            info.get("cuda_capability", "?"), info.get("cuda_vram_gb", 0.0),
            info["torch"], info.get("cuda_version", "?"),
        )
        log.info("GPU 模式：建议 model.mode=finetune + train.amp=true 提速；"
                 "显存不足时调小 train.batch_size")
    elif device.type == "cpu":
        log.info(
            "设备=%s | torch=%s | CPU线程=%d | CUDA可用=%s",
            info["selected_device"], info["torch"], info["torch_threads"],
            info["cuda_available"],
        )
        if not info["cuda_available"]:
            log.info("未检测到可用 CUDA GPU，保持 model.mode=feature（冻结主干 + 轻量分类头）；"
                     "若有 NVIDIA 显卡，装 CUDA 版 torch 后设 --device auto 即可自动切换")
        elif not info["cuda_kernel_ok"]:
            # 到达这里说明 is_available()=True 但内核探测失败，即典型的
            # wheel 架构不匹配问题；device 已经回退 CPU，提醒用户修环境
            log.warning("检测到 CUDA 设备但内核不可用，当前走 CPU。请升级到匹配 GPU 架构的 torch wheel")
    else:
        log.info("设备=%s | torch=%s | CPU线程=%d",
                 info["selected_device"], info["torch"], info["torch_threads"])
    return info
