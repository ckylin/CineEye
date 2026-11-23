"""影瞳 CineEye —— 用图片/视频帧识别电影的监督学习工具箱。

用途：包的入口模块。除定义版本号与子模块清单外，核心职责是在 **import 时**
    完成两项环境自愈（见文末两个 _ensure/_use 调用），让后续所有模块在
    "受限的 /tmp" 与 "离线/本地权重" 场景下都能正常工作。
被谁使用：所有 `import cineeye` / `from cineeye.xxx import ...` 的模块与脚本；
    两个环境自愈函数在其依赖模块（features、train 等）加载前执行，
    所以必须保持在本模块顶层调用，不能挪进函数里。
依赖：仅标准库（os / shutil / tempfile / pathlib）。torch / torchvision 的
    路径调整发生在环境变量层面，本模块不直接 import 它们。
运维要点：
    - 核心链路：视频 → 抽帧 → 构建标注清单（按视频分组划分）→ 预训练 CNN 提特征
      → 训练分类头 → 帧级/视频级评估 → 单图或整段视频推理。

    - __all__ 列出的子模块是包的"官方公开面"；新增子模块时应同步维护此列表。
    - 两个自愈函数在 import 期执行且静默吞异常：诊断 /tmp 或权重缓存问题时
      先从这里查起（TMPDIR / TORCH_HOME 两个环境变量是关键线索）。
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

__version__ = "0.1.0"
__all__ = ["config", "utils", "device", "data", "models", "features", "train", "evaluate", "predict"]


def _ensure_writable_tmp() -> None:
    """本机的 /tmp 是 10MB tmpfs，下载模型权重 / 大 wheel 会直接 ENOSPC。

    临时空间不足时自动把 TMPDIR 改到项目目录下的 .tmp（数据盘），
    避免「磁盘明明还有 600G 却提示设备上没有空间」这种坑。

    实现要点：阈值设为 512MB——低于它就认为放不下 torchvision 权重
    （几百 MB 起）或 pip 缓存；用 os.environ.setdefault 而非直接赋值，
    尊重用户已显式设置的 TMPDIR；同时给 tempfile.tempdir 赋值，因为
    CPython 在本进程内已经缓存过 tempdir，只改环境变量不生效。

    Raises:
        不抛异常——import 期的环境探测失败不应阻碍整个包的加载。
    """
    try:
        tmp = Path(tempfile.gettempdir())
        # /tmp 剩余空间 >= 512MB 才认为是可用的临时盘；否则切换到项目内 .tmp
        if shutil.disk_usage(tmp).free >= 512 * 1024 * 1024:
            return
        alt = Path(__file__).resolve().parents[1] / ".tmp"
        alt.mkdir(parents=True, exist_ok=True)
        # setdefault：用户显式设过 TMPDIR 时不覆盖；再直接改 tempfile.tempdir，
        # 因为该值在进程内已被缓存，仅改环境变量对本次进程无效
        os.environ.setdefault("TMPDIR", str(alt))
        tempfile.tempdir = str(alt)
    except Exception:  # noqa: BLE001 - 环境探测失败不应阻碍导入
        pass


def _use_local_pretrained_weights() -> None:
    """项目内若已预置 ImageNet 权重，让 torchvision 直接读本地缓存，离线也能训练。

    只有确实存在 .pth 文件时才改 TORCH_HOME，避免覆盖用户已有的全局权重缓存。

    依赖关系：必须在本包被 import 时（即 features/models 加载 torchvision 之前）
    执行，环境变量才会被 torchvision 的 load_state_dict_from_url 读取到。

    Raises:
        不抛异常——同样属于 import 期自愈逻辑，失败静默跳过。
    """
    try:
        local = Path(__file__).resolve().parents[1] / ".cache" / "torch"
        ckpt_dir = local / "hub" / "checkpoints"
        # 三个条件缺一不可：用户没显式设 TORCH_HOME、目录存在、里面确有 .pth；
        # 不检查"确有权重"的话会把用户本来可用的全局缓存指到一个空目录
        if "TORCH_HOME" not in os.environ and ckpt_dir.is_dir() and any(ckpt_dir.glob("*.pth")):
            os.environ["TORCH_HOME"] = str(local)
    except Exception:  # noqa: BLE001
        pass


# import 期执行环境自愈：顺序无关，但都必须在子模块被加载前完成
_ensure_writable_tmp()
_use_local_pretrained_weights()
