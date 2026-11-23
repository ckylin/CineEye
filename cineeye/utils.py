"""通用工具模块：日志、随机种子、JSON IO、目录创建、文件签名、计时。

用途：收纳各模块都会用到的小工具函数，避免重复实现。重点是
    file_signature()——用「路径+大小+mtime」的内容签名判断特征缓存是否过期，
    是 features 模块缓存机制的基石。
被谁使用：cineeye/device.py（get_logger）、cineeye/features.py（file_signature
    生成缓存键）、cineeye/train.py / evaluate.py / predict.py（seed_everything、
    save_json、timer）。
依赖：numpy（随机种子）；torch 为可选依赖（未安装时 seed_everything 自动跳过）。
运维要点：
    - get_logger 有"只加一次 handler"的保护，但多次以不同 name 调用会产生
      多个 logger； propagate=False 防止日志重复输出。
    - PYTHONHASHSEED 必须在解释器启动前设置才对 str 哈希生效，本模块在运行期
      设置它只影响子进程；主进程内 str 哈希随机化无法回退。
    - file_signature 只看 stat（路径/大小/mtime），不读文件内容——速度快，
      但"改内容保持同大小同 mtime"的极端情况会漏检；正常运维中可忽略。
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np


def get_logger(name: str = "cineeye", level: str = "INFO") -> logging.Logger:
    """获取/创建项目统一格式的 logger。

    同名 logger 已存在（有 handler）时直接复用，保证「多次 import、多次调用
    只挂一个 handler」，否则日志会按 handler 数量重复打印。

    Args:
        name: logger 名称，默认项目级 "cineeye"。
        level: 日志级别字符串（如 "INFO"/"DEBUG"），仅在首次创建时生效。

    Returns:
        配置好格式（[时间] 级别 消息）且 propagate=False 的 Logger。
    """
    logger = logging.getLogger(name)
    # 已有 handler 说明初始化过，直接复用，避免重复添加 handler 导致日志翻倍
    if logger.handlers:
        return logger
    # 级别字符串非法时兜底 INFO，而不是抛 AttributeError 让进程崩掉
    logger.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)-7s %(message)s", "%H:%M:%S"))
    logger.addHandler(handler)
    # 关闭向 root logger 传播，防止 root 上另有 handler 时同一行日志输出两遍
    logger.propagate = False
    return logger


def seed_everything(seed: int = 42) -> None:
    """统一设置 random / numpy / torch 的随机种子，保证实验可复现。

    Args:
        seed: 随机种子值，默认 42。

    Note:
        torch 是可选依赖：纯工具环境（未装 torch）下静默跳过 torch 部分，
        不抛异常——本函数被入口脚本调用，不应因缺依赖而中断。
    """
    random.seed(seed)
    # 严格说 PYTHONHASHSEED 要在解释器启动前设置才对当前进程生效；
    # 这里设置它主要为了让后续启动的子进程（DataLoader worker 等）继承同一种子
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        # manual_seed_all 覆盖所有可见 GPU，而不仅是 cuda:0
        torch.cuda.manual_seed_all(seed)
        # 让 cuDNN 等后端结果稳定；CPU 上无副作用
        torch.use_deterministic_algorithms(False)
    except ImportError:  # pragma: no cover
        pass


def save_json(obj: Any, path: str | Path) -> None:
    """把对象序列化为 UTF-8 JSON 并落盘，父目录不存在时自动创建。

    Args:
        obj: 可 JSON 序列化的对象（dict / list / 标量等）。
        path: 输出文件路径。

    Raises:
        TypeError: obj 含不可序列化类型（如 numpy 数组需先转 list/float）。
    """
    path = Path(path)
    # 报告/指标常写到尚不存在的实验目录，顺手建目录省去调用方 mkdir
    path.parent.mkdir(parents=True, exist_ok=True)
    # ensure_ascii=False 让中文直接以 UTF-8 落盘，肉眼检查报告时不用解码
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> Any:
    """读取 UTF-8 JSON 文件。

    Args:
        path: JSON 文件路径。

    Returns:
        反序列化后的 Python 对象。

    Raises:
        FileNotFoundError: 文件不存在。
        json.JSONDecodeError: 文件内容不是合法 JSON。
    """
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str | Path) -> Path:
    """确保目录存在（含全部父目录），并返回 Path 便于链式使用。

    Args:
        path: 目标目录路径，str 或 Path 均可。

    Returns:
        该目录的 Path 对象。
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def file_signature(paths: list[str | Path] | None = None, extra: str = "") -> str:
    """根据文件内容特征生成短签名，用于判断特征缓存是否过期。

    用「绝对路径 + 文件大小 + mtime」拼进 SHA-256，而不是直接哈希文件内容：
    数据集动辄几 GB，逐字节读文件做签名太慢；而大小+mtime 的组合在正常
    运维场景（重新抽帧、重新标注、改配置）下足以发现变化。extra 里通常
    混入模型名/分辨率等超参，保证"换模型后不会错用旧缓存"。

    Args:
        paths: 参与签名的文件路径列表（如标注清单、视频清单）；None 视为空。
        extra: 额外混入签名的字符串（模型名、图像尺寸等）。

    Returns:
        16 位十六进制签名字符串（截断到 16 字符足够防碰撞，且便于做文件名）。

    Raises:
        OSError: stat 失败等文件系统错误（exists 与 stat 之间文件被删除的
            竞态极小，未特殊处理）。
    """
    import hashlib

    h = hashlib.sha256(extra.encode("utf-8"))
    for p in paths or []:
        p = Path(p)
        if not p.exists():
            # 文件"曾经参与过缓存、现在消失了"也要改变签名，
            # 否则删文件后可能命中旧缓存
            h.update(f"missing:{p}".encode())
            continue
        h.update(str(p).encode())
        h.update(str(p.stat().st_size).encode())
        # mtime 取整到秒：不同机器/复制过程的小数误差不影响同一份文件的签名
        h.update(str(int(p.stat().st_mtime)).encode())
    return h.hexdigest()[:16]


@contextmanager
def timer(label: str, logger: logging.Logger | None = None):
    """计时上下文管理器，退出时输出「label 耗时 X.Xs」。

    Args:
        label: 阶段名称，如 "抽帧"、"特征提取"。
        logger: 指定 logger；为 None 时用默认的 get_logger()。

    Yields:
        None（仅作为上下文占位）。

    Note:
        用 perf_counter 而非 time.time：前者是单调时钟，不受系统改时间影响。
    """
    start = time.perf_counter()
    yield
    msg = f"{label} 耗时 {time.perf_counter() - start:.1f}s"
    (logger or get_logger()).info(msg)
