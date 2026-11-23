"""CLI 公共参数模块：``--config`` + 任意个 ``--set a.b.c=value`` + ``--device``。

用途：为所有入口脚本（train / evaluate / predict 等）提供一套统一的命令行
    参数骨架，保证每个子命令的参数风格一致，运维人员学一次就能用全部命令。
被谁使用：cineeye/train.py、cineeye/evaluate.py、cineeye/predict.py 等入口
    模块（各自 base_parser() 后再追加自己的专属参数）。
依赖：标准库 argparse；cineeye.config（load_config / set_value）。
运维要点：
    - --device 的优先级高于 --config 和 --set（load_cfg 中最后写入
      train.device），排查"设备不生效"问题时先确认是否被 --device 覆盖。
    - 修改本文件会影响所有子命令的帮助文本与行为，属公共接口。
"""

from __future__ import annotations

import argparse

from .config import get_value, load_config, set_value


def base_parser(description: str) -> argparse.ArgumentParser:
    """构建所有入口脚本共享的基础 argparse 解析器。

    Args:
        description: 各子命令自己的描述文本，显示在 --help 顶部。

    Returns:
        已注册 ``--config`` / ``--device`` / ``--set`` 三个参数的解析器；
        调用方应继续在其上 p.add_argument(...) 追加专属参数。

    Note:
        formatter_class 用 ArgumentDefaultsHelpFormatter，使 --help 自动
        显示默认值，减少"参数默认是什么"的翻文档成本。
        ``--set`` 用 action="append"，可重复出现，收集进 args.overrides 列表。
    """
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", default=None, help="YAML 配置文件路径（默认 configs/default.yaml）")
    p.add_argument("--device", default=None,
                   help="推理/训练设备：auto/cpu/cuda/cuda:0/mps/xpu（默认 auto，有 NVIDIA GPU 时自动用 GPU）")
    p.add_argument("--set", action="append", default=[], dest="overrides", metavar="a.b.c=VALUE",
                   help="覆盖任意配置项，可重复，例如 --set train.epochs=10")
    return p


def load_cfg(args) -> dict:
    """把解析后的命令行参数（args）转换为最终生效的配置 dict。

    Args:
        args: base_parser() 解析出的 argparse.Namespace（含 config / overrides /
            可选的 device 属性）。

    Returns:
        合并了 --config 文件、全部 --set 覆盖项、--device 快捷方式后的配置 dict。

    Raises:
        ValueError: --set 覆盖项缺少 "="（由 load_config 抛出）。
        FileNotFoundError: --config 指向的文件不存在。
    """
    cfg = load_config(args.config, args.overrides)
    # --device 是 train.device 的快捷方式；优先级高于 --config/--set，
    # 便于同一份配置在 CPU 笔记本与 GPU 台式机间切换而不改文件。
    # 注意：未传 --device 时（default=None）不写入，保留配置文件里的原值。
    device = getattr(args, "device", None)
    if device:
        set_value(cfg, "train.device", device)
    return cfg
