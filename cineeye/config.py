"""配置模块：YAML 默认值 + 点号路径覆盖 + 相对路径解析。

用途：为全项目提供统一的配置读取/写入方式。所有入口脚本（train/predict 等 CLI）
    都通过 load_config() 得到一份 dict 配置，再用 get_value()/path() 按点号路径
    （如 train.batch_size）访问嵌套项，避免各处手写 YAML 解析逻辑。
被谁使用：cineeye/cli.py（把命令行 --set 覆盖项合并进配置）、cineeye/train.py、
    cineeye/evaluate.py、cineeye/predict.py 等入口模块。
依赖：PyYAML（yaml.safe_load / safe_dump）。标准库 copy / pathlib。
运维要点：
    - 默认配置文件是项目根下的 configs/default.yaml（DEFAULT_CONFIG），改默认行为
      应改那个文件，而不是这里。
    - ROOT 取的是「本文件上溯两级目录」，即项目根目录；所有相对路径都基于它解析，
      因此从任何工作目录启动脚本结果一致。
    - 覆盖项的值按 YAML 标量规则解析（yaml.safe_load），意味着值里若含 YAML 特殊
      字符（如冒号、#）需要小心引号；传字符串 "0.5" 会变成 float 0.5。
    - 本模块为纯函数式，无全局可变状态，修改风险低。
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

import yaml

# 项目根目录 = cineeye/ 的上一级；resolve() 消解符号链接，保证路径稳定
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None, overrides: Iterable[str] | None = None) -> dict:
    """读取配置文件并应用 ``--set a.b.c=value`` 形式的覆盖项。

    覆盖值会按 YAML 标量规则解析，因此 ``0.5`` 是 float、``true`` 是 bool、
    ``[1,2]`` 是 list，而不是字符串。这样命令行覆盖与 YAML 文件里的类型语义一致，
    下游代码无需对两种来源做不同处理。

    Args:
        path: YAML 配置文件路径；为 None 时使用 DEFAULT_CONFIG。
        overrides: 形如 ``a.b.c=value`` 的字符串序列（来自 CLI 的 --set，可重复）。

    Returns:
        合并覆盖项之后的配置 dict（注意：是独立对象，不与任何调用方共享）。

    Raises:
        FileNotFoundError: 配置文件不存在（由 open 抛出）。
        ValueError: 覆盖项格式不合法（缺少 "="）。
        yaml.YAMLError: YAML 文件或覆盖值本身无法解析。
    """
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with cfg_path.open("r", encoding="utf-8") as f:
        # 空文件/全注释文件 safe_load 返回 None，兜底成空 dict
        cfg = yaml.safe_load(f) or {}

    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"无效的配置覆盖项：{item!r}，应为 a.b.c=value")
        # split("=", 1)：值本身可能含 "="（如 base64 串），只按第一个 = 切 key
        key, raw = item.split("=", 1)
        # 用 YAML 解析覆盖值，使 "0.5"/"true"/"[1,2]" 得到正确类型而非字符串
        value = yaml.safe_load(raw.strip())
        set_value(cfg, key.strip(), value)
    return cfg


def get_value(cfg: dict, dotted: str, default: Any = None) -> Any:
    """按 ``a.b.c`` 点号路径读取嵌套配置。

    Args:
        cfg: 配置 dict。
        dotted: 点号分隔的键路径，如 ``train.batch_size``。
        default: 任一层级缺失时返回的兜底值。

    Returns:
        对应的配置值；路径不存在或中间层不是 dict 时返回 default。
    """
    node: Any = cfg
    for part in dotted.split("."):
        # 中间层不是 dict（例如某键被覆盖成了标量）也算"不存在"，返回 default
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def set_value(cfg: dict, dotted: str, value: Any) -> None:
    """按 ``a.b.c`` 点号路径写入嵌套配置（自动创建中间层）。

    Args:
        cfg: 被原地修改的配置 dict。
        dotted: 点号分隔的键路径。
        value: 要写入的值。

    Raises:
        TypeError: dotted 为空字符串时 parts[-1] 越界；正常调用不会触发。
    """
    parts = dotted.split(".")
    node = cfg
    # 逐层下钻；中间层缺失或不是 dict 时强制建成 dict，
    # 使 set_value("a.b.c", v) 在 cfg 为空 dict 时也能成功——覆盖项经常
    # 引用默认配置里没有的新键，必须容忍"从零建树"
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def path(cfg: dict, dotted: str, mkdir: bool = False) -> Path:
    """把配置里的相对路径项解析为基于项目根目录（ROOT）的绝对路径。

    Args:
        cfg: 配置 dict。
        dotted: 指向路径字符串的点号键路径，如 ``data.raw_dir``。
        mkdir: 为 True 时顺带创建目录（含父目录），常用于输出目录。

    Returns:
        绝对 Path；相对路径以 ROOT 为基准拼接。

    Raises:
        KeyError: 该配置项不存在（值为 None 或键缺失）——路径是必需项，
            与 get_value 的 default 语义不同。
    """
    raw = get_value(cfg, dotted)
    if raw is None:
        raise KeyError(f"配置中不存在路径项：{dotted}")
    p = Path(raw)
    # 只有相对路径才基于 ROOT 拼接；写绝对路径的配置不受项目位置影响
    if not p.is_absolute():
        p = ROOT / p
    if mkdir:
        # exist_ok=True：并发/重复调用不会因目录已存在而报错
        p.mkdir(parents=True, exist_ok=True)
    return p


def snapshot(cfg: dict, dest: str | Path | None = None) -> dict:
    """返回配置的深拷贝；给定 dest 时同时落盘为 YAML，便于复现实验。

    Args:
        cfg: 要快照的配置 dict。
        dest: 可选的 YAML 输出路径；父目录不存在会自动创建。

    Returns:
        cfg 的深拷贝（包含已应用的覆盖项）。深拷贝是为了防止后续训练代码
        原地修改配置后，落盘的"实验记录"也跟着变。
    """
    cfg = copy.deepcopy(cfg)
    if dest is not None:
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        # allow_unicode 保留中文可读性；sort_keys=False 保持与源文件相同的键序，
        # diff 两次实验的快照时更直观
        with dest.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return cfg
