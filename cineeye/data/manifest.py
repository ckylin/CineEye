"""扫描抽帧结果，生成带 split 列的标注清单。

用途：把 ``data/frames/`` 下的目录结构还原成一张帧级表（DataFrame），
按 ``video_id`` 分组划分 train/val/test，输出 ``data/manifest.csv``
与 ``outputs/label_map.json``（类别名 → 整数下标）。

最关键的一点：**按 video_id 分组划分**。电影相邻帧几乎一模一样，
若按帧随机划分，验证集准确率会虚高到接近 100%，但模型实际什么都没学会。

被谁使用：
    - ``scripts/02_build_manifest.py``（训练流水线第二步）；
    - ``scripts/run_all.py`` 依次调用 01 → 02；
    - ``tests/test_pipeline.py`` 单独测试 :func:`group_split` 的防泄漏性。

依赖：pandas（清单读写）、``cineeye.config``、``cineeye.utils``。
纯 CPU、纯 I/O，无 GPU 依赖。

运维要点：
    1. 改动手工整理过 ``manifest.csv`` 后**不要**再跑 02，否则会被全量覆盖。
    2. split 比例/seed 改动会让 ``manifest.csv`` 内容变化，
       ``features.py`` 的缓存签名随之失效，深度特征会自动重抽（耗时）。
    3. 新加一部电影后必须重跑 02；只加视频而不重跑，新视频不会进训练集。
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from ..config import ROOT, get_value
from ..config import path as cfg_path
from ..utils import ensure_dir, get_logger, save_json

log = get_logger()
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _rel(p: Path) -> str:
    """同 ``extract_frames._relative_to_root``：绝对路径 → 相对项目根的 POSIX 字符串。

    两个模块各留一份而不抽公共工具，是为了让 data 层不互相 import
    （extract_frames 与 manifest 可独立运行，避免循环依赖）。

    Args:
        p: 图片路径。

    Returns:
        相对 ``ROOT`` 的路径；不在项目内时返回绝对路径。
    """
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return p.resolve().as_posix()


def scan_frames(frames_dir: str | Path) -> pd.DataFrame:
    """支持两种布局：

    1. ``frames/<电影名>/<视频id>/000000.jpg`` —— 视频抽帧（推荐）
    2. ``frames/<电影名>/xxx.jpg``             —— 用户已有的散装截图

    散装截图按每张图一个 video_id 处理，仍然不会跨集合泄漏。

    注意：如果某个标签目录下同时存在子目录和散装图片，两种布局会**同时生效**，
    即两类图片都会进入清单；这是有意为之，方便用户混放补图。

    Args:
        frames_dir: 帧根目录（``data/frames``）。

    Returns:
        DataFrame，列为 ``frame_path`` / ``label`` / ``video_id`` / ``frame_index``。
        已按 ``frame_path`` 去重。

    Raises:
        FileNotFoundError: ``frames_dir`` 不存在，异常信息里带有操作指引。
        RuntimeError: 目录存在但一张图片都没有。
    """
    frames_dir = Path(frames_dir)
    if not frames_dir.exists():
        raise FileNotFoundError(
            f"帧目录不存在：{frames_dir}\n请先放入 data/videos/<电影名>/*.mp4 并运行抽帧脚本，"
            "或把图片直接放到 data/frames/<电影名>/ 下。"
        )

    rows: list[dict] = []
    for label_dir in sorted(p for p in frames_dir.iterdir() if p.is_dir() and not p.name.startswith(".")):
        label = label_dir.name
        # 布局 1：含子目录
        subdirs = sorted(p for p in label_dir.iterdir() if p.is_dir() and not p.name.startswith("."))
        if subdirs:
            for sub in subdirs:
                for img in sorted(p for p in sub.iterdir() if p.suffix.lower() in IMAGE_EXTS):
                    rows.append({
                        "frame_path": _rel(img), "label": label,
                        "video_id": f"{label}/{sub.name}", "frame_index": img.stem,
                    })
        # 布局 2：图片直接躺在标签目录下
        # 每张散装图单独当一个 video_id：宁可把「同片不同截图」分到不同集合
        # （导致验证略偏乐观），也不能让同一张图跨集合（那是硬泄漏）。
        for img in sorted(p for p in label_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS):
            rows.append({
                "frame_path": _rel(img), "label": label,
                "video_id": f"{label}/{img.stem}", "frame_index": img.stem,
            })

    if not rows:
        raise RuntimeError(f"在 {frames_dir} 下没有找到任何图片。")

    df = pd.DataFrame(rows)
    # 去重兜底：布局 1 与布局 2 同时命中、或软链接重复时可能产生重复路径。
    # 重复帧在训练里等于给同一张图加权，会让类别分布悄悄偏移。
    duplicated = df["frame_path"].duplicated().sum()
    if duplicated:
        df = df.drop_duplicates("frame_path").reset_index(drop=True)
    return df


def _allocate(n: int, ratios: dict[str, float]) -> dict[str, int]:
    """把 n 个视频分到三个集合，尽量保证每类在每个集合都有代表。

    以「视频」而不是「帧」为最小分配单位——这是防泄漏的前提。
    三个集合各至少 1 段视频（n>=3 时），因为 val/test 为空会让训练指标
    完全失去参考意义，也容易在后面除零。

    Args:
        n: 该类别的视频段数。
        ratios: ``{"train": .., "val": .., "test": ..}``，不必归一化，内部会归一。

    Returns:
        ``{"train": int, "val": int, "test": int}``，三者之和 == n。

    Raises:
        无。任何 n 都有确定性的降级方案（n==2 时 test=0，n==1 时全给 train）。
    """
    keys = ["train", "val", "test"]
    total = sum(ratios[k] for k in keys) or 1.0
    if n >= 3:
        out = {k: max(1, int(round(n * ratios[k] / total))) for k in keys}
        # 四舍五入 + 每集合至少 1 段，可能多分或少分，这里双向修正到正好 n
        while sum(out.values()) > n:
            k = max(keys, key=lambda x: out[x])
            if out[k] > 1:
                out[k] -= 1
            else:
                break
        while sum(out.values()) < n:
            out["train"] += 1
    elif n == 2:
        # 两段视频只能覆盖两个集合，约定丢掉 test
        out = {"train": 1, "val": 1, "test": 0}
    else:
        out = {"train": 1, "val": 0, "test": 0}
    return out


def group_split(df: pd.DataFrame, ratios: dict[str, float], seed: int = 42) -> pd.Series:
    """分层 + 分组划分，返回与 df 对齐的 split 列。

    流程：按 label 分层 → 每层收集该电影的所有 video_id → 打乱 → 按比例
    把「整段视频」整体塞进某个 split。同一 video_id 的所有帧必然同属一个 split。

    为什么必须按 video_id 分组：同一部电影相邻 1 秒的帧相似度常常 >0.99，
    若按帧随机划分，验证集会包含训练集的「近重复图」，指标虚高但毫无泛化能力。
    这是影视类图像任务最常见、也最容易被忽略的数据泄漏。

    Args:
        df: 至少包含 ``label`` 与 ``video_id`` 两列。
        ratios: train/val/test 比例。
        seed: 随机种子，默认取 ``project.seed``，保证多次运行划分一致
            （否则每次重跑 manifest 都会让特征缓存失效）。

    Returns:
        与 ``df`` 等长的 Series，值为 ``"train"`` / ``"val"`` / ``"test"``。
        任何未分配到的 video_id（理论上不会发生）一律兜底为 train。

    Raises:
        KeyError: df 缺少 ``label`` 或 ``video_id`` 列时由 pandas 抛出。
    """
    rng = random.Random(seed)
    assignment: dict[str, str] = {}
    low_data_labels: list[str] = []

    # 按 label 分组后再划分，保证每个类别在 train/val/test 都有代表，
    # 否则极端情况下某个类可能整个落在 test 里，训练时模型根本没见过它。
    for label, group in df.groupby("label", sort=True):
        vids = sorted(group["video_id"].unique().tolist())
        # 先排序再 shuffle：pandas groupby / unique 的顺序依赖字典实现，
        # 排序是让结果只由 seed 决定，跨机器、跨 pandas 版本可复现。
        rng.shuffle(vids)
        counts = _allocate(len(vids), ratios)
        if len(vids) < 3:
            low_data_labels.append(f"{label}({len(vids)}段视频)")
        cursor = 0
        for split, cnt in counts.items():
            # 切片是「整段视频」级别的搬运，绝不会把一段视频拆到两个集合
            for vid in vids[cursor:cursor + cnt]:
                assignment[vid] = split
            cursor += cnt

    if low_data_labels:
        log.warning("以下类别视频数 < 3，无法在 train/val/test 中都有独立视频，"
                    "建议每个电影至少准备 3 段不同片段：%s", ", ".join(low_data_labels))
    return df["video_id"].map(assignment).fillna("train")


def build_manifest(cfg: dict) -> tuple[pd.DataFrame, dict[str, int]]:
    """扫描帧 → 划分数据集 → 写 manifest.csv 与 label_map.json。

    Args:
        cfg: 全局配置。读取 ``data.frames_dir``、``data.split.*``、
            ``project.seed``，输出到 ``data.manifest`` / ``data.label_map``。

    Returns:
        ``(df, label_map)``：
            df 额外含 ``label_idx``、``split`` 两列，并按 split/label/video_id
            排序；label_map 为 ``{电影名: 类别下标}``（按字典序编号）。

    Raises:
        FileNotFoundError: 帧目录不存在（来自 :func:`scan_frames`）。
        RuntimeError: 没有图片；或自检发现同一 video_id 跨 split（理论上不应出现，
            属于最后一道防线，一旦触发说明划分逻辑被改坏了）。
    """
    frames_dir = cfg_path(cfg, "data.frames_dir")
    df = scan_frames(frames_dir)

    # 类别下标按电影名字典序固定编号：只要不增删电影，已有 checkpoint 的分类头
    # 权重仍然与 label_map 对齐。增删电影会让所有下标平移，必须重新训练。
    labels = sorted(df["label"].unique().tolist())
    label_map = {name: i for i, name in enumerate(labels)}
    df["label_idx"] = df["label"].map(label_map).astype(int)

    ratios = {
        "train": float(get_value(cfg, "data.split.train", 0.7)),
        "val": float(get_value(cfg, "data.split.val", 0.15)),
        "test": float(get_value(cfg, "data.split.test", 0.15)),
    }
    df["split"] = group_split(df, ratios, int(get_value(cfg, "project.seed", 42)))
    df = df.sort_values(["split", "label", "video_id", "frame_index"]).reset_index(drop=True)

    manifest_path = cfg_path(cfg, "data.manifest")
    ensure_dir(manifest_path.parent)
    df.to_csv(manifest_path, index=False, encoding="utf-8")
    save_json(label_map, cfg_path(cfg, "data.label_map"))

    log.info("类别数=%d | 帧数=%d | 视频数=%d", len(labels), len(df), df["video_id"].nunique())
    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split]
        log.info("  %-5s: %6d 帧 / %3d 段视频 / %d 类", split, len(sub),
                 sub["video_id"].nunique(), sub["label"].nunique())
    # 泄漏自检：同一 video_id 不应出现在两个 split
    # 这是 post-condition 断言——不信任上面的划分逻辑，落盘后再独立验一遍。
    # 成本极低（一次 groupby），但如果哪天有人改了 group_split，这里能立刻拦住。
    leak = df.groupby("video_id")["split"].nunique()
    if (leak > 1).any():
        raise RuntimeError("数据泄漏：检测到同一视频跨 split 出现！")
    log.info("数据泄漏自检通过 (OK)  manifest → %s", _rel(manifest_path))
    return df, label_map
