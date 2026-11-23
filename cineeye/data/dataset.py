"""帧数据集、图像增强与 DataLoader。

用途：把 manifest.csv 变成一个可迭代的 ``(image_tensor, label_idx, video_id)``
数据流。训练、评估、特征抽取三条链路共用这里的数据集与增强定义——
保证「训练时的预处理」与「推理时的预处理」严格一致。

被谁使用：
    - ``cineeye/features.py``（冻结主干抽特征，train=False）；
    - ``cineeye/train.py``（feature / finetune 两种模式）；
    - ``cineeye/evaluate.py``（评估，train=False）；
    - ``tests/test_pipeline.py``（单测增强与数据集）。

依赖：torch、torchvision、PIL、pandas、numpy、cv2（仅推理数据集里用到）。

运维要点：
    1. 本机 ``/dev/shm`` 只有 10MB，多进程 DataLoader 会 Bus error，
       所以 :func:`_safe_num_workers` 会自动把 ``num_workers`` 压到 0。
       物理机上可以把 ``train.num_workers`` 调到 2~4，代码会自行判断。
    2. 本机无 GPU，``pin_memory`` 自动为 False；有 CUDA（含 RTX 5070 /
       Blackwell sm_120）时自动为 True，无需改配置。
    3. 增强策略改动会让「训练分布」变化，但**不影响**特征缓存签名，
       改完建议重跑特征抽取或至少重新训练分类头。
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from ..config import ROOT, get_value
from ..config import path as cfg_path
from ..utils import get_logger, load_json

log = get_logger()

# ImageNet 统计量，与 torchvision 预训练权重配套
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_manifest(cfg: dict) -> tuple[pd.DataFrame, dict[str, int]]:
    """读取清单与标签映射；清单里的路径是相对项目根目录的。

    Args:
        cfg: 全局配置，读取 ``data.manifest``（CSV）与 ``data.label_map``（JSON）。

    Returns:
        ``(df, label_map)``。``frame_index`` 强制按字符串读入，避免
        ``000001`` 被解析成整数 1 而丢失前导零（重写清单时会改变文件名语义）。

    Raises:
        FileNotFoundError: manifest 不存在，提示先跑 02 脚本。
        json.JSONDecodeError: label_map 损坏（由 ``load_json`` 抛出）。
    """
    manifest_path = cfg_path(cfg, "data.manifest")
    if not manifest_path.exists():
        raise FileNotFoundError(f"清单不存在：{manifest_path}，请先运行 02_build_manifest.py")
    df = pd.read_csv(manifest_path, dtype={"frame_index": str})
    label_map = load_json(cfg_path(cfg, "data.label_map"))
    # 只抽查前 50 行而不是全量 stat：全量 os.stat 在几万帧时会明显拖慢启动，
    # 而「路径整体失效」这种系统性问题抽 50 行足够发现。
    missing = [p for p in df["frame_path"].head(50) if not (ROOT / p).exists()]
    if missing:
        log.warning("抽查发现 %d 张图片路径不存在（可能移动过 data/ 目录）：%s", len(missing), missing[:3])
    return df, label_map


def build_transforms(cfg: dict, train: bool = False):
    """构造图像预处理 / 增强流水线。

    验证与推理（``train=False``）只做 Resize + Normalize，必须完全确定，
    否则同一张图两次评估结果会不同，指标不可比。

    训练（``train=True``）的增强策略是「保守增强」：
    - 先用 Resize 放大到 1.15 倍再 RandomResizedCrop，等价于在略微裁切的
      同时保留几乎全部构图（scale 下限 0.85、长宽比只允许 0.95~1.05）；
    - 水平翻转默认关闭：电影画面翻转会镜像字幕与场景文字，
      而这些恰恰是强判别线索，翻转反而制造了不可能出现在测试集的样本；
    - ColorJitter 只做轻度扰动（默认 0.15），用来抵抗不同片源的调色/亮度差异。

    Args:
        cfg: 全局配置，读取 ``data.image_size`` 与 ``data.augmentation.*``。
        train: True 返回训练增强流水线，False 返回确定性流水线。

    Returns:
        ``torchvision.transforms.Compose``，输出归一化后的 float32 Tensor。
        ImageNet 均值方差与 torchvision 预训练权重配套，不能随意改。

    Raises:
        KeyError: ``data.augmentation.random_resized_crop_scale`` 不是长度为 2
            的可迭代对象时（由 ``tuple()`` 抛出）。
    """
    size = int(get_value(cfg, "data.image_size", 224))
    base = [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if not train:
        # 电影画面不做 CenterCrop：字幕和边缘构图都可能是判别线索
        return transforms.Compose([transforms.Resize((size, size)), *base])

    aug = []
    hflip = float(get_value(cfg, "data.augmentation.random_hflip", 0.0))
    jitter = float(get_value(cfg, "data.augmentation.color_jitter", 0.15))
    scale = get_value(cfg, "data.augmentation.random_resized_crop_scale", [0.85, 1.0])
    # 先放大再随机裁：等价于「轻微 zoom + 平移」，比直接 RandomCrop 更抗构图差异
    aug.append(transforms.Resize((int(size * 1.15), int(size * 1.15))))
    aug.append(transforms.RandomResizedCrop(size, scale=tuple(scale), ratio=(0.95, 1.05)))
    if hflip > 0:
        aug.append(transforms.RandomHorizontalFlip(hflip))
    if jitter > 0:
        aug.append(transforms.ColorJitter(brightness=jitter, contrast=jitter, saturation=jitter, hue=0.03))
    return transforms.Compose([*aug, *base])


class FrameDataset(Dataset):
    """按清单读取帧图片。损坏的图片用中性灰占位，避免整个训练中断。

    设计取舍：读图失败**不抛异常**。训练集里只要有一张坏图，
    抛出就会让整个 epoch 崩掉，而重新跑一次代价很高；
    用 128 灰图占位相当于给模型一个「无信息样本」，对收敛影响可忽略，
    同时 warning 只打一次，避免刷屏。

    ``__getitem__`` 额外返回 ``video_id``，是为了让上层（评估、置信度分析）
    能按「整段视频」聚合预测结果，而不是按单帧。
    """

    def __init__(self, df: pd.DataFrame, transform=None, root: Path = ROOT):
        """保存清单与预处理，不做任何 I/O（构造必须是毫秒级的）。

        Args:
            df: 至少含 ``frame_path`` 与 ``label_idx`` 两列的 DataFrame。
            transform: torchvision 变换；None 时返回 PIL Image（单测用）。
            root: 相对路径的基准目录，默认项目根目录；便于单测指向临时目录。
        """
        # reset_index 保证 iloc[idx] 与 DataLoader 的 0..N-1 序号严格对应，
        # 否则调用方传入的 df 若是切片（index 不连续）会取错行。
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.root = Path(root)
        # 只报一次坏图，防止一个坏目录导致几千行 warning 把日志冲掉
        self._bad_reported = False

    def __len__(self) -> int:
        """样本数（= 清单行数）。"""
        return len(self.df)

    def __getitem__(self, idx: int):
        """取第 idx 个样本。

        Args:
            idx: 0-based 序号。

        Returns:
            ``(image, label_idx, video_id)``：
                image 为 transform 后的 Tensor（或 PIL Image）；
                label_idx 为 int；video_id 为 str。

        Raises:
            不抛异常。任何读图/解码失败都会退化为灰色占位图。
        """
        row = self.df.iloc[idx]
        p = Path(row["frame_path"])
        if not p.is_absolute():
            p = self.root / p
        try:
            # with 语句确保文件句柄立刻释放：DataLoader 反复取图时
            # 句柄泄漏在容器里很快会撞到 ulimit -n 上限。
            with Image.open(p) as im:
                # 统一转 RGB：清单里可能混有 PNG 灰度图/带 alpha 的图
                image = im.convert("RGB")
        except Exception as e:  # noqa: BLE001
            if not self._bad_reported:
                log.warning("图片读取失败，用空白帧占位：%s (%s)", p, e)
                self._bad_reported = True
            # 128 灰 ≈ ImageNet 归一化后接近 0，相当于「无信息」输入
            image = Image.new("RGB", (224, 224), (128, 128, 128))
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["label_idx"]), str(row["video_id"])


class ImageListDataset(Dataset):
    """推理用：直接吃内存里的 BGR ndarray 列表（不落盘）。

    在线识别时视频帧已经在内存里（:func:`sample_frames_from_video`），
    没必要再编码成 JPEG 写盘。OpenCV 给的是 BGR，PIL/torchvision 期望 RGB，
    所以必须先 ``cvtColor``，否则通道顺序错了会导致预测结果显著偏移
    ——这是这类「训练用 PIL、推理用 cv2」项目最常见的隐蔽 bug。

    预处理与训练/评估严格同构（Resize + Normalize，无增强）。
    """

    def __init__(self, frames: list[np.ndarray], size: int = 224):
        """Args:
            frames: BGR 帧列表，元素形状 ``(H, W, 3)``，uint8。
            size: 目标分辨率，应与 ``data.image_size`` 保持一致。
        """
        self.frames = frames
        self.tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self) -> int:
        """帧数。"""
        return len(self.frames)

    def __getitem__(self, idx: int):
        """取第 idx 帧并转成模型输入 Tensor。

        Args:
            idx: 0-based 帧序号。

        Returns:
            ``(3, size, size)`` 的归一化 float32 Tensor。

        Raises:
            IndexError: idx 越界。
            ValueError: 帧不是 3 通道图像时（cv2.cvtColor 抛出）。
        """
        # cv2 延迟导入：训练/评估路径不必加载 OpenCV，节省启动时间与内存
        import cv2

        rgb = cv2.cvtColor(self.frames[idx], cv2.COLOR_BGR2RGB)
        return self.tf(Image.fromarray(rgb))


def _safe_num_workers(requested: int) -> int:
    """容器里 /dev/shm 往往只有几 MB，多进程 DataLoader 会直接 Bus error。

    共享内存不足时退回单进程；CPU 训练下小数据集本来就不需要多进程解码。

    原理：``num_workers>0`` 时 PyTorch 会把每个 batch 的 Tensor 放进
    ``/dev/shm`` 由子进程传给主进程。本机 ``/dev/shm`` 仅 10MB，
    一个 ``64x3x224x224`` 的 float32 batch 就要约 38MB，第一次传输就会
    SIGBUS 崩溃，而报错只有一句 "Bus error"，完全看不出与共享内存有关，
    排查成本极高。因此这里宁可保守地退回单进程。

    阈值 256MB 是经验值：低于它时连稍大的 batch 都不安全。

    Args:
        requested: 配置里的 ``train.num_workers``；``<=0`` 表示调用方已明确要单进程。

    Returns:
        实际可用的 worker 数：0 或原值。

    Raises:
        无。``shutil.disk_usage`` 失败时静默放行（例如 /dev/shm 不存在或权限不足）。
    """
    if requested <= 0:
        return 0
    try:
        shm = Path("/dev/shm")
        if shm.exists():
            free_mb = shutil.disk_usage(shm).free / 1024 ** 2
            if free_mb < 256:
                log.warning("/dev/shm 仅 %.0f MiB，DataLoader 强制 num_workers=0（避免共享内存 Bus error）；"
                            "物理机可设 train.num_workers=2~4", free_mb)
                return 0
    except Exception:  # noqa: BLE001
        pass
    return requested


def build_dataloader(
    df: pd.DataFrame,
    cfg: dict,
    train: bool = False,
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> DataLoader:
    """构造 DataLoader（训练 / 评估 / 特征抽取共用）。

    Args:
        df: 已切好 split 的清单（调用方负责只传单个 split）。
        cfg: 全局配置，读取 ``train.batch_size`` / ``train.num_workers`` /
            ``train.balanced_sampler``。
        train: True 时启用训练增强与（可选）类别平衡采样。
        batch_size: 覆盖配置里的 batch size。
        num_workers: 覆盖配置里的 worker 数；实际值还要经
            :func:`_safe_num_workers` 按 /dev/shm 余量降级。

    Returns:
        ``torch.utils.data.DataLoader``，产出 ``(image, label_idx, video_id)``。

    Raises:
        KeyError: df 缺少 ``label_idx`` 列且启用了平衡采样时（由 pandas 抛出）。
    """
    tf = build_transforms(cfg, train=train)
    ds = FrameDataset(df, transform=tf)
    bs = int(batch_size or get_value(cfg, "train.batch_size", 64))
    # 注意：即使配置写 4，只要 /dev/shm < 256MB 也会被压回 0（本机正是如此）
    workers = _safe_num_workers(
        int(num_workers if num_workers is not None else get_value(cfg, "train.num_workers", 0))
    )

    sampler = None
    shuffle = train
    if train and bool(get_value(cfg, "train.balanced_sampler", True)) and len(df) > 0:
        # 类别平衡采样：让每类被抽中的总期望次数相同。
        # 电影帧数天然不均（热门片 200 帧、冷门片可能只有 30 帧），
        # 不做平衡模型会偏向帧多的类别，macro-F1 明显偏低。
        counts = df["label_idx"].value_counts()
        # .copy() 保证可写：WeightedRandomSampler 会把数组包成 tensor，只读数组会告警
        weights = df["label_idx"].map(lambda c: 1.0 / counts[c]).to_numpy(dtype=np.float64).copy()
        # replacement=True 才能从少数类里反复取样到与多数类等量
        sampler = WeightedRandomSampler(weights, num_samples=len(df), replacement=True)
        # sampler 与 shuffle 互斥，同时给会让 DataLoader 直接报错
        shuffle = False

    return DataLoader(
        ds,
        batch_size=bs,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=workers,
        # pin_memory：把 batch 放进锁页内存，GPU 异步拷贝可提速 10~20%。
        # 本机无 GPU 时自动 False；插上 RTX 5070（sm_120）并装好 cu128 版 torch 后自动生效。
        pin_memory=torch.cuda.is_available(),
        # 只有真多进程时让 worker 常驻才有意义，否则 torch 会告警
        persistent_workers=workers > 0,
        # 最后一个小 batch 不舍弃：评估时丢样本会让指标有偏
        drop_last=False,
    )
