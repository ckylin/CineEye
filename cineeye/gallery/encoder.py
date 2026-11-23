"""统一图片编码器：复用训练好的 ResNet18 主干，输出 512 维 L2 归一化特征。

用途：把任意图片（海报/演员照/视频帧/花絮照）编码成 512 维向量，
    供 GalleryIndex 与 ActorRecognizer 做余弦相似度检索。
被谁使用：gallery/index.py、actors/recognizer.py、identify.py（共享同一实例）。
依赖：torch / torchvision / cv2 / PIL；权重文件 outputs/checkpoints/best.pt
    （由训练脚本产出）。
运维要点（改这里之前必须知道的事）：
    1. 权重加载用 pretrained=False 从 checkpoint 读，全程不联网；
    2. device 解析（历史 bug 修复点）：早期版本曾把 device 写死为 "cpu"，
       导致 GPU 机器上图库/演员编码白白走 CPU。现在缺省时按配置
       train.device（默认 auto）经 resolve_device() 自动探测——上层
       （identify.py）也可显式传入 device 参数统一管理。换 GPU/换 CPU
       不影响任何已落盘产物；
    3. 只用主干、去掉分类头，因此不依赖训练类别数，可编码任意新片库；
    4. fingerprint() 变化时索引必须全量重建（见 index.py）。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from ..config import ROOT, get_value
from ..data.dataset import IMAGENET_MEAN, IMAGENET_STD
from ..device import resolve_device
from ..models.cnn import CineEyeNet
from ..utils import get_logger

log = get_logger("cineeye.gallery")


def _load_ckpt(path: Path) -> dict:
    """加载训练 checkpoint 到 CPU（与最终推理设备无关，之后统一 .to(device)）。

    兼容两种 torch 版本：新版有 weights_only 参数，旧版没有（TypeError 回退）。

    Args:
        path: checkpoint 文件路径。

    Returns:
        checkpoint dict（含 model_state / backbone / feature_dim 等键）。
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class ImageEncoder:
    """把 BGR ndarray / 图片路径编码成 L2 归一化向量。

    Args:
        checkpoint: 权重文件路径；缺省读 cfg 的 gallery.checkpoint
            （默认 outputs/checkpoints/best.pt）。
        cfg: 全局配置 dict。
        device: 显式指定的运行设备（如 "cuda" / "cpu" / torch.device）。
            语义：None 时按 cfg["train.device"]（默认 "auto"）自动探测——
            本机无 GPU 则回退 CPU（多线程加速），有 GPU 则走 CUDA
            （RTX 5070 等新卡需 cu128 wheel + torch>=2.7）。上层统一
            传一次即可，保证编码器与分类器在同一设备上。
    """

    def __init__(self, checkpoint: str | Path | None = None, cfg: dict | None = None,
                 device: str | torch.device | None = None):
        cfg = cfg or {}
        ckpt_path = Path(checkpoint or get_value(cfg, "gallery.checkpoint", "outputs/checkpoints/best.pt"))
        if not ckpt_path.is_absolute():
            ckpt_path = ROOT / ckpt_path
        if not ckpt_path.exists():
            raise FileNotFoundError(f"找不到模型权重：{ckpt_path}")
        ckpt = _load_ckpt(ckpt_path)
        self.backbone_name = ckpt.get("backbone", "resnet18")
        self.feature_dim = int(ckpt.get("feature_dim", 512))
        self.image_size = int(get_value(cfg, "data.image_size", 224))
        self.batch_size = int(get_value(cfg, "gallery.batch_size", 32))

        # pretrained=False：权重全部来自 checkpoint，不触发 torchvision 下载
        model = CineEyeNet(self.backbone_name, num_classes=int(ckpt.get("num_classes", 2)), pretrained=False)
        model.load_state_dict(ckpt["model_state"])
        model.eval()  # 固定推理模式，关闭 dropout/BN 更新，保证向量可复现
        # 与 Predictor 一致：缺省时按配置的 train.device（默认 auto）自动选设备，
        # GPU 机器上图库/演员编码同样走 GPU，不再硬编码 CPU。
        # （历史 bug：这里曾写死 "cpu"，导致 GPU 机器编码极慢，现已修复。）
        if isinstance(device, torch.device):
            self.device = device
        elif device:
            self.device = resolve_device(str(device))
        else:
            self.device = resolve_device(
                str(get_value(cfg, "train.device", "auto")),
                int(get_value(cfg, "train.cpu_threads", 8)),
            )
        self.model = model.to(self.device)

        # 预处理与训练时保持一致：缩放到 image_size、ImageNet 均值方差归一化
        self.tf = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        log.info("图库编码器就绪：%s / %d 维 / %s", self.backbone_name, self.feature_dim, self.device)

    @staticmethod
    def load_image(path: str | Path) -> np.ndarray:
        """读图并返回 BGR ndarray（OpenCV 约定）。

        相对路径先尝试相对当前目录，再回退到项目根目录解析。

        Args:
            path: 图片路径。

        Returns:
            BGR 格式 ndarray。

        Raises:
            FileNotFoundError: 路径不存在或 cv2 读取失败（损坏/不支持格式）。
        """
        p = Path(path)
        if not p.is_absolute() and not p.exists():
            p = ROOT / p
        img = cv2.imread(str(p))
        if img is None:
            raise FileNotFoundError(f"图片无法读取：{p}")
        return img

    @staticmethod
    def to_rgb(bgr: np.ndarray) -> np.ndarray:
        """BGR → RGB（cv2 读图是 BGR，PIL/torch 管线需要 RGB）。

        Args:
            bgr: BGR ndarray。

        Returns:
            RGB ndarray。
        """
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def encode(self, images: list[np.ndarray] | np.ndarray, *, is_bgr: bool = True) -> np.ndarray:
        """(N,H,W,3) → (N,D) L2 归一化特征。单张传 list[img] 即可。

        Args:
            images: 图片数组列表；3 维 ndarray 视为单张自动包装。
            is_bgr: 输入是否为 BGR（cv2 读图默认是）。

        Returns:
            (N, feature_dim) float32 矩阵，每行 L2 范数为 1。
        """
        if isinstance(images, np.ndarray) and images.ndim == 3:
            images = [images]  # 单张图片自动包装成 batch
        if not images:
            return np.zeros((0, self.feature_dim), dtype=np.float32)
        tensors = []
        for img in images:
            rgb = self.to_rgb(img) if is_bgr else img
            tensors.append(self.tf(Image.fromarray(rgb)))
        out: list[np.ndarray] = []
        with torch.no_grad():  # 推理模式，不构建计算图省显存
            for i in range(0, len(tensors), self.batch_size):
                batch = torch.stack(tensors[i:i + self.batch_size]).to(self.device)
                feat = self.model.extract_features(batch).cpu().numpy().astype(np.float32)
                out.append(feat)
        X = np.concatenate(out, axis=0) if out else np.zeros((0, self.feature_dim), np.float32)
        # L2 归一化后点积即余弦相似度（index.py 直接用 X @ q 检索的基础）
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
        return X / norms

    def encode_paths(self, paths: list[str | Path]) -> np.ndarray:
        """按路径批量读图并编码；读不出的图片跳过并告警。

        Args:
            paths: 图片路径列表（相对路径回退项目根解析）。

        Returns:
            与成功读取的图片对应的 (N,D) 归一化特征矩阵。
        """
        imgs = []
        valid: list[Path] = []
        for p in paths:
            q = Path(p)
            if not q.is_absolute() and not q.exists():
                q = ROOT / q
            img = cv2.imread(str(q))
            if img is None:
                log.warning("跳过无法读取的图片：%s", q)
                continue  # 单张损坏不拖垮整批编码
            imgs.append(img)
            valid.append(q)
        return self.encode(imgs)

    def fingerprint(self) -> str:
        """编码器指纹：权重 + 分辨率变化时让索引失效重建。

        index.py 会把它存进 index.meta.json；若与当前指纹不一致，
        说明旧索引的向量不是本编码器产出的，必须全量重建。

        Returns:
            12 位 MD5 前缀字符串。
        """
        import hashlib
        raw = f"{self.backbone_name}:{self.feature_dim}:{self.image_size}".encode()
        return hashlib.md5(raw).hexdigest()[:12]
