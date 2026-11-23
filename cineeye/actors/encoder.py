"""演员专用图片编码器：把「脸部裁剪图」编码成 L2 归一化特征向量。

用途：演员识别与电影识别是两件事，特征需求也不同——
    电影识别看场景/色调/构图，演员识别看五官/脸型。本模块提供一条
    独立的编码通道，不再强行复用电影图库的编码器：

        actors.encoder = imagenet : 用 ImageNet 预训练主干（默认，离线可用，
                                    不依赖 best.pt；适合只有演员数据、还没训电影模型的场景）
        actors.encoder = movie    : 复用 outputs/checkpoints/best.pt 的冻结主干
                                    （主干本身也是 ImageNet 权重，差别主要在"是否存在"）
        actors.encoder = auto     : best.pt 存在则 movie，否则 imagenet

被谁使用：actors/train.py（抽特征训练分类头）、actors/recognizer.py
    （建检索库与在线识别）。两者必须用同一 encoder 配置，否则向量空间不一致。
依赖：torch / torchvision / cv2 / PIL；models/cnn.py（主干构建）。
运维要点（改这里之前必须知道的事）：
    1. 特征向量已做 L2 归一化，点积即余弦相似度；
    2. 换 backbone / image_size / encoder 模式都会改变向量空间，已建好的
       检索库与 actor_head.pt 必须重训（fingerprint 用于检测这种不一致）；
    3. 本模块不写任何文件，纯粹是"图 → 向量"；
    4. GPU 机器上自动走 CUDA（RTX 5070 需 cu128 wheel + torch>=2.7）。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from ..config import ROOT, get_value
from ..data.dataset import IMAGENET_MEAN, IMAGENET_STD
from ..device import resolve_device
from ..models.cnn import build_backbone
from ..utils import get_logger

log = get_logger("cineeye.actors.encoder")


class _BackboneEncoder:
    """ImageNet 预训练主干 → 特征向量的最简实现（内部使用）。

    与 gallery.ImageEncoder 的区别：不加载 best.pt，也不需要分类头，
    因此"只有演员数据"时也能工作；所有参数来自 actors.* 配置。
    """

    def __init__(self, backbone: str, image_size: int, batch_size: int, device: torch.device):
        self.backbone_name = backbone
        self.image_size = int(image_size)
        self.batch_size = int(batch_size)
        self.device = device
        model, dim, used = build_backbone(backbone, pretrained=True)
        if not used:
            # build_backbone 内部已打过 warning；这里再明确提示影响
            log.warning("演员编码器未能加载预训练权重，识别效果会明显下降")
        model.eval()
        self.model = model.to(device)
        self.feature_dim = int(dim)
        self.tf = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def encode(self, images: list[np.ndarray]) -> np.ndarray:
        """BGR 图列表 → (N,D) L2 归一化特征。"""
        if not images:
            return np.zeros((0, self.feature_dim), dtype=np.float32)
        tensors = []
        for img in images:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensors.append(self.tf(Image.fromarray(rgb)))
        out: list[np.ndarray] = []
        with torch.no_grad():  # 推理模式，不建计算图
            for i in range(0, len(tensors), self.batch_size):
                batch = torch.stack(tensors[i:i + self.batch_size]).to(self.device)
                out.append(self.model(batch).cpu().numpy().astype(np.float32))
        X = np.concatenate(out, axis=0)
        # L2 归一化：之后用点积代替余弦相似度，且特征幅值不影响分类头
        return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


class ActorEncoder:
    """演员编码器门面：按配置选择 imagenet / movie 两种实现。

    Args:
        cfg: 全局配置，读取 actors.encoder / backbone / image_size / batch_size。
        device: 显式设备（"cuda" / "cpu" / torch.device）；None 时按 train.device 自动探测。
        checkpoint: movie 模式下的权重路径，缺省用 gallery.checkpoint。

    Raises:
        FileNotFoundError: encoder=movie 但指定的 best.pt 不存在。
    """

    def __init__(self, cfg: dict | None = None, device: str | torch.device | None = None,
                 checkpoint: str | Path | None = None):
        cfg = cfg or {}
        self.cfg = cfg
        mode = str(get_value(cfg, "actors.encoder", "imagenet")).lower()
        backbone = str(get_value(cfg, "actors.backbone", "resnet18"))
        self.image_size = int(get_value(cfg, "actors.image_size", 224))
        batch_size = int(get_value(cfg, "actors.batch_size", 32))

        # auto：优先用已训练的电影主干（如果存在），否则退回 ImageNet 主干
        if mode == "auto":
            ckpt = Path(checkpoint or get_value(cfg, "gallery.checkpoint",
                                                "outputs/checkpoints/best.pt"))
            if not ckpt.is_absolute():
                ckpt = ROOT / ckpt
            mode = "movie" if ckpt.exists() else "imagenet"
        if mode not in ("imagenet", "movie"):
            log.warning("未知 actors.encoder=%s，回退 imagenet", mode)
            mode = "imagenet"
        self.mode = mode

        if isinstance(device, torch.device):
            dev = device
        elif device:
            dev = resolve_device(str(device))
        else:
            dev = resolve_device(str(get_value(cfg, "train.device", "auto")),
                                 int(get_value(cfg, "train.cpu_threads", 8)))
        self.device = dev

        if mode == "movie":
            # 延迟导入：只有电影模式才依赖 gallery.encoder / best.pt
            from ..gallery.encoder import ImageEncoder
            self._impl = ImageEncoder(checkpoint=checkpoint, cfg=cfg, device=dev)
            self.backbone_name = self._impl.backbone_name
            self.feature_dim = self._impl.feature_dim
        else:
            self._impl = _BackboneEncoder(backbone, self.image_size, batch_size, dev)
            self.backbone_name = backbone
            self.feature_dim = self._impl.feature_dim
        log.info("演员编码器就绪：%s / %s / %d 维 / %s",
                 self.mode, self.backbone_name, self.feature_dim, self.device)

    def encode(self, images: list[np.ndarray] | np.ndarray, *, is_bgr: bool = True) -> np.ndarray:
        """编码一组脸部图；单张 3 维 ndarray 会自动包成 batch。

        Args:
            images: BGR ndarray 或 ndarray 列表。
            is_bgr: 输入是否为 BGR（cv2 读图默认是）。

        Returns:
            (N, feature_dim) float32，每行 L2 范数为 1。
        """
        if isinstance(images, np.ndarray) and images.ndim == 3:
            images = [images]
        if not images:
            return np.zeros((0, self.feature_dim), dtype=np.float32)
        if not is_bgr and self.mode == "imagenet":
            # _BackboneEncoder 内部假定 BGR；RGB 输入先转回来，保持调用契约统一
            images = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in images]
        return self._impl.encode(images)

    def encode_paths(self, paths: list[str | Path]) -> np.ndarray:
        """按路径批量读图并编码（读不出的单张跳过并告警，不拖垮整批）。"""
        imgs = []
        for p in paths:
            q = Path(p)
            if not q.is_absolute():
                q = ROOT / q
            img = cv2.imread(str(q))
            if img is None:
                log.warning("跳过无法读取的图片：%s", q)
                continue
            imgs.append(img)
        return self.encode(imgs)

    def fingerprint(self) -> str:
        """编码器指纹（12 位）：模式 + 主干 + 维度 + 输入分辨率。

        检索库与分类头都会记录该指纹；指纹不一致说明旧向量不可用，
        必须重建索引/重训，避免"用 A 的向量查 B 的库"这种静默错误。
        """
        raw = f"actor:{self.mode}:{self.backbone_name}:{self.feature_dim}:{self.image_size}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]
