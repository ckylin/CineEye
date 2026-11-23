"""开放集图库检索：统一特征编码 + 向量检索。

用途：对外导出 ImageEncoder（图片 → 512 维向量）与 GalleryIndex
    （向量落盘 + 余弦检索），是 identify.py 的开放集识别通道。
被谁使用：identify.py、索引构建脚本、serve.py。
依赖：torch/torchvision/cv2/numpy；权重 outputs/checkpoints/best.pt。
运维要点：embeddings.npy 设备无关，换 GPU 无需重建索引；换 checkpoint
    或分辨率后必须 rebuild（见 encoder.fingerprint）。
"""

from .encoder import ImageEncoder
from .index import GalleryIndex

__all__ = ["ImageEncoder", "GalleryIndex"]
