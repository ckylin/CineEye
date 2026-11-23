"""人脸检测：OpenCV Haar 级联，随 opencv 安装、零下载、CPU 毫秒级。

用途：在图片里框出人脸并裁出最大人脸，作为演员比对的输入区域。
被谁使用：actors/recognizer.py（ActorRecognizer 的检测环节）。
依赖：仅 OpenCV（级联分类器数据随包分发，无需额外下载模型）。
运维要点（改这里之前必须知道的事）：
    1. Haar 级联是 2001 年的老算法，局限明显：只适合正面人脸，侧脸/
       大角度/遮挡/逆光容易漏检或误检——它不是生产级人脸识别方案；
    2. 当前检测置信度恒为 1.0（Haar 不输出概率），上层不要把
       FaceBox.confidence 当真值使用；
    3. min_face_ratio 过滤掉过小的人脸（海报远景处的人头），避免噪声；
    4. 若识别准确率不达标，升级路径是换 DNN/InsightFace 检测 +
       ArcFace 特征（FaceEncoderStub 预留了接口），上层不用改。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..config import get_value
from ..utils import get_logger

log = get_logger("cineeye.actors.face")


@dataclass
class FaceBox:
    """一个人脸框（像素坐标）。

    Attributes:
        x / y / w / h: 左上角坐标与宽高（像素）。
        confidence: 检测置信度；Haar 不输出概率，恒为 1.0 占位。
    """

    x: int
    y: int
    w: int
    h: int
    confidence: float = 0.0

    def crop(self, image: np.ndarray, pad: float = 0.1) -> np.ndarray:
        """按框裁剪人脸区域，四周留 pad 比例的余量（保留发型/轮廓信息）。

        Args:
            image: 原图（BGR ndarray）。
            pad: 四周外扩比例（相对人脸宽高）。

        Returns:
            裁剪后的子图（不会越界，边界自动钳制）。
        """
        h, w = image.shape[:2]
        px, py = int(self.w * pad), int(self.h * pad)
        x1, y1 = max(0, self.x - px), max(0, self.y - py)
        x2, y2 = min(w, self.x + self.w + px), min(h, self.y + self.h + py)
        return image[y1:y2, x1:x2]


class FaceDetector:
    """Haar 级联人脸检测器。

    Args:
        cfg: 全局配置，读取 actors.cascade 与 actors.min_face_ratio。
        cascade: 显式指定级联文件名或路径；缺省用 OpenCV 自带的
            haarcascade_frontalface_default.xml。

    Raises:
        RuntimeError: 级联文件加载失败（路径错误或文件损坏）。
    """

    def __init__(self, cfg: dict | None = None, cascade: str | None = None):
        cfg = cfg or {}
        name = cascade or get_value(cfg, "actors.cascade", "haarcascade_frontalface_default.xml")
        # 不含路径分隔符视为 OpenCV 内置级联名，自动拼官方数据目录
        path = str(cv2.data.haarcascades) + name if "/" not in name and "\\" not in name else name
        self.cascade = cv2.CascadeClassifier(path)
        if self.cascade.empty():
            raise RuntimeError(f"人脸级联加载失败：{path}")
        self.min_ratio = float(get_value(cfg, "actors.min_face_ratio", 0.03))
        self.scale_factor = 1.1      # 图像金字塔步长：越小越细但越慢
        self.min_neighbors = 3       # 邻居判定阈值：越大越严格（误检少、漏检多）

    def detect(self, image: np.ndarray) -> list[FaceBox]:
        """输入 BGR 图，返回人脸框列表（按面积降序）。

        Args:
            image: BGR ndarray（可为 None/空图，直接返回空列表）。

        Returns:
            FaceBox 列表，最大的人脸排最前；无检出返回空列表。
        """
        if image is None or image.size == 0:
            return []
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # 直方图均衡：提升逆光/暗光场景的检出率
        rects = self.cascade.detectMultiScale(
            gray, scaleFactor=self.scale_factor, minNeighbors=self.min_neighbors,
            minSize=(max(24, int(w * 0.05)), max(24, int(h * 0.05))),  # 过滤过小的远距人脸
        )
        if len(rects) == 0:
            return []
        area = float(h * w)
        # 面积占比过滤：海报远景人头等噪声不参与后续比对
        boxes = [FaceBox(int(x), int(y), int(bw), int(bh), 1.0)
                 for (x, y, bw, bh) in rects if (bw * bh) / area >= self.min_ratio]
        return sorted(boxes, key=lambda b: -(b.w * b.h))

    def largest_face(self, image: np.ndarray, pad: float = 0.15) -> np.ndarray | None:
        """检出并裁出最大的人脸区域（海报/剧照通常主角占最大人脸）。

        Args:
            image: BGR ndarray。
            pad: 裁剪时四周外扩比例。

        Returns:
            最大人脸的裁剪子图；无人脸返回 None（上层据此决定是否回退整图）。
        """
        boxes = self.detect(image)
        return boxes[0].crop(image, pad=pad) if boxes else None


class FaceEncoderStub:
    """预留接口：未来接入 InsightFace/ArcFace 时实现 encode(faces)->(N,D)。

    上层（recognizer.py）只依赖"输入人脸列表、输出归一化向量矩阵"这一
    约定，替换实现不需要改动比对流程。
    """

    def encode(self, faces: list[np.ndarray]) -> np.ndarray:
        """把人脸图列表编码为特征矩阵；预留接口，当前未实现。

        Args:
            faces: 人脸裁剪图列表。

        Returns:
            (N, D) 特征矩阵（实现时应做 L2 归一化）。

        Raises:
            NotImplementedError: 当前实现恒抛出，提示接入专用模型。
        """
        raise NotImplementedError("请实现专用人脸编码器")
