"""影瞳推理：单张图片、批量图片、整段视频（多帧聚合 + 未知影片拒识）。

用途：
    加载 best.pt，把图片/视频帧转成模型输入并输出“最像哪部电影 + 置信度”，
    同时给出 top-k 候选、是否低置信度（unknown）以及视频的帧间一致性。
    视频不是逐帧各报一个答案，而是先逐帧算概率、再聚合成一个电影级结论。

被谁使用：
    - ``scripts/05_predict.py``：命令行入口（支持 --json 供外部程序消费）
    - ``cineeye/serve.py``：HTTP 服务 /predict 与 /predict/video
    - ``cineeye/identify.py``：图库与演员扩展识别复用同一套模型加载逻辑
    - ``cineeye/evaluate.py``：特征缓存缺失时回退到这里的整图推理

依赖：
    opencv（读写图片/视频）、numpy、torch、Pillow；torchvision 只在预处理时延迟导入。

运维要点：
    - 推理全程只在内存中处理帧，不写临时文件，所以 /tmp 只有 10MB 也不影响。
    - checkpoint 自带训练时的 config，加载时优先使用它，避免推理与训练的
      图像尺寸/预处理不一致导致精度骤降。
    - 建网络固定 pretrained=False：权重全部来自 checkpoint，部署无需联网。
    - RTX 5070（Blackwell sm_120）需 cu128 wheel + torch>=2.7；auto 会先做
      CUDA 内核预检（真跑一次矩阵乘），失败自动回退 CPU，不会中途崩溃。
    - best.pt 字段契约：model_state / label_map / num_classes / backbone / config / metrics。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from .config import ROOT, get_value, load_config
from .config import path as cfg_path
from .data.dataset import IMAGENET_MEAN, IMAGENET_STD
from .data.extract_frames import sample_frames_from_video
from .device import resolve_device
from .models.cnn import CineEyeNet
from .utils import get_logger

log = get_logger()


def _load_checkpoint(ckpt_path: str | Path) -> dict:
    """读取 checkpoint，统一映射到 CPU，并兼容新旧 torch 的 weights_only 参数。

    先 map_location="cpu" 有两个好处：无 GPU 的机器也能加载；不会因为保存时
    用的是 cuda:0 而在别的设备上反序列化失败。真正的设备搬运交给调用方。

    Args:
        ckpt_path: best.pt 路径。

    Returns:
        dict: 含 model_state / label_map / num_classes / backbone / config / metrics 等。

    Raises:
        FileNotFoundError: 路径不存在（torch.load 抛出）。
        RuntimeError / pickle.UnpicklingError: 文件损坏或不是 PyTorch checkpoint。
    """
    try:
        # weights_only=False 允许反序列化 numpy 标量与 dict，checkpoint 是本项目自己产出的，可信
        return torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:  # 旧版 torch 不认识 weights_only
        return torch.load(ckpt_path, map_location="cpu")


class Predictor:
    """加载 best.pt，提供图片级与视频级预测。

    构建网络时用 ``pretrained=False``：权重全部来自 checkpoint，
    因此部署时不需要联网、也不需要 torchvision 预训练权重。

    典型用法：构造一次后反复调用；模型已 eval + 主干冻结，所有推理都在 no_grad 下进行。

    Attributes:
        cfg: 实际生效的配置（优先取 checkpoint 内保存的训练配置）。
        label_map: 片名 → 类别索引；idx2label 为反向映射。
        num_classes: 类别总数。
        mode: 训练模式（feature / finetune），仅用于日志与排查。
        device: 最终选中的 torch 设备。
        model: 已加载权重、处于 eval 的 CineEyeNet。
        top_k: 返回多少个候选。
        unknown_threshold: 拒识阈值，<=0 表示关闭。
        aggregation: 视频默认聚合方式（mean_prob / log_prob / majority）。
        num_frames: 视频默认抽帧数。
        image_size: 输入分辨率，必须与训练一致。
    """

    def __init__(self, checkpoint: str | Path | None = None, cfg: dict | None = None,
                 device: str | torch.device | None = None):
        """加载 checkpoint 并准备推理环境。

        Args:
            checkpoint: best.pt 路径；None 时读配置项 inference.checkpoint。
            cfg: 配置字典；None 时 load_config()。
                注意：checkpoint 内保存的 config 优先级更高，见下方赋值。
            device: 显式设备（str 或 torch.device）；None 时按配置 train.device 解析。

        Returns:
            None。

        Raises:
            FileNotFoundError: checkpoint 不存在。
            RuntimeError: state_dict 与网络结构不匹配（backbone/dropout/head_hidden 被改过）。
        """
        if cfg is None:
            cfg = load_config()
        ckpt_path = Path(checkpoint or cfg_path(cfg, "inference.checkpoint"))
        if not ckpt_path.exists():
            raise FileNotFoundError(f"找不到模型：{ckpt_path}，请先运行训练脚本")
        ckpt = _load_checkpoint(ckpt_path)
        # checkpoint 里保存了训练当时的配置，优先用它，避免增强/分辨率不一致
        self.cfg = ckpt.get("config") or cfg
        self.label_map: dict[str, int] = ckpt["label_map"]
        self.idx2label = {v: k for k, v in self.label_map.items()}
        self.num_classes = int(ckpt["num_classes"])
        self.mode = ckpt.get("mode", "feature")

        # 结构必须与训练时完全一致才能 load_state_dict：backbone/dropout/head_hidden 都从配置取
        model = CineEyeNet(
            backbone=ckpt["backbone"], num_classes=self.num_classes,
            pretrained=False,
            dropout=float(get_value(self.cfg, "model.dropout", 0.2)),
            head_hidden=int(get_value(self.cfg, "model.head_hidden", 0)),
        )
        model.load_state_dict(ckpt["model_state"])
        # 推理不需要主干梯度；freeze 后 train() 也会让 backbone 保持 eval，BN 统计不漂移
        model.freeze_backbone(True)
        model.eval()

        if isinstance(device, torch.device):
            # 调用方已解析过设备，直接使用，避免重复探测 CUDA
            self.device = device
        else:
            # 设备选择链路：
            #   显式字符串（cpu/cuda/cuda:N/mps/xpu）→ resolve_device；
            #   auto 时依次尝试 CUDA（先 cuda_kernel_ok 真跑一次 8×8 矩阵乘，
            #   排除 sm_120 等“is_available 为真但无内核镜像”的陷阱）→ MPS → XPU → CPU。
            # 无论如何都会返回一个能跑的设备，不会因为没 GPU 直接失败。
            self.device = resolve_device(
                str(device or get_value(self.cfg, "train.device", "auto")),
                int(get_value(self.cfg, "train.cpu_threads", 8)),
            )
        self.model = model.to(self.device)
        self.top_k = int(get_value(self.cfg, "inference.top_k", 5))
        # 0 表示关闭拒识；>0 时最高概率低于该值就标记为「未知影片」
        self.unknown_threshold = float(get_value(self.cfg, "inference.unknown_threshold", 0.0))
        self.aggregation = str(get_value(self.cfg, "inference.aggregation", "mean_prob"))
        self.num_frames = int(get_value(self.cfg, "inference.num_frames", 16))
        self.image_size = int(get_value(self.cfg, "data.image_size", 224))
        log.info("模型已加载：%s | 主干=%s | 类别=%d | 模式=%s | 设备=%s",
                 ckpt_path.name, ckpt["backbone"], self.num_classes, self.mode, self.device)

    # ---------------------------------------------------------------- 底层
    def _transform(self, rgb_or_bgr: np.ndarray, is_bgr: bool = True) -> torch.Tensor:
        """把 OpenCV 风格的图像数组转成模型输入张量 (3, H, W)。

        预处理必须与训练/评估的验证变换一致：只 Resize 到方形，不做 CenterCrop、
        不做随机增强——电影画面的字幕和边缘构图本身可能就是判别线索，裁掉会丢信息。

        Args:
            rgb_or_bgr: H×W×3 的 uint8 图像。
            is_bgr: True 表示输入来自 cv2（BGR），会先转 RGB；
                ImageNet 预训练权重按 RGB 训练，通道顺序错了精度会明显下降。

        Returns:
            torch.Tensor: float32，已按 ImageNet 均值/方差归一化，形状 (3, image_size, image_size)。

        Raises:
            ValueError: 通道数不是 3 时 cv2.cvtColor / PIL 会报错。
        """
        from torchvision import transforms  # 延迟导入：服务常驻时减少启动依赖

        if is_bgr:
            rgb_or_bgr = cv2.cvtColor(rgb_or_bgr, cv2.COLOR_BGR2RGB)
        tf = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),  # HWC uint8 [0,255] → CHW float [0,1]
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        return tf(Image.fromarray(rgb_or_bgr))

    @torch.no_grad()
    def predict_proba(self, tensors: torch.Tensor) -> np.ndarray:
        """(N,3,H,W) → (N,C) 概率。

        按 32 张小批前向，控制 CPU 峰值内存；空输入返回 (0,C) 而不是报错，
        这样批量接口在“所有图片都读失败”时仍能正常返回空列表。

        Args:
            tensors: 已预处理的批量张量。

        Returns:
            np.ndarray: (N, num_classes) float32，每行和为 1。
        """
        self.model.eval()
        out = []
        bs = 32
        for i in range(0, len(tensors), bs):
            logits = self.model(tensors[i:i + bs].to(self.device))
            out.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, self.num_classes), np.float32)

    def _format(self, prob: np.ndarray) -> dict:
        """把一条概率向量整理成对外统一的 JSON 结构。

        unknown 的含义：仅当 unknown_threshold > 0 时启用拒识，
        最高概率低于阈值就标记 unknown=True。此时 label 仍保留 top-1，
        便于人工复查“模型最像哪部片、只是不够自信”，而不是直接丢弃信息。

        Args:
            prob: (C,) 概率向量，通常已归一化。

        Returns:
            dict: label / label_idx / confidence / unknown / top_k。
        """
        k = min(self.top_k, self.num_classes)
        order = np.argsort(prob)[::-1][:k]  # 降序取前 k
        best = int(order[0])
        confidence = float(prob[best])
        return {
            "label": self.idx2label.get(best, str(best)),
            "label_idx": best,
            "confidence": round(confidence, 6),
            # 阈值 <=0 视为关闭拒识；否则低置信度输出「未知影片」提示
            "unknown": bool(self.unknown_threshold > 0 and confidence < self.unknown_threshold),
            "top_k": [
                {"label": self.idx2label.get(int(i), str(i)), "confidence": round(float(prob[i]), 6)}
                for i in order
            ],
        }

    # ---------------------------------------------------------------- 图片
    def predict_image(self, image_path: str | Path) -> dict:
        """单张图片推理。

        路径解析顺序：绝对路径直接用；相对路径先按当前工作目录找，
        找不到再按项目根目录找——这样无论从仓库根还是 scripts/ 下运行都能工作。

        Args:
            image_path: 图片路径。

        Returns:
            dict: _format() 的结果，额外带 source（实际读取的绝对路径）。

        Raises:
            FileNotFoundError: cv2.imread 返回 None（文件不存在或不是可解码的图片）。
        """
        p = Path(image_path)
        if not p.is_absolute() and not p.exists():
            p = ROOT / p
        bgr = cv2.imread(str(p))
        if bgr is None:  # cv2 读失败不抛异常，只返回 None，必须显式检查
            raise FileNotFoundError(f"图片无法读取：{p}")
        # unsqueeze(0) 把单张 (3,H,W) 补成 batch=1
        prob = self.predict_proba(self._transform(bgr).unsqueeze(0))[0]
        result = self._format(prob)
        result["source"] = str(p)
        return result

    def predict_images(self, image_paths: list[str | Path]) -> list[dict]:
        """批量图片推理。

        读不出来的图片会跳过并记 warning，而不是让整批失败；
        返回列表与“成功读取的图片”一一对应，顺序保持输入顺序。

        Args:
            image_paths: 图片路径列表。

        Returns:
            list[dict]: 每项为 _format() 结果 + source。全部失败时返回空列表。

        Raises:
            无显式异常；单张解码失败被吞掉并降级为 warning。
        """
        tensors = []
        valid = []
        for p in image_paths:
            q = Path(p)
            if not q.is_absolute() and not q.exists():
                q = ROOT / q
            bgr = cv2.imread(str(q))
            if bgr is None:
                log.warning("跳过无法读取的图片：%s", q)
                continue
            tensors.append(self._transform(bgr))
            valid.append(str(q))
        if not tensors:
            return []
        probs = self.predict_proba(torch.stack(tensors))  # 一次批量前向，比逐张循环快
        return [{**self._format(pr), "source": src} for pr, src in zip(probs, valid)]

    # ---------------------------------------------------------------- 视频
    def predict_video(
        self,
        video_path: str | Path,
        num_frames: int | None = None,
        aggregation: str | None = None,
    ) -> dict:
        """均匀抽帧 → 逐帧预测 → 聚合，输出电影级结果。

        单帧可能恰好是黑屏或通用镜头；用多帧聚合能显著提升整段视频的稳定性。

        Args:
            video_path: 视频文件路径。
            num_frames: 抽帧数；None 时用配置 inference.num_frames（默认 16）。
            aggregation: 聚合方式；None 时用配置 inference.aggregation。

        Returns:
            dict: predict_frames() 的结果（含 label/confidence/top_k/unknown/
            aggregation/num_frames/frame_agreement/frame_votes/source）。

        Raises:
            FileNotFoundError: 视频无法打开（由 sample_frames_from_video 抛出）。
        """
        # 均匀采样而非从头截取：片头常是黑屏/logo，均匀分布更能覆盖全片风格
        frames = sample_frames_from_video(video_path, num_frames or self.num_frames)
        return self.predict_frames(frames, aggregation=aggregation, source=str(video_path))

    def predict_frames(self, frames: list[np.ndarray], aggregation: str | None = None,
                       source: str = "<frames>") -> dict:
        """对已解码的帧列表做多帧聚合，返回统一结果结构。

        聚合策略：
            mean_prob（默认）：各帧 softmax 概率求算术平均。等价于概率投票，
                对单帧噪声最稳，也保留了“接近但不确定”的信息；
            log_prob：对数概率求平均再 exp（几何平均）。会放大高置信帧的影响，
                结果更“锐”，适合画面质量差异大的片子；
            majority：每帧先 argmax 再按票数归一。最直观但丢弃置信度，
                帧数少（<8）时容易被个别错帧带偏。

        frame_agreement 是最高票类别的占比，用于判断视频内部是否一致：
        接近 1 说明各帧意见统一；很低说明模型在该视频上摇摆，结果应谨慎看待。

        Args:
            frames: BGR 帧数组列表。
            aggregation: 覆盖默认聚合方式；None 时用 self.aggregation。
            source: 结果里标注的来源（视频路径或自定义标识）。

        Returns:
            dict: _format() 的结果，另含 source / aggregation / num_frames /
            frame_agreement / frame_votes。

        Raises:
            RuntimeError: frames 为空时 torch.stack 抛出（调用方应先保证有帧）。
        """
        tensors = torch.stack([self._transform(f) for f in frames])
        probs = self.predict_proba(tensors)
        agg = aggregation or self.aggregation

        if agg == "majority":
            # 每帧一票，再把票数归一成“概率”形状，保证 _format 的接口不变
            votes = probs.argmax(axis=1)
            counts = np.bincount(votes, minlength=self.num_classes).astype(np.float32)
            fused = counts / counts.sum()
        elif agg == "log_prob":
            # 几何平均：log 域求均值再 exp，相当于各帧概率连乘后开 n 次方
            fused = np.exp(np.log(probs + 1e-8).mean(axis=0))  # +1e-8 防 log(0)
            fused = fused / fused.sum()
        else:  # mean_prob
            fused = probs.mean(axis=0)
            fused = fused / fused.sum()  # 每帧已归一，均值天然归一，这里再保险一次

        result = self._format(fused)
        per_frame = probs.argmax(axis=1)
        counts = np.bincount(per_frame, minlength=self.num_classes)
        result.update({
            "source": source,
            "aggregation": agg,
            "num_frames": len(frames),
            # 帧一致性：最高票类别占全部帧的比例，用来提示“模型是否犹豫”
            "frame_agreement": round(float(counts.max() / len(frames)), 4),
            # 只保留出现过票数的类别，避免输出一堆 0
            "frame_votes": {
                self.idx2label.get(i, str(i)): int(c)
                for i, c in enumerate(counts) if c > 0
            },
        })
        return result
