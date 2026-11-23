"""基于 torchvision 预训练 CNN 的迁移学习模型。

用途：提供「ImageNet 预训练主干 + 轻量分类头」的模型封装，支持两种工作模式
（模式由 ``model.mode`` 决定，实际切换逻辑在 ``train.py``）：

    feature  —— 冻结主干，只前向一次把固定维度特征缓存到磁盘，
                之后只训练分类头。CPU 上几分钟就能跑完，是默认模式。
    finetune —— 解冻主干端到端微调，backbone 用小学习率（lr*0.1）、head 用正常 lr。
                CPU 上很慢，建议有 GPU（RTX 5070 / sm_120）时使用。

CPU 机器上正确的打开方式不是「从头训练一个深度网络」，而是：
    ImageNet 预训练主干（冻结，只做前向）→ 固定维度特征 → 轻量分类头
这样一次前向算完就能把特征缓存到磁盘，后续调参、换分类器都是秒级的。

支持的 5 种主干及取舍（SUPPORTED_BACKBONES）：
    - resnet18        默认。约 11.7M 参数，CPU 前向最快，小数据集上性价比最高。
    - resnet34        resnet18 的加深版，精度略高、CPU 耗时明显增加，数据少时提升不明显。
    - resnet50        Bottleneck 结构，约 25.6M 参数，特征更强但 CPU 上慢得多，
                      只有数据量上千帧、且能上 GPU 时才值得。
    - mobilenet_v3_small  最轻量，适合极致 CPU 场景；但 ImageNet 特征判别性偏弱，
                      对「靠调色/构图区分电影」的任务提升有限。
    - efficientnet_b0 精度/参数量平衡好，但 CPU 推理比 resnet18 慢，
                      且对输入归一化更敏感（必须用 ImageNet 均值方差）。

被谁使用：
    - ``cineeye/features.py``（抽特征，只用 ``extract_features`` + ``freeze_backbone``）；
    - ``cineeye/train.py``（两种模式训练）；
    - ``cineeye/predict.py`` / ``identify.py`` / ``gallery/encoder.py``（推理时复用主干）；
    - ``tests/test_pipeline.py``。

依赖：torch、torchvision。本机无 GPU 也能正常运行（全部算子走 CPU）；
代码本身设备无关，由 ``cineeye.device.resolve_device`` 决定放到 CPU 还是 CUDA。
注意 RTX 50 系是 Blackwell sm_120，必须装 CUDA 12.8 版 torch（>=2.7），
否则 ``is_available()`` 为 True 但第一次运算会抛 "no kernel image"。

运维要点：
    1. 换 ``model.backbone`` 会改变特征维度与缓存签名，``outputs/features/``
       下的深度特征会自动失效重抽；旧 checkpoint 也与新主干不兼容。
    2. 分类头结构（dropout / head_hidden）必须与 checkpoint 一致，
       否则 ``load_state_dict`` 会报 size mismatch。
    3. 这里不负责设备选择与 seed，也不负责保存 checkpoint，保持「纯模型定义」。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from ..utils import get_logger

log = get_logger()

SUPPORTED_BACKBONES = (
    "resnet18", "resnet34", "resnet50",
    "mobilenet_v3_small", "efficientnet_b0",
)
# 白名单而非「随便传个名字」：不同主干的分类头属性名（fc/classifier/head）不同，
# 且特征维度差异很大。提前校验能给出清晰报错，也避免拼错名字后
# 静默用一个随机初始化的网络跑完整个训练。


def _first_linear_in_features(module: nn.Module) -> int | None:
    """找出模块里第一个 ``nn.Linear`` 的输入维度。

    用于 mobilenet/efficientnet 这类分类头是 ``Sequential`` 的主干：
    真正的特征维度藏在第一个全连接层上，直接读 ``classifier`` 拿不到。

    Args:
        module: 任意子模块（通常是 ``model.classifier`` 或 ``model.head``）。

    Returns:
        第一个线性层的 ``in_features``；模块里没有线性层时返回 None。

    Raises:
        无。
    """
    for m in module.modules():
        if isinstance(m, nn.Linear):
            return m.in_features
    return None


def build_backbone(name: str, pretrained: bool = True) -> tuple[nn.Module, int, bool]:
    """返回 (去掉分类层的主干, 特征维度, 是否真的加载到了预训练权重)。

    分类头被替换为 ``nn.Identity()``，这样 ``backbone(x)`` 直接输出特征向量，
    既便于抽特征，也让 finetune 模式下分类头独立可换。

    Args:
        name: 主干名，必须是 :data:`SUPPORTED_BACKBONES` 之一。
        pretrained: 是否加载 ImageNet 预训练权重。

    Returns:
        ``(model, feature_dim, used_pretrained)``。第三个值很关键：
        离线环境下载失败时会降级为随机初始化，但**不报错**，
        调用方（训练报告、特征缓存元信息）需要据此提示「效果会明显下降」。

    Raises:
        ValueError: 主干名不在白名单内。
        RuntimeError: 无法推断特征维度，或无法剥离分类头
            （通常是 torchvision 改版后模型结构变了，需要同步改这里的分支）。
    """
    if name not in SUPPORTED_BACKBONES:
        raise ValueError(f"不支持的主干 {name}，可选：{SUPPORTED_BACKBONES}")

    weights = "DEFAULT" if pretrained else None
    used = False
    try:
        # 优先走新版 API ``models.get_model``（torchvision>=0.15），
        # 老版本没有该函数时退回 ``models.<name>(weights=...)``。
        getter = getattr(models, "get_model", None)
        model = getter(name, weights=weights) if getter else getattr(models, name)(weights=weights)
        used = bool(pretrained)
    except Exception as e:  # noqa: BLE001
        # 离线 / 权重下载失败时降级为随机初始化，仍保证流程可跑通
        log.warning("预训练权重加载失败（%s），改用随机初始化。效果会明显下降，"
                    "建议联网后重试或手动放置权重。", e)
        model = getattr(models, name)(weights=None)

    # 三种分类头命名风格，按 torchvision 的实际结构分支处理：
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        # ResNet 系列：单层 fc
        dim = model.fc.in_features
        model.fc = nn.Identity()
    elif hasattr(model, "classifier"):
        # MobileNet / EfficientNet：classifier 是 Sequential，特征维度在第一个 Linear 上
        dim = _first_linear_in_features(model.classifier)
        if dim is None:
            raise RuntimeError(f"无法推断 {name} 的特征维度")
        model.classifier = nn.Identity()
    elif hasattr(model, "head"):
        # 兼容 ViT 等 head 结构的模型（当前白名单里没有，留作扩展）
        dim = _first_linear_in_features(model.head)
        if dim is None:
            raise RuntimeError(f"无法推断 {name} 的特征维度")
        model.head = nn.Identity()
    else:
        raise RuntimeError(f"无法剥离 {name} 的分类头")
    return model, int(dim), used


class CineEyeNet(nn.Module):
    """影瞳分类网络：主干 + Dropout + (可选隐藏层) + 线性分类头。

    结构固定为「backbone → head」，两种模式共用同一套参数：

        feature  模式：backbone 冻结（requires_grad=False），只训练 head。
                      此时 ``extract_features`` 的输出可缓存成 ``.npz``，
                      训练 loop 根本不进 CNN（在特征矩阵上跑），速度极快。
        finetune 模式：backbone 解冻，与 head 一起端到端训练。
                      主干学习率通常设为主学习率的 0.1 倍（由 train.py 控制），
                      避免预训练权重被大 lr 破坏（catastrophic forgetting）。

    注意 ``head`` 里已包含 Dropout 与可选的隐藏层，因此 checkpoint 的
    ``head.*`` 权重形状取决于 ``dropout`` / ``head_hidden`` 配置，
    推理时必须用相同配置构造模型再 load_state_dict。
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.2,
        head_hidden: int = 0,
    ):
        """Args:
            backbone: 主干名，见 :data:`SUPPORTED_BACKBONES`。
            num_classes: 分类数（电影部数）。
            pretrained: 是否用 ImageNet 权重；False 仅用于对比实验。
            dropout: 分类头前的 dropout 概率，<=0 表示不用。
            head_hidden: >0 时在特征与输出之间插入一层隐藏层，
                数据量较大且类别较多时可提升拟合能力；小数据集建议保持 0。
        """
        super().__init__()
        self.backbone_name = backbone
        # 记录 pretrained_ok：下载失败时这里是 False，训练报告会据此提示
        self.backbone, self.feature_dim, self.pretrained_ok = build_backbone(backbone, pretrained)
        self.num_classes = num_classes

        # 分类头用 nn.Sequential 拼装，好处是 checkpoint 里键名稳定（head.0/head.1...），
        # 且与「是否有隐藏层」解耦，换结构只需要改这里。
        layers: list[nn.Module] = []
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        if head_hidden > 0:
            layers += [nn.Linear(self.feature_dim, head_hidden), nn.ReLU(inplace=True)]
            if dropout > 0:
                # 隐藏层后再加一次 dropout：中间层参数量大，是最容易过拟合的地方
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(head_hidden, num_classes))
        else:
            # 默认单层线性探针（linear probe）：在冻结特征上训练，几乎不会过拟合
            layers.append(nn.Linear(self.feature_dim, num_classes))
        self.head = nn.Sequential(*layers)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """只跑主干，返回 ``(B, feature_dim)`` 的特征向量。

        ResNet 等主干的输出已经是二维 ``(B, C)``；个别主干（若换成带空间维度的）
        会返回 ``(B, C, H, W)``，这里统一 flatten，保证下游拿到固定二维矩阵。

        Args:
            x: ``(B, 3, H, W)`` 已归一化的输入张量。

        Returns:
            二维特征张量，dtype/device 与输入一致。

        Raises:
            RuntimeError: 输入通道数或空间尺寸不满足主干要求时（torch 抛出）。
        """
        f = self.backbone(x)
        if f.dim() > 2:
            # 全局特征：把空间维摊平。训练/推理必须走同一条路径，
            # 否则缓存下来的特征维度会和实时推理不一致。
            f = torch.flatten(f, 1)
        return f

    def forward(self, x: torch.Tensor, return_feature: bool = False):
        """前向：``return_feature=True`` 时跳过分类头，直接给特征。

        抽特征阶段（features.py）用它一次拿到整批特征；
        普通推理走默认分支，输出 ``(B, num_classes)`` 的 logits（未 softmax）。

        Args:
            x: ``(B, 3, H, W)`` 输入张量。
            return_feature: 是否只返回特征、不经过分类头。

        Returns:
            特征张量或 logits 张量。

        Raises:
            RuntimeError: 主干前向失败时（torch 抛出）。
        """
        f = self.extract_features(x)
        return f if return_feature else self.head(f)

    def freeze_backbone(self, freeze: bool = True) -> None:
        """冻结 / 解冻主干参数，对应 feature / finetune 两种模式。

        只改 ``requires_grad``，不改结构，因此可以在同一个模型实例上切换：
        train.py 里 finetune 模式显式调 ``freeze_backbone(False)``。

        Args:
            freeze: True 冻结（不更新、也不计算梯度），False 解冻。

        Returns:
            None

        Raises:
            无。
        """
        for p in self.backbone.parameters():
            p.requires_grad_(not freeze)
        # 冻结时主干固定走 eval，避免 BatchNorm 用 batch 统计漂移
        # （注意：这只在 train() 被覆写后生效，见下面的 train 方法）
        self._backbone_frozen = freeze

    def train(self, mode: bool = True):
        """覆写 ``nn.Module.train``：冻结主干时强制主干保持 eval 模式。

        背景：``model.train()`` 会把所有子模块设为训练态，BatchNorm 会改用
        **当前 batch** 的均值方差并更新 running stats。在 feature 模式下
        主干是冻结的，若 BN 统计量仍被小 batch 反复改写，缓存特征就会漂移，
        同一条视频在不同 batch 里得到不同特征。

        Args:
            mode: True 训练态，False 推理态。

        Returns:
            self（支持链式调用，与 PyTorch 惯例一致）。

        Raises:
            无。
        """
        super().train(mode)
        if getattr(self, "_backbone_frozen", False):
            self.backbone.eval()
        return self


def build_model(cfg: dict, num_classes: int, pretrained: bool | None = None) -> CineEyeNet:
    """按配置构造模型（训练、评估、推理、图库编码共用这一个入口）。

    Args:
        cfg: 全局配置，读取 ``model.backbone`` / ``model.pretrained`` /
            ``model.dropout`` / ``model.head_hidden``。
        num_classes: 输出类别数，必须与 ``label_map`` 的长度一致；
            与 checkpoint 不一致会在 ``load_state_dict`` 时报 size mismatch。
        pretrained: 显式覆盖配置里的 ``model.pretrained``；
            None 表示跟随配置。

    Returns:
        未搬到设备的 ``CineEyeNet`` 实例（调用方负责 ``.to(device)``）。

    Raises:
        ValueError: 配置里的主干名不受支持（来自 :func:`build_backbone`）。
        RuntimeError: 无法剥离主干分类头。
    """
    # 延迟导入：cnn.py 只依赖 torch/torchvision，把它做成「无 config 也能用」的纯模型层，
    # 方便单测和离线实验直接构造，不必先加载 YAML。
    from ..config import get_value

    return CineEyeNet(
        backbone=str(get_value(cfg, "model.backbone", "resnet18")),
        num_classes=num_classes,
        pretrained=bool(get_value(cfg, "model.pretrained", True)) if pretrained is None else pretrained,
        dropout=float(get_value(cfg, "model.dropout", 0.2)),
        head_hidden=int(get_value(cfg, "model.head_hidden", 0)),
    )
