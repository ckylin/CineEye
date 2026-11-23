"""冻结主干 → 一次性抽取并缓存深度特征。

用途：把清单里每个 split 的图片批量送进冻结的 CNN 主干，输出
``outputs/features/deep_<split>.npz``（X / y / video_id / path）与同名
``.meta.json``（缓存签名与主干信息）。之后分类头训练、评估、调参
全部在内存里的 .npy 上完成，不再碰图片解码与卷积。

这是整个项目在「没有 GPU」前提下还能高效迭代的关键：
CNN 前向只跑一遍，之后分类头、基线、评估、调参全部在 .npy 上完成。
本机 CPU 上 resnet18 抽几千帧大约几分钟；插上 RTX 5070（Blackwell sm_120，
需 CUDA 12.8 版 torch）后会自动走 CUDA，快一个数量级。

被谁使用：
    - ``cineeye/train.py``（feature 模式先抽特征再训练线性头）；
    - 间接被 ``scripts/03_train.py``、``scripts/run_all.py`` 调用。

依赖：torch、numpy、pandas、``cineeye.data.dataset``、``cineeye.models.cnn``、
``cineeye.device``、``cineeye.config``。

运维要点：
    1. 缓存是否命中由 :func:`_signature` 决定（内容签名，不看 mtime）。
       换主干 / 改 image_size / 改 split 都会自动失效并重抽，属于预期行为。
    2. 缓存写在 ``outputs/features/``（真实磁盘），不会占用只有 10MB 的
       /tmp 与 /dev/shm；但 DataLoader 的 worker 会被压到 0（见 dataset 模块）。
    3. 抽特征必须在 ``torch.no_grad()`` 下进行，否则会保留计算图，
       显存/内存会迅速爆掉——本模块已处理，改代码时不要删这个上下文。
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import torch

from .config import get_value
from .config import path as cfg_path
from .data.dataset import build_dataloader
from .device import resolve_device
from .models.cnn import build_model
from .utils import get_logger

log = get_logger()


def _signature(cfg: dict, df: pd.DataFrame, split: str) -> str:
    """基于清单「内容」而非 mtime：重建清单但内容未变时，不该重跑一遍 CNN 前向。

    为什么不用 mtime：``02_build_manifest.py`` 每次都会重写 ``manifest.csv``，
    即使数据一模一样，mtime 也必然变化。若用 mtime 做缓存键，每次重跑
    manifest 都会导致特征全部重抽（CPU 上要几分钟），运维会以为「缓存没生效」。
    而 mtime 本身也极不稳定：rsync/备份恢复/容器重建都会重置它，
    但它并不代表内容变了。

    这里改为「内容指纹」：把影响特征结果的配置、样本数、少量帧路径，
    以及 manifest 文件**字节内容**一起做 SHA-256，取前 16 位。
    任何一项变化（换主干、改分辨率、样本增删）都会得到不同签名；
    只是 touch 文件则签名不变，缓存继续命中。

    Args:
        cfg: 全局配置，读取 backbone / pretrained / image_size。
        df: 完整清单。
        split: 当前 split 名。

    Returns:
        16 位十六进制字符串。

    Raises:
        OSError: manifest 存在但读取失败（权限问题）时。
    """
    payload = "|".join([
        str(get_value(cfg, "model.backbone", "resnet18")),
        # pretrained 也进签名：预训练与随机初始化的特征完全不同，不能共用缓存
        str(get_value(cfg, "model.pretrained", True)),
        str(get_value(cfg, "data.image_size", 224)),
        str(len(df[df["split"] == split])),
        split,
        # 抽前 20 条路径做「样本指纹」，防止样本数相同但内容整体换了一批的情况。
        # 不拼全量路径是为了避免 payload 过长；manifest 的字节哈希已能覆盖全量。
        "|".join(df[df["split"] == split]["frame_path"].head(20)),
    ])
    h = hashlib.sha256(payload.encode("utf-8"))
    manifest = cfg_path(cfg, "data.manifest")
    if manifest.exists():
        h.update(manifest.read_bytes())
    return h.hexdigest()[:16]


def _cache_valid(cfg: dict, df: pd.DataFrame, split: str) -> bool:
    """判断某个 split 的特征缓存是否仍然可用。

    Args:
        cfg: 全局配置。
        df: 完整清单。
        split: split 名。

    Returns:
        True 仅当 ``.npz`` 与 ``.meta.json`` 都存在、且 meta 里的 signature
        与当前计算出的签名一致。

    Raises:
        不抛异常。meta 损坏/缺字段一律视为「缓存无效」，走重抽路径。
    """
    cache_dir = cfg_path(cfg, "features.cache_dir")
    npz_path = cache_dir / f"deep_{split}.npz"
    meta_path = cache_dir / f"deep_{split}.meta.json"
    if not (npz_path.exists() and meta_path.exists()):
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return meta.get("signature") == _signature(cfg, df, split)
    except Exception:  # noqa: BLE001
        return False


def extract_split_features(
    cfg: dict,
    df: pd.DataFrame,
    split: str,
    num_classes: int,
    device: torch.device | None = None,
    force: bool = False,
    model: torch.nn.Module | None = None,
) -> dict:
    """返回 {"X": 特征矩阵, "y": 标签, "video_id": 分组, "path": 帧路径}。

    ``model`` 可由调用方复用（三个 split 共享同一份主干权重，避免重复加载）。

    Args:
        cfg: 全局配置，读取 ``features.*`` 与 ``train.device`` 等。
        df: 完整清单（内部按 ``split`` 过滤）。
        split: 要处理的名字。
        num_classes: 类别数；仅在需要新建模型时用到（复用 model 时被忽略）。
        device: 指定设备；None 时按 ``train.device`` 自动探测。
        force: True 时忽略缓存强制重抽（调试或怀疑缓存坏了时用）。
        model: 外部传入的已构造模型，用于三个 split 共享权重。

    Returns:
        dict，键为 ``X`` / ``y`` / ``video_id`` / ``path``：
            X 为 ``(N, feature_dim)`` float32；y 为 ``(N,)`` int64；
            video_id、path 为 object 数组（便于 npz 保存与后续按视频聚合）。

    Raises:
        FileNotFoundError: 图片路径失效等读图错误不会抛（数据集有占位逻辑），
            但缓存目录不可写时会抛 ``OSError``。
    """
    cache_dir = cfg_path(cfg, "features.cache_dir", mkdir=True)
    npz_path = cache_dir / f"deep_{split}.npz"
    meta_path = cache_dir / f"deep_{split}.meta.json"
    sig = _signature(cfg, df, split)

    if not force and _cache_valid(cfg, df, split):
        # allow_pickle=True：video_id / path 是 object 数组，必须允许 pickle
        data = np.load(npz_path, allow_pickle=True)
        log.info("命中特征缓存 %s：%s", split, tuple(data["X"].shape))
        return {k: data[k] for k in ["X", "y", "video_id", "path"]}
    if npz_path.exists():
        log.info("特征缓存已过期，重新抽取 %s ...", split)

    device = device or resolve_device(
        str(get_value(cfg, "train.device", "auto")),
        int(get_value(cfg, "train.cpu_threads", 8)),
    )
    if model is None:
        model = build_model(cfg, num_classes)
    model = model.to(device)
    # 即使调用方想 finetune，抽特征阶段也必须冻结：我们只要主干输出，
    # 不解冻可以省掉梯度计算（再配合下面的 no_grad，速度与显存都最优）。
    model.freeze_backbone(True)
    # eval() 必须显式调用：BatchNorm 在 train 模式会用 batch 统计，
    # 会让同一帧在不同 batch 中得到不同特征，缓存就失去意义。
    model.eval()

    sub = df[df["split"] == split].reset_index(drop=True)
    loader = build_dataloader(
        sub, cfg, train=False,
        batch_size=int(get_value(cfg, "features.batch_size", 64)),
    )
    feats, labels, vids, paths = [], [], [], []
    with torch.no_grad():
        for images, ys, video_ids in loader:
            # .cpu().numpy() 再 astype：避免在 GPU 上保留半精度或梯度引用
            f = model.extract_features(images.to(device)).cpu().numpy()
            feats.append(f.astype(np.float32))  # float32 足够，float64 会让 npz 翻倍
            labels.append(ys.numpy())
            vids.extend(video_ids)
            # 用「已收集条数」做游标而不是 enumerate 计数：
            # 与 loader 的实际产出顺序严格对应（shuffle=False，顺序即 sub 的顺序）。
            paths.extend(sub["frame_path"].iloc[
                len(paths):len(paths) + len(ys)].tolist())

    # 空 split（例如某类视频太少导致 test=0）不能 np.concatenate，需返回形状正确的空矩阵
    X = np.concatenate(feats, axis=0) if feats else np.zeros((0, model.feature_dim), np.float32)
    y = np.concatenate(labels).astype(np.int64) if labels else np.array([], np.int64)
    out = {"X": X, "y": y, "video_id": np.asarray(vids, dtype=object), "path": np.asarray(paths, dtype=object)}
    np.savez_compressed(npz_path, **out)
    # meta 里记录签名与主干信息：既用于缓存校验，也方便运维一眼看出这批特征是用什么模型抽的
    meta_path.write_text(json.dumps({
        "signature": sig, "backbone": model.backbone_name, "feature_dim": model.feature_dim,
        "pretrained": model.pretrained_ok, "split": split, "n": int(len(y)),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("深度特征 %-5s: X=%s → %s", split, X.shape, npz_path)
    return out


def extract_all_splits(cfg: dict, df: pd.DataFrame, label_map: dict,
                       device: torch.device | None = None, force: bool = False) -> dict[str, dict]:
    """三个 split 共享同一份主干权重，避免重复加载 3 次预训练模型。

    Args:
        cfg: 全局配置。
        df: 完整清单。
        label_map: 类别映射，长度即 num_classes。
        device: 指定设备；None 时自动探测。
        force: 是否强制重抽（忽略缓存）。

    Returns:
        ``{"train": {...}, "val": {...}, "test": {...}}``，值同
        :func:`extract_split_features` 的返回。

    Raises:
        OSError: 特征缓存目录不可写时。
    """
    if device is None:
        # 提前解析一次设备：让三个 split 用完全相同的 device，
        # 也避免在循环里重复做 CUDA 内核探测（每次探测都要跑一次小矩阵乘法）。
        device = resolve_device(
            str(get_value(cfg, "train.device", "auto")),
            int(get_value(cfg, "train.cpu_threads", 8)),
        )
    # 只 build 一次：resnet 权重加载 + 搬设备是秒级开销，三份会明显拖慢启动
    shared = build_model(cfg, len(label_map)).to(device)
    return {
        split: extract_split_features(
            cfg, df, split, len(label_map), device, force, model=shared,
        )
        for split in ("train", "val", "test")
    }
