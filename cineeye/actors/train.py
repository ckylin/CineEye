"""演员分类头训练：冻结主干抽特征 → 训练线性分类头 → 评估/落盘。

为什么要"先抽特征再训分类头"，而不是端到端微调：
    1. 演员数据通常只有几十到几百张，端到端微调容易过拟合，且 CPU 上很慢；
    2. 冻结主干后，同一张脸的特征是固定的，训练只是在做一个"线性判别"，
       几百轮全批梯度下降在 CPU 上通常 1~3 秒完成；
    3. 特征可以缓存，换分类头/调超参时不用重复解码和跑 CNN。
    需要更强效果时，可把 actors.encoder 换成 movie（用电影数据训过的主干），
    或在后续版本接入 ArcFace 专用人脸特征（face.FaceEncoderStub 预留接口）。

被谁使用：scripts/16_train_actors.py。
依赖：actors/dataset.py（manifest/裁剪规则）、actors/encoder.py（特征）、
    actors/face.py（人脸检测）、torch。
运维要点（改这里之前必须知道的事）：
    1. 训练与推理必须使用同一套裁剪规则（crop mode / pad / fallback），
       否则线上特征分布与训练不一致，准确率会莫名下降；这些参数会写进
       checkpoint，推理端直接读，不需要人工同步；
    2. 特征缓存带签名（manifest 内容 + 编码器指纹 + 裁剪参数），
       任何一项变化都会自动失效重算；也可 --rebuild-cache 强制重算；
    3. 分类头只认识训练时出现过的演员。新演员必须重新导入并重训，
       否则会被强行分到某个旧演员上——靠 unknown_threshold 拒识兜底；
    4. 阈值默认按验证集置信度分布自动估计，属于经验值；若有"非目标演员"
       照片，建议用 actors.unknown_dir 做真实校准（见 calibrate_threshold）。
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn

from ..config import ROOT, get_value
from ..utils import ensure_dir, get_logger
from .dataset import PhotoRecord, abs_path, prepare_face, read_manifest
from .encoder import ActorEncoder
from .face import FaceDetector

log = get_logger("cineeye.actors.train")

CROP_MODES = ("detect", "center", "whole")


def _cfg_path(cfg: dict, key: str, default: str) -> Path:
    """读取配置里的路径项并解析为绝对路径（相对项目根）。"""
    raw = get_value(cfg, key, default)
    p = Path(raw)
    return p if p.is_absolute() else ROOT / p


def _cache_files(cfg: dict) -> tuple[Path, Path]:
    """特征缓存的两个文件：npz 数据 + meta 签名。"""
    cache = _cfg_path(cfg, "actors.cache", "outputs/features/actor_features.npz")
    return cache, cache.with_suffix(".meta.json")


def cache_signature(records: list[PhotoRecord], encoder: ActorEncoder,
                    crop_mode: str, pad: float) -> str:
    """计算特征缓存签名：数据内容 + 编码器 + 裁剪规则。

    任何一项变化都应让旧缓存失效：
        - manifest 增删照片 / 换了照片（sha1 变）；
        - 换 backbone / encoder 模式 / 输入分辨率（fingerprint 变）；
        - 改人脸裁剪方式或外扩比例（特征区域变）。
    """
    h = hashlib.md5()
    for r in sorted(records, key=lambda x: x.path):
        h.update(f"{r.path}|{r.sha1}|{r.split}".encode())
    h.update(f"|{encoder.fingerprint()}|{crop_mode}|{pad:.3f}".encode())
    return h.hexdigest()


def extract_features(cfg: dict, records: list[PhotoRecord], encoder: ActorEncoder,
                     detector: FaceDetector | None, crop_mode: str = "detect",
                     pad: float = 0.2, rebuild: bool = False) -> dict[str, Any]:
    """把人脸裁剪图编码成特征矩阵，并尽量复用缓存。

    Args:
        cfg: 全局配置。
        records: manifest 记录（决定样本与标签）。
        encoder: 演员编码器。
        detector: 人脸检测器；None 表示不做检测。
        crop_mode: detect / center / whole。
        pad: 人脸外扩比例。
        rebuild: True 时忽略缓存强制重算。

    Returns:
        dict：X (N,D) float32、y (N,) int、actors 列表（下标即类别）、
        splits (N,) 字符串数组、paths、face_ok (N,) bool、cache_hit bool。

    Raises:
        ValueError: 有效样本为 0（照片全都读不出来）。
    """
    npz_path, meta_path = _cache_files(cfg)
    sig = cache_signature(records, encoder, crop_mode, pad)
    if not rebuild and npz_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("signature") == sig:
                data = np.load(npz_path, allow_pickle=True)
                log.info("命中演员特征缓存：%s（%d 张）", npz_path.name, len(data["y"]))
                return {
                    "X": data["X"].astype(np.float32),
                    "y": data["y"].astype(np.int64),
                    "actors": [str(a) for a in data["actors"]],
                    "splits": np.array([str(s) for s in data["splits"]]),
                    "paths": [str(p) for p in data["paths"]],
                    "face_ok": data["face_ok"].astype(bool),
                    "cache_hit": True,
                }
        except (OSError, ValueError, KeyError) as exc:
            # 缓存损坏/字段缺失：不致命，重算即可
            log.warning("演员特征缓存不可用，重新计算：%s", exc)

    actors = sorted({r.actor for r in records})
    label_of = {a: i for i, a in enumerate(actors)}
    crops: list[np.ndarray] = []
    keep: list[PhotoRecord] = []
    face_ok: list[bool] = []
    for r in records:
        img = cv2.imread(str(abs_path(r.path)))
        if img is None:
            log.warning("跳过无法读取的图片：%s", r.path)
            continue
        crop, ok = prepare_face(
            img, detector, mode=crop_mode, pad=pad,
            # 训练时一定回退（否则样本会被丢掉），是否允许"整图识别"由配置决定
            fallback_whole=bool(get_value(cfg, "actors.fallback_whole_image", True)),
        )
        if crop is None or crop.size == 0:
            continue
        crops.append(crop)
        keep.append(r)
        face_ok.append(ok)
    if not keep:
        raise ValueError("没有任何可用样本：请检查 manifest 里的图片路径是否有效")

    t0 = time.time()
    X = encoder.encode(crops)
    y = np.array([label_of[r.actor] for r in keep], dtype=np.int64)
    splits = np.array([r.split for r in keep])
    log.info("演员特征提取完成：%d 张 / %d 类 / %.1fs（检测到人脸 %.0f%%）",
             len(keep), len(actors), time.time() - t0, 100 * np.mean(face_ok))

    ensure_dir(npz_path.parent)
    np.savez_compressed(
        npz_path, X=X, y=y, actors=np.array(actors),
        splits=splits, paths=np.array([r.path for r in keep]),
        face_ok=np.array(face_ok),
    )
    meta_path.write_text(json.dumps({
        "signature": sig, "encoder": encoder.fingerprint(),
        "mode": encoder.mode, "backbone": encoder.backbone_name,
        "crop": crop_mode, "pad": pad, "images": len(keep),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "X": X, "y": y, "actors": actors, "splits": splits,
        "paths": [r.path for r in keep], "face_ok": np.array(face_ok),
        "cache_hit": False,
    }


class ActorHead(nn.Module):
    """线性分类头：特征 (D) → 演员 logits (C)。

    Args:
        feature_dim: 输入特征维度（由编码器决定）。
        num_classes: 演员数量。

    Notes:
        特征已 L2 归一化，线性层等价于"每个演员一个原型向量的余弦分类器"，
        小样本下比多层 MLP 更稳（参数少、不易过拟合）。
    """

    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(int(feature_dim), int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def _class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    """按类别频率的倒数加权（均值归一到 1），缓解演员样本数不均。"""
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0  # 理论上不会出现；防御除零
    w = counts.sum() / (num_classes * counts)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def _accuracy(logits: torch.Tensor, y: torch.Tensor, k: int = 1) -> float:
    """top-k 准确率（k=1 时为常规准确率）。"""
    if len(y) == 0:
        return 0.0
    k = min(k, logits.shape[1])
    topk = logits.topk(k, dim=1).indices
    hit = (topk == y.unsqueeze(1)).any(dim=1)
    return float(hit.float().mean().item())


def train_head(X: np.ndarray, y: np.ndarray, splits: np.ndarray, num_classes: int,
               epochs: int = 200, lr: float = 0.01, weight_decay: float = 1e-4,
               batch_size: int = 64, patience: int = 40, seed: int = 42) -> dict[str, Any]:
    """在缓存特征上训练线性分类头。

    Args:
        X: (N,D) 特征。
        y: (N,) 类别下标。
        splits: (N,) train/val/test。
        num_classes: 类别数。
        epochs: 最大轮数。
        lr: 学习率。
        weight_decay: L2 正则（小样本下建议保留）。
        batch_size: 批大小；样本少于该值时退化为全批训练。
        patience: 验证准确率连续多少轮不提升就早停。
        seed: 随机种子（影响参数初始化与打乱顺序）。

    Returns:
        {state_dict, metrics, history, best_epoch}。
        没有 val 样本时用 train 准确率做早停依据，并在 metrics 里标明。
    """
    torch.manual_seed(seed)
    device = torch.device("cpu")  # 线性头极小，CPU 足够；避免与编码器抢显存
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    tr = torch.tensor(splits == "train")
    va = torch.tensor(splits == "val")
    if tr.sum() == 0:
        raise ValueError("train 划分为空：请检查 manifest 的 split 列")

    model = ActorHead(X.shape[1], num_classes).to(device)
    # 类别权重只在训练集上统计（验证集不参与，避免信息泄漏）
    weights = _class_weights(y[tr.numpy()], num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    Xtr, ytr = Xt[tr].to(device), yt[tr].to(device)
    Xva, yva = (Xt[va].to(device), yt[va].to(device)) if va.sum() > 0 else (None, None)
    use_val = Xva is not None
    bs = len(Xtr) if len(Xtr) <= batch_size else batch_size
    g = torch.Generator().manual_seed(seed)

    best = {"score": -1.0, "state": None, "epoch": 0}
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    history: list[dict[str, float]] = []
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr), generator=g) if bs < len(Xtr) else torch.arange(len(Xtr))
        total_loss = 0.0
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            opt.zero_grad(set_to_none=True)
            logits = model(Xtr[idx])
            loss = criterion(logits, ytr[idx])
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * len(idx)
        model.eval()
        with torch.no_grad():
            tr_acc = _accuracy(model(Xtr), ytr)
            va_acc = _accuracy(model(Xva), yva) if use_val else tr_acc
        score = va_acc if use_val else -total_loss  # 无验证集时用 loss 反号当"越大越好"
        history.append({"epoch": epoch, "loss": round(total_loss / len(Xtr), 6),
                        "train_acc": round(tr_acc, 4), "val_acc": round(va_acc, 4)})
        if score > best["score"] + 1e-6:
            best = {"score": score, "epoch": epoch}
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                log.info("演员分类头早停于第 %d 轮（最佳第 %d 轮）", epoch, best["epoch"])
                break

    model.load_state_dict(best_state)
    model.eval()
    metrics = {"best_epoch": best["epoch"], "epochs_run": len(history),
               "has_val": use_val}
    with torch.no_grad():
        for split in ("train", "val", "test"):
            mask = torch.tensor(splits == split)
            if mask.sum() == 0:
                continue
            logits = model(Xt[mask].to(device))
            metrics[split] = {
                "n": int(mask.sum()),
                "top1": round(_accuracy(logits, yt[mask].to(device), 1), 4),
                "top3": round(_accuracy(logits, yt[mask].to(device), 3), 4),
            }
    return {"state_dict": model.state_dict(), "metrics": metrics,
            "history": history, "best_epoch": best["epoch"]}


def _probs(state_dict: dict, X: np.ndarray) -> np.ndarray:
    """用给定权重算 softmax 概率（评估与阈值估计共用）。"""
    model = ActorHead(X.shape[1], state_dict["fc.weight"].shape[0])
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        return torch.softmax(logits, dim=1).numpy()


def calibrate_threshold(known_conf: np.ndarray, unknown_conf: np.ndarray | None = None,
                        default_q: float = 0.05) -> dict[str, Any]:
    """估计 unknown 拒识阈值。

    有两种模式：
        - 有未知样本（unknown_dir 提供了"非目标演员"的照片）：
          扫描 0.05~0.95 的阈值，最大化 TPR - FPR（正确接受 - 错误接受）；
        - 无未知样本：用已知样本置信度的低分位数 × 0.8 作为经验阈值。
          这是启发式值，目的是"宁可少认，不要乱认"，生产中建议补未知样本校准。

    Args:
        known_conf: 验证集上正确分类样本的 top-1 置信度。
        unknown_conf: 未知样本的 top-1 置信度（可为 None）。
        default_q: 无未知样本时取的分位数。

    Returns:
        {threshold, mode, tpr, fpr}。
    """
    known_conf = np.asarray(known_conf, dtype=np.float32)
    if unknown_conf is not None and len(unknown_conf) > 0 and len(known_conf) > 0:
        unk = np.asarray(unknown_conf, dtype=np.float32)
        best = {"threshold": 0.5, "tpr": 0.0, "fpr": 1.0, "score": -2.0}
        for t in np.arange(0.05, 0.96, 0.01):
            tpr = float((known_conf >= t).mean())   # 已知样本被接受的比例
            fpr = float((unk >= t).mean())          # 未知样本被误接受的比例
            score = tpr - fpr
            if score > best["score"]:
                best = {"threshold": round(float(t), 2), "tpr": round(tpr, 4),
                        "fpr": round(fpr, 4), "score": score}
        return {"threshold": best["threshold"], "mode": "calibrated",
                "tpr": best["tpr"], "fpr": best["fpr"]}
    if len(known_conf) == 0:
        return {"threshold": 0.0, "mode": "disabled", "tpr": None, "fpr": None}
    t = float(np.quantile(known_conf, default_q)) * 0.8
    return {"threshold": round(max(0.0, min(0.95, t)), 3), "mode": "heuristic",
            "tpr": None, "fpr": None}


def _unknown_confs(cfg: dict, encoder: ActorEncoder, detector: FaceDetector | None,
                   crop_mode: str, pad: float, state_dict: dict) -> np.ndarray | None:
    """若配置了 actors.unknown_dir，就编码该目录下的照片并返回其 top-1 置信度。

    这些照片应当是"不在演员名单里"的人（路人、其他演员），用于校准拒识阈值。
    目录可以平铺图片，也可以按人分子目录（结构不限，因为不关心标签）。
    """
    raw = get_value(cfg, "actors.unknown_dir")
    if not raw:
        return None
    d = Path(raw)
    if not d.is_absolute():
        d = ROOT / d
    if not d.exists():
        log.warning("actors.unknown_dir 不存在，跳过阈值校准：%s", d)
        return None
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = sorted(p for p in d.rglob("*") if p.suffix.lower() in exts)
    if not paths:
        return None
    crops = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        crop, _ = prepare_face(img, detector, mode=crop_mode, pad=pad, fallback_whole=True)
        if crop is not None and crop.size:
            crops.append(crop)
    if not crops:
        return None
    X = encoder.encode(crops)
    return _probs(state_dict, X).max(axis=1)


def run_training(cfg: dict, manifest: str | Path | None = None, *,
                 epochs: int | None = None, lr: float | None = None,
                 batch_size: int | None = None, crop_mode: str | None = None,
                 rebuild_cache: bool = False, device: str | None = None,
                 eval_only: bool = False) -> dict[str, Any]:
    """端到端训练：读 manifest → 抽特征 → 训头 → 评估 → 落盘。

    Args:
        cfg: 全局配置。
        manifest: manifest 路径；None 用 actors.manifest。
        epochs / lr / batch_size / crop_mode: 覆盖配置里的训练参数。
        rebuild_cache: 强制重算特征缓存。
        device: 覆盖训练设备（编码用，分类头固定 CPU）。
        eval_only: 只评估已有 checkpoint（不训练）。

    Returns:
        训练报告 dict（同时写入 outputs/reports/actor_train_report.json）。

    Raises:
        FileNotFoundError: manifest 不存在，或 eval_only 时找不到 checkpoint。
        ValueError: 样本数为 0 / 只有 1 个演员。
    """
    manifest_path = Path(manifest or get_value(cfg, "actors.manifest", "data/actors/manifest.csv"))
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    records = read_manifest(manifest_path)
    if not records:
        raise ValueError(f"演员清单为空：{manifest_path}")
    if len({r.actor for r in records}) < 2:
        raise ValueError("至少需要 2 位演员才能训练分类头（当前只有 1 位）")

    enc = ActorEncoder(cfg=cfg, device=device)
    det = FaceDetector(cfg) if bool(get_value(cfg, "actors.face_detection", True)) else None
    crop_mode = crop_mode or str(get_value(cfg, "actors.crop", "detect"))
    if crop_mode not in CROP_MODES:
        log.warning("未知裁剪模式 %s，回退 detect", crop_mode)
        crop_mode = "detect"
    pad = float(get_value(cfg, "actors.face_pad", 0.2))

    feats = extract_features(cfg, records, enc, det, crop_mode=crop_mode,
                             pad=pad, rebuild=rebuild_cache)
    X, y, actors, splits = feats["X"], feats["y"], feats["actors"], feats["splits"]
    head_path = _cfg_path(cfg, "actors.head_path", "outputs/checkpoints/actor_head.pt")

    if eval_only:
        if not head_path.exists():
            raise FileNotFoundError(f"找不到演员分类头：{head_path}")
        ckpt = torch.load(head_path, map_location="cpu", weights_only=False)
        state = ckpt["state_dict"]
        metrics = ckpt.get("metrics", {})
        history: list[dict] = []
        threshold = float(ckpt.get("threshold", 0.0))
    else:
        out = train_head(
            X, y, splits, num_classes=len(actors),
            epochs=int(epochs or get_value(cfg, "actors.train.epochs", 200)),
            lr=float(lr or get_value(cfg, "actors.train.lr", 0.01)),
            weight_decay=float(get_value(cfg, "actors.train.weight_decay", 1e-4)),
            batch_size=int(batch_size or get_value(cfg, "actors.train.batch_size", 64)),
            patience=int(get_value(cfg, "actors.train.patience", 40)),
            seed=int(get_value(cfg, "train.seed", get_value(cfg, "seed", 42))),
        )
        state, metrics, history = out["state_dict"], out["metrics"], out["history"]

        # 阈值：配置显式给了就用配置；否则用验证集置信度估计（有未知样本则真实校准）
        probs = _probs(state, X)
        conf = probs.max(axis=1)
        pred = probs.argmax(axis=1)
        val_mask = splits == "val"
        known_conf = conf[val_mask & (pred == y)]
        if len(known_conf) == 0:  # 没有验证集或全错，退而用训练集正确样本
            tr_mask = splits == "train"
            known_conf = conf[tr_mask & (pred == y)]
        unk_conf = _unknown_confs(cfg, enc, det, crop_mode, pad, state)
        cfg_thr = get_value(cfg, "actors.unknown_threshold")
        if cfg_thr is not None:
            threshold = float(cfg_thr)
            thr_info = {"threshold": threshold, "mode": "config"}
        else:
            thr_info = calibrate_threshold(known_conf, unk_conf)
            threshold = float(thr_info["threshold"])
        metrics["threshold"] = {**thr_info, "threshold": round(threshold, 4)}

        # 混淆矩阵用测试集（没有则用验证集/训练集），便于定位"总认成谁"
        eval_mask = splits == "test"
        if not eval_mask.any():
            eval_mask = val_mask if val_mask.any() else np.ones(len(y), bool)
        cm = np.zeros((len(actors), len(actors)), dtype=int)
        for t, p in zip(y[eval_mask], pred[eval_mask]):
            cm[int(t), int(p)] += 1

        ckpt = {
            "version": 1,
            "state_dict": state,
            "label_map": {str(i): a for i, a in enumerate(actors)},
            "actors": actors,
            "feature_dim": int(X.shape[1]),
            "encoder": {"mode": enc.mode, "backbone": enc.backbone_name,
                        "image_size": enc.image_size, "fingerprint": enc.fingerprint()},
            "crop": {"mode": crop_mode, "pad": pad,
                     "fallback_whole": bool(get_value(cfg, "actors.fallback_whole_image", True))},
            "threshold": threshold,
            "metrics": metrics,
            "confusion": {"labels": actors, "matrix": cm.tolist()},
            "config": cfg.get("actors", {}),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        ensure_dir(head_path.parent)
        torch.save(ckpt, head_path)

        # 检索库：用 train 划分的样本建库（避免验证/测试样本进库导致"作弊"）
        train_mask = splits == "train"
        _save_gallery(cfg, X[train_mask], [actors[i] for i in y[train_mask]], enc)

    report = {
        "manifest": str(manifest_path),
        "images": int(len(y)),
        "actors": actors,
        "num_classes": len(actors),
        "split_counts": {s: int((splits == s).sum()) for s in ("train", "val", "test")},
        "face_detected_rate": round(float(feats["face_ok"].mean()) if len(feats["face_ok"]) else 0.0, 4),
        "cache_hit": feats["cache_hit"],
        "encoder": {"mode": enc.mode, "backbone": enc.backbone_name,
                    "feature_dim": int(X.shape[1]), "fingerprint": enc.fingerprint()},
        "threshold": threshold,
        "metrics": metrics,
        "history": history[-20:],  # 只留最后 20 轮，报告文件不至于太大
        "checkpoint": str(head_path),
    }
    report_path = ROOT / "outputs" / "reports" / "actor_train_report.json"
    ensure_dir(report_path.parent)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("演员训练完成：%d 类 / %d 张；%s", len(actors), len(y),
             {s: report["split_counts"][s] for s in ("train", "val", "test")})
    return report


def _save_gallery(cfg: dict, X: np.ndarray, names: list[str], enc: ActorEncoder) -> None:
    """把训练集向量写成演员检索库（embeddings.npy + names.json + encoder.json）。

    检索库是分类头的补充证据：分类头给"概率"，检索给"最像的一张有多像"。
    指纹文件用于检测"库是用另一个编码器建的"，不一致时识别器会拒绝使用旧库。
    """
    ref_dir = _cfg_path(cfg, "actors.reference_dir", "outputs/actors")
    ensure_dir(ref_dir)
    np.save(ref_dir / "embeddings.npy", X.astype(np.float32))
    (ref_dir / "names.json").write_text(
        json.dumps(names, ensure_ascii=False, indent=2), encoding="utf-8")
    (ref_dir / "encoder.json").write_text(json.dumps({
        "fingerprint": enc.fingerprint(), "mode": enc.mode,
        "backbone": enc.backbone_name, "image_size": enc.image_size,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
