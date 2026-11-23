"""影瞳评估：同时给出帧级与视频级指标、混淆矩阵与逐帧预测明细。

用途：
    加载 best.pt，对指定 split 计算 top-1 / top-5、macro-F1、balanced accuracy，
    并输出混淆矩阵（CSV+PNG）、分类报告（TXT）、逐帧预测明细（CSV）与汇总（JSON）。
    帧级指标衡量“单张截图识别得准不准”，视频级指标衡量“用户传一段视频、
    系统给一个答案”时的真实体验，两者一起看才能判断模型是否可上线。

被谁使用：
    - ``scripts/04_evaluate.py``：命令行入口，调用 evaluate()
    - 产出的 ``outputs/reports/eval_<split>.json`` 是验收与回归对比的依据

依赖：
    matplotlib（无界面 Agg 后端）、numpy、pandas、torch、scikit-learn；
    模型结构与 checkpoint 读取复用 predict.py。

运维要点：
    - 无显示环境，matplotlib 必须在 import pyplot 之前切到 Agg，否则可能尝试连 X11。
    - 中文字体缺失时混淆矩阵自动退化为 C0/C1 编号，指标不受影响
      （要显示中文可安装 Noto Sans CJK）。
    - matplotlib 的字体缓存目录若被指到 /tmp（本机仅约 10MB），首次出图可能失败；
      可把 MPLCONFIGDIR 指到项目内可写目录。
    - feature 模式会优先复用 features.cache_dir 下的 deep_*.npz；删了缓存只是变慢，
      结果应当一致（除非缓存与当前 backbone/分辨率不匹配）。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
# 必须在 import pyplot 之前指定后端：服务器/容器没有 DISPLAY，
# 交互后端会尝试连接 X11 并报错；Agg 只负责把图写到文件
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score,
)

from .config import ROOT, get_value
from .config import path as cfg_path
from .data.dataset import build_dataloader, load_manifest
from .device import resolve_device
from .models.cnn import CineEyeNet
from .predict import Predictor, _load_checkpoint
from .utils import ensure_dir, get_logger, save_json

log = get_logger()


def _cache_probabilities(cfg: dict, split: str, ckpt: dict, device: torch.device):
    """feature 模式下直接复用缓存的深度特征，避免重新跑一遍 CNN。

    缓存里存的是主干输出 (N, D)，这里只把分类头搬到 device 上做前向并 softmax，
    因此即使几千帧也能在几秒内完成评估。

    Args:
        cfg: 配置字典（决定 cache_dir 与 dropout/head_hidden）。
        split: 数据划分名（train/val/test），对应 deep_<split>.npz。
        ckpt: 已加载的 checkpoint，需含 mode/backbone/num_classes/model_state。
        device: 计算设备。

    Returns:
        (probs, y, video_id) 三元组，probs 为 (N, C) 概率；
        当 checkpoint 不是 feature 模式或缓存文件不存在时返回 None，
        由调用方回退到整图推理。

    Raises:
        KeyError: checkpoint 缺少必需字段。
        RuntimeError: state_dict 与重建的网络结构不匹配（改过 backbone 或 head 结构）。
    """
    npz = cfg_path(cfg, "features.cache_dir") / f"deep_{split}.npz"
    if ckpt.get("mode") != "feature" or not npz.exists():
        return None
    data = np.load(npz, allow_pickle=True)
    # 从头构建网络只为拿到 head；pretrained=False，权重马上被 checkpoint 覆盖，
    # 这样评估不需要联网下载 torchvision 预训练权重
    model = CineEyeNet(
        backbone=ckpt["backbone"], num_classes=int(ckpt["num_classes"]),
        pretrained=False,
        dropout=float(get_value(cfg, "model.dropout", 0.2)),
        head_hidden=int(get_value(cfg, "model.head_hidden", 0)),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)
    X = torch.from_numpy(data["X"]).float()
    with torch.no_grad():
        # 分块 256：控制 CPU 峰值内存，避免一次性把几千×D 的激活全展开
        probs = torch.cat([
            torch.softmax(model.head(X[i:i + 256].to(device)), dim=1).cpu()
            for i in range(0, len(X), 256)
        ]).numpy()
    log.info("评估复用特征缓存：%s %s", split, data["X"].shape)
    return probs, data["y"].astype(np.int64), data["video_id"].astype(str)


def _image_probabilities(cfg: dict, df: pd.DataFrame, predictor: Predictor):
    """缓存不存在时，从原图重新推理。

    Args:
        cfg: 配置字典（决定图像尺寸、batch 与 num_workers）。
        df: 该 split 的 manifest 子集。
        predictor: 已加载模型的 Predictor。

    Returns:
        (probs (N,C) float32, y (N,) int64, vids (N,) str)；
        df 为空时返回形状为 (0, num_classes) 的空概率数组。

    Raises:
        torch.cuda.OutOfMemoryError: GPU 显存不足（可调小 train.batch_size）。
    """
    loader = build_dataloader(df, cfg, train=False)  # train=False → 无随机增强，评估可复现
    probs = []
    with torch.no_grad():
        for images, _, _ in loader:
            probs.append(predictor.predict_proba(images))
    return (
        np.concatenate(probs, axis=0) if probs else np.zeros((0, predictor.num_classes), np.float32),
        df["label_idx"].to_numpy(np.int64),
        df["video_id"].astype(str).to_numpy(),
    )


def _topk_accuracy(y: np.ndarray, probs: np.ndarray, k: int) -> float:
    """numpy 版 top-k 准确率，语义与 train._topk 一致。

    Args:
        y: (N,) 真实标签索引。
        probs: (N, C) 概率或分数。
        k: 取前 k 个；超过类别数时自动截断。

    Returns:
        float: 命中比例。

    Raises:
        ValueError: y 与 probs 行数不一致时可能静默截断（zip 按较短者停止），
            调用方需自行保证两者对齐。
    """
    k = min(k, probs.shape[1])
    # argsort 升序，取最后 k 列即分数最高的 k 类
    top = np.argsort(probs, axis=1)[:, -k:]
    return float(np.mean([yi in row for yi, row in zip(y, top)]))


def _setup_cjk_font() -> bool:
    """挑一个能显示中文的字体；找不到时改用类别编号，避免混淆矩阵变成方框。

    副作用：会修改 matplotlib 的全局 rcParams，对后续所有图生效。

    Returns:
        bool: True 表示已找到并启用中文字体；False 表示只能用默认字体。
    """
    import logging
    from matplotlib import font_manager

    # Noto CJK 只提供部分字重，font_manager 会以 WARNING 级别刷 "Failed to find font weight"。
    # 已确认字体能正常渲染中文，这里临时静噪，不影响出图。
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

    preferred = [
        "Noto Sans CJK SC", "Noto Sans CJK JP", "Source Han Sans SC",
        "WenQuanYi Zen Hei", "Microsoft YaHei", "SimHei", "PingFang SC",
        "Droid Sans Fallback",
    ]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in installed:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            # 中文字体常缺 U+2212（真减号），关掉 unicode_minus 防止负号显示为方框
            plt.rcParams["axes.unicode_minus"] = False
            return True
    return False


def _plot_confusion(cm: np.ndarray, names: list[str], path: Path) -> None:
    """把混淆矩阵画成 PNG 热力图。

    类别数超过 20 时省略坐标刻度文字和格子数字，否则字会糊成一团；
    这种情况下仍可看颜色分布，精确数值查同名 CSV。

    Args:
        cm: (N, N) 混淆矩阵，行=真实类别，列=预测类别。
        names: 长度 N 的类别名，用于坐标轴。
        path: 输出 PNG 路径。

    Returns:
        None。

    Raises:
        OSError: 输出目录不可写或磁盘空间不足。
    """
    n = len(names)
    if not _setup_cjk_font():
        log.warning("未找到中文字体，混淆矩阵改用类别编号标注（安装 Noto Sans CJK 可显示中文）")
        names = [f"C{i}" for i in range(n)]
    # 画布随类别数增长，但限制在 6~20 英寸，避免几十类时图片过大
    fig_w = max(6, min(20, n * 0.9))
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * 0.85))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if n <= 20:
        ax.set_xticks(range(n), names, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(n), names, fontsize=8)
        # 以矩阵最大值为界决定格子文字黑白：深色格子用白字才看得清
        thresh = cm.max() / 2 if cm.max() else 1
        for i in range(n):
            for j in range(n):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black", fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("CineEye / Confusion Matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)  # 主动关闭，批量评估多次时避免 figure 泄漏


def evaluate(cfg: dict, split: str = "test", checkpoint: str | Path | None = None) -> dict:
    """对指定 split 做完整评估，落盘全部报告并返回汇总。

    流程：
        1. 读 checkpoint，还原 idx→片名 映射与运行设备；
        2. 帧级：每帧独立 argmax，统计 top1/top5/macro-F1/balanced accuracy；
        3. 视频级：同一 video_id 的多帧概率取平均后再 argmax（多帧投票）。
           为什么视频级更代表真实体验：用户实际输入的是一段视频，单帧可能恰好
           是黑屏、片头、通用街景或转场，帧级指标会把这些偶然失误全部计入；
           聚合后噪声被平均掉，更接近“看完几帧再下结论”的人类判断方式，
           也是线上 /predict/video 接口的真实工作方式；
        4. 输出混淆矩阵、分类报告、逐帧明细与 JSON 汇总。

    Args:
        cfg: 配置字典。
        split: 要评估的划分，默认 test。
        checkpoint: 可选 checkpoint 路径；None 时读 inference.checkpoint。

    Returns:
        dict: 含 checkpoint / mode / frame_level / video_level / per_class。
        frame_level 与 video_level 均含 top1/top5/macro_f1（帧级另有
        balanced_accuracy，视频级另有 n_videos）。

    Raises:
        FileNotFoundError: checkpoint 不存在。
        RuntimeError: 该 split 没有样本，或 checkpoint 与网络结构不匹配。
    """
    ckpt_path = Path(checkpoint or cfg_path(cfg, "inference.checkpoint"))
    ckpt = _load_checkpoint(ckpt_path)
    label_map: dict[str, int] = ckpt["label_map"]
    idx2label = {v: k for k, v in label_map.items()}
    names = [idx2label[i] for i in range(len(label_map))]
    device = resolve_device(
        str(get_value(cfg, "train.device", "auto")),
        int(get_value(cfg, "train.cpu_threads", 8)),
    )

    df, _ = load_manifest(cfg)
    sub = df[df["split"] == split].reset_index(drop=True)
    if len(sub) == 0:
        raise RuntimeError(f"{split} 集为空，无法评估")

    cached = _cache_probabilities(cfg, split, ckpt, device)
    if cached is not None:
        probs, y, vids = cached
    else:
        # 缓存缺失就老老实实走整图推理；结果应与缓存路径一致，只是慢一些
        predictor = Predictor(ckpt_path, cfg=cfg, device=device)
        probs, y, vids = _image_probabilities(cfg, sub, predictor)

    pred = probs.argmax(axis=1)
    labels_all = list(range(len(label_map)))  # 显式传 labels，避免某类没被预测时指标维度错位
    frame_metrics = {
        "split": split, "n_frames": int(len(y)), "n_videos": int(len(set(vids))),
        "top1": float(accuracy_score(y, pred)),
        "top5": _topk_accuracy(y, probs, 5),
        # macro 平均：每类权重相同，长尾小类不会被大类的样本量淹没
        "macro_f1": float(f1_score(y, pred, labels=labels_all, average="macro", zero_division=0)),
        # balanced accuracy = 各类召回率的平均，类别不均衡时比普通 accuracy 更诚实
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
    }

    # 视频级：把同一段视频所有帧的概率平均后再判定
    # （平均概率而非平均 logits：softmax 后的概率有明确权重含义，且对极端 logits 更稳健）
    vdf = pd.DataFrame({"video_id": vids, "y": y})
    for c in range(probs.shape[1]):
        vdf[f"p{c}"] = probs[:, c]
    grouped = vdf.groupby("video_id")
    # 视频真实标签取帧标签的众数：同一视频内标签本应一致，众数能容忍少量脏标注
    v_true = grouped["y"].agg(lambda s: int(s.mode().iloc[0])).to_numpy()
    v_prob = grouped[[f"p{c}" for c in range(probs.shape[1])]].mean().to_numpy()
    v_pred = v_prob.argmax(axis=1)
    video_metrics = {
        "n_videos": int(len(v_true)),
        "top1": float(accuracy_score(v_true, v_pred)),
        # 视频级 top5 同样基于“聚合后的概率”，而不是“任一帧命中就算对”
        "top5": _topk_accuracy(v_true, v_prob, 5),
        "macro_f1": float(f1_score(v_true, v_pred, labels=labels_all, average="macro", zero_division=0)),
    }

    reports_dir = ensure_dir(ROOT / "outputs" / "reports")
    cm = confusion_matrix(y, pred, labels=labels_all)
    # utf-8-sig 带 BOM，Windows/Excel 直接双击打开中文不会乱码
    pd.DataFrame(cm, index=names, columns=names).to_csv(
        reports_dir / f"confusion_matrix_{split}.csv", encoding="utf-8-sig")
    _plot_confusion(cm, names, reports_dir / f"confusion_matrix_{split}.png")

    txt = classification_report(y, pred, labels=labels_all, target_names=names, zero_division=0)
    (reports_dir / f"classification_report_{split}.txt").write_text(txt, encoding="utf-8")

    # 逐帧明细：排查“哪一帧被错认成哪部片”最直接的证据，confidence 取最大概率
    pd.DataFrame({
        "video_id": vids,
        "true_label": [idx2label.get(int(i), str(i)) for i in y],
        "pred_label": [idx2label.get(int(i), str(i)) for i in pred],
        "confidence": probs.max(axis=1).round(6),
        "correct": pred == y,
    }).to_csv(reports_dir / f"predictions_{split}.csv", index=False, encoding="utf-8-sig")

    result = {
        "checkpoint": str(ckpt_path), "mode": ckpt.get("mode"),
        "frame_level": frame_metrics, "video_level": video_metrics,
        # per_class 用 dict 形式，方便下游脚本直接按片名取 precision/recall，无需再解析文本
        "per_class": classification_report(
            y, pred, labels=labels_all, target_names=names,
            output_dict=True, zero_division=0,
        ),
    }
    save_json(result, reports_dir / f"eval_{split}.json")
    log.info("帧级  top1=%.4f top5=%.4f macro_f1=%.4f (%d 帧)",
             frame_metrics["top1"], frame_metrics["top5"], frame_metrics["macro_f1"], len(y))
    log.info("视频级 top1=%.4f top5=%.4f macro_f1=%.4f (%d 段)",
             video_metrics["top1"], video_metrics["top5"], video_metrics["macro_f1"], len(v_true))
    log.info("评估报告 → %s", reports_dir)
    return result
