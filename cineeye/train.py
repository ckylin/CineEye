"""影瞳训练入口：feature（冻结主干 + 缓存特征训练分类头）与 finetune（端到端微调）两种模式。

用途：
    读取 manifest 与 label_map，按配置训练电影分类模型，产出可直接推理的
    ``outputs/checkpoints/best.pt``、逐 epoch 历史 CSV 与训练报告 JSON。
    feature 模式先对整个数据集做一次 CNN 前向并把特征缓存到磁盘，之后
    调分类头只需在缓存的向量上迭代，因此没有 GPU 也能在几十秒内跑完一轮。

被谁使用：
    - ``scripts/03_train.py``：命令行入口，调用 run_training()
    - ``scripts/run_all.py``：一键流水线的第 ③ 步

依赖：
    config / data.dataset / device / features / models.cnn / models.classical / utils。
    张量计算依赖 torch，特征缓存依赖 numpy，清单与历史依赖 pandas/csv。

运维要点：
    - 本机无 GPU，但代码保留完整 CUDA 路径；RTX 5070（Blackwell sm_120）
      需要 cu128 wheel + torch>=2.7，否则会出现 is_available() 为真、内核却
      加载失败的情况。device.cuda_kernel_ok() 会提前做内核预检并回退 CPU。
    - /tmp 与 /dev/shm 只有约 10MB，DataLoader 多进程会触发 Bus error；
      dataset._safe_num_workers() 已自动降为 num_workers=0，不要随意调大。
    - best.pt 的字段（model_state/label_map/num_classes/backbone/config/metrics）
      是 predict.py 与 evaluate.py 的读取契约，改字段名必须同步改读取方。
    - 早停看验证 loss 而非准确率：loss 对“置信度下降但排序没变”更敏感，
      能在精度饱和后及时停，省掉无效 epoch。
"""

from __future__ import annotations

import copy
import csv
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:  # 仅供类型注解；配合 from __future__ import annotations 运行时不导入
    import pandas as pd

from .config import ROOT, get_value
from .config import path as cfg_path
from .data.dataset import build_dataloader, load_manifest
from .device import log_device, resolve_device
from .features import extract_all_splits
from .models.cnn import build_model
from .utils import ensure_dir, get_logger, save_json, seed_everything

log = get_logger()


def _topk(logits: torch.Tensor, y: torch.Tensor, k: int) -> float:
    """计算 top-k 命中率（取值 0~1）。

    Args:
        logits: (N, C) 未归一化分数，通常直接来自模型输出。
        y: (N,) 真实类别索引，取值范围 [0, C)。
        k: 取分数最高的前 k 个；大于类别数时自动截断，避免 topk 越界。

    Returns:
        float: 命中样本占比。k=1 时即普通准确率。

    Raises:
        RuntimeError: logits 与 y 的批次维度不一致时，由 torch 的比较/广播抛错。
    """
    k = min(k, logits.size(1))  # 类别数少于 k（例如只有 2 部电影）时截断
    top = logits.topk(k, dim=1).indices
    # y[:, None] 广播成 (N, k)，逐行看真实类别是否落在 top-k 里，再对样本求均值
    return float((top == y[:, None]).any(dim=1).float().mean().item())


def _class_weights(y: np.ndarray, num_classes: int, device: torch.device) -> torch.Tensor:
    """按类别频率的倒数生成 CrossEntropyLoss 的类别权重。

    电影帧数往往长尾：热门片几千帧、冷门片几十帧。不加权时模型会偏向
    多数类，macro-F1 被少数类拖垮；倒数权重把少数类样本的 loss 放大。

    权重最后除以均值做归一化，使加权 loss 的量级与不加权时接近，
    这样换数据集时不必重新调 lr。

    Args:
        y: (N,) int 数组，训练集标签索引。
        num_classes: 类别总数，用于对齐 bincount 的长度。
        device: 目标设备，权重张量必须与 logits 同设备，否则 loss 会报设备不一致。

    Returns:
        torch.Tensor: (num_classes,) float32，均值约为 1。

    Raises:
        ValueError: y 中出现负数标签时 numpy.bincount 抛出。
    """
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0  # 某类在该 split 中缺失时避免除零
    w = len(y) / (num_classes * counts)  # 经典的 inverse-frequency 权重
    return torch.tensor(w / w.mean(), dtype=torch.float32, device=device)


def _write_history(history: list[dict], path: Path) -> None:
    """把逐 epoch 指标写成 CSV，方便 Excel 或画曲线脚本直接消费。

    Args:
        history: 每个 epoch 一个 dict，键集合需一致（以第一行的键作为表头）。
        path: 输出 CSV 路径；父目录不存在时自动创建。

    Returns:
        None。history 为空时直接返回，不创建空文件。

    Raises:
        OSError: 目录不可写或磁盘已满（注意本机 /tmp 空间很小，别往那里写）。
    """
    ensure_dir(path.parent)
    if not history:  # 极端情况下（0 轮训练）不生成只有表头的空文件
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


# --------------------------------------------------------------------- 特征模式
def train_feature_mode(
    cfg: dict, df: pd.DataFrame, label_map: dict, device: torch.device
) -> tuple[dict, list[dict]]:
    """feature 模式：冻结主干抽一次特征，再在特征上训练轻量分类头。

    与 finetune 的流程差异：
        1. 先调用 extract_all_splits() 对 train/val 全量做一次 CNN 前向并落盘缓存，
           之后每个 epoch 只在前向结果（N×D 矩阵）上迭代，不再解码图片、不再跑主干；
        2. 优化器只持有 head 参数，backbone 完全不更新；
        3. 因此 CPU 上也能快速跑完，而且反复调 lr/epochs 不会重复付出 CNN 成本。

    Args:
        cfg: 已合并命令行覆盖项的配置字典。
        df: manifest 明细，必须包含 split / label_idx 等列。
        label_map: 电影名 → 类别索引。
        device: 已解析好的运行设备（特征抽取仍然需要它）。

    Returns:
        dict，包含：
            model: 合并好分类头、主干已冻结的完整网络，可直接保存或推理；
            metrics: 最佳 epoch 的验证指标与数据规模；
            history: 逐 epoch 指标列表；
            feature_data: 各 split 的缓存特征，供上层（如 sklearn 基线）复用。

    Raises:
        RuntimeError: 训练集为空（数据划分或 manifest 异常）。
        FileNotFoundError: manifest / label_map 缺失（由 load_manifest 抛出）。
    """
    n_cls = len(label_map)
    # 一次前向 + 落盘；命中缓存时直接读 npz，不重复跑 CNN
    data = extract_all_splits(cfg, df, label_map, device=device)
    # from_numpy 与 numpy 共享内存，.float()/.long() 会复制成 torch 自己的张量，
    # 避免后续切片搬运时出现只读数组或 dtype 不匹配的问题
    Xtr = torch.from_numpy(data["train"]["X"]).float()
    ytr = torch.from_numpy(data["train"]["y"]).long()
    Xva = torch.from_numpy(data["val"]["X"]).float()
    yva = torch.from_numpy(data["val"]["y"]).long()

    if len(ytr) == 0:
        raise RuntimeError("训练集为空，请检查数据划分或 manifest")
    if len(yva) == 0:
        # 没有 val 时用 train 顶替，只为跑通链路；指标会虚高，不能当验收依据
        log.warning("验证集为空，改用训练集做验证（仅用于冒烟测试，指标会虚高）")
        Xva, yva = Xtr, ytr

    seed_everything(int(get_value(cfg, "project.seed", 42)))
    full_model = build_model(cfg, n_cls)  # 用于取得与特征匹配的网络结构
    # 只保留分类头，主干用完立即丢弃 —— CPU 上没必要常驻一份主干权重
    head = copy.deepcopy(full_model.head).to(device)
    feature_dim = full_model.feature_dim
    del full_model

    epochs = int(get_value(cfg, "train.epochs", 60))
    bs = int(get_value(cfg, "train.batch_size", 64))
    lr = float(get_value(cfg, "train.lr", 1e-3))
    wd = float(get_value(cfg, "train.weight_decay", 1e-4))
    patience = int(get_value(cfg, "train.patience", 10))
    # min_delta 是“算作改善”的最小幅度：小于它的波动视为噪声，避免早停计数器反复清零
    min_delta = float(get_value(cfg, "train.min_delta", 5e-4))
    smoothing = float(get_value(cfg, "train.label_smoothing", 0.05))
    grad_clip = float(get_value(cfg, "train.grad_clip", 5.0))

    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    # 监控 val_loss，连续 3 轮不降就把 lr 砍半（与早停相互配合：先降 lr 再停）
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    # 类别权重抵消长尾；label_smoothing 抑制过度自信，改善 unknown 拒识的置信度分布
    criterion = nn.CrossEntropyLoss(
        weight=_class_weights(data["train"]["y"], n_cls, device), label_smoothing=smoothing,
    )

    best_loss, best_epoch, best_state = float("inf"), -1, None
    best_row: dict = {"val_top1": 0.0, "val_top5": 0.0}
    stale = 0  # 连续未改善的 epoch 数，达到 patience 就早停
    history: list[dict] = []
    log.info("特征模式训练：X_train=%s X_val=%s | epochs=%d batch=%d lr=%g",
             tuple(Xtr.shape), tuple(Xva.shape), epochs, bs, lr)

    for epoch in range(1, epochs + 1):
        head.train()
        perm = torch.randperm(len(ytr))  # 每个 epoch 重新打乱，避免固定顺序带来的周期性
        total_loss, seen = 0.0, 0
        for start in range(0, len(ytr), bs):
            idx = perm[start:start + bs]
            xb, yb = Xtr[idx].to(device), ytr[idx].to(device)
            optimizer.zero_grad(set_to_none=True)  # 置 None 比填 0 省一次显存/内存写
            loss = criterion(head(xb), yb)
            loss.backward()
            if grad_clip > 0:  # 特征维度高时偶发大梯度，裁剪可防单个 batch 把权重带飞
                nn.utils.clip_grad_norm_(head.parameters(), grad_clip)
            optimizer.step()
            total_loss += loss.item() * len(idx)  # 按样本数加权，最后一个不满 batch 也公平
            seen += len(idx)

        head.eval()
        with torch.no_grad():
            val_logits = head(Xva.to(device))
            # 验证 loss 不加类别权重：它反映真实分布下的表现，用于早停/调 lr
            val_loss = float(nn.functional.cross_entropy(val_logits, yva.to(device)).item())
            val_top1 = _topk(val_logits, yva.to(device), 1)
            val_top5 = _topk(val_logits, yva.to(device), 5)
        scheduler.step(val_loss)
        row = {
            "epoch": epoch,
            "train_loss": round(total_loss / max(seen, 1), 6),
            "val_loss": round(val_loss, 6),
            "val_top1": round(val_top1, 6),
            "val_top5": round(val_top5, 6),
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        if epoch % 5 == 0 or epoch == 1:  # 稀疏打日志：CPU 上每轮都打会刷屏
            log.info("epoch %3d | train %.4f | val %.4f | top1 %.3f top5 %.3f | lr %.2e",
                     epoch, row["train_loss"], val_loss, val_top1, val_top5, row["lr"])

        # 只有改善超过 min_delta 才刷新最佳；best_state 立刻拷到 CPU，
        # 既避免长期占着 GPU 显存，也防止后续 epoch 原地更新污染已保存的权重
        if val_loss < best_loss - min_delta:
            best_loss, best_epoch = val_loss, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            best_row = row
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                log.info("早停：%d 轮未改善，最佳 epoch=%d val_loss=%.4f", patience, best_epoch, best_loss)
                break

    if best_state is None:  # 理论上不会发生；兜底用最后一轮权重，保证能产出 checkpoint
        best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        best_epoch = epochs

    # 把训练好的分类头合并回完整网络，得到一个可直接推理的 checkpoint
    model = build_model(cfg, n_cls)
    model.head.load_state_dict(best_state)
    # 冻结主干并让其保持 eval：推理只需前向，也避免 BatchNorm 用到 batch 统计
    model.freeze_backbone(True)
    metrics = {
        "mode": "feature", "best_epoch": best_epoch, "best_val_loss": round(best_loss, 6),
        "val_top1": best_row["val_top1"], "val_top5": best_row["val_top5"],
        "feature_dim": feature_dim, "n_train": int(len(ytr)), "n_val": int(len(yva)),
        "n_classes": n_cls,
    }
    return {"model": model, "metrics": metrics, "history": history, "feature_data": data}


# --------------------------------------------------------------------- 端到端微调
def train_finetune_mode(
    cfg: dict, df: pd.DataFrame, label_map: dict, device: torch.device
) -> dict:
    """finetune 模式：解冻主干，端到端微调整网。

    与 feature 模式的差别：每个 epoch 都要重新解码图片、跑完整前向与反向，
    CPU 上非常慢；换来的是主干特征可以针对电影画面重新适配。建议只在
    GPU 上、数据量不大、且 feature 模式精度不够时使用。

    Args:
        cfg: 配置字典；会读取 train.amp / train.amp_dtype 等训练参数。
        df: manifest 明细（含 split / label_idx / frame_path）。
        label_map: 电影名 → 类别索引。
        device: 运行设备；只有 CUDA 才会启用 AMP。

    Returns:
        dict，包含 model（已载入最佳权重）、metrics、history。

    Raises:
        FileNotFoundError: manifest 缺失。
        RuntimeError: state_dict 与网络结构不匹配（由 load_state_dict 抛出）。
    """
    n_cls = len(label_map)
    model = build_model(cfg, n_cls).to(device)
    # 与 feature 模式相反：这里必须解冻主干，否则整个微调退化成只训分类头
    model.freeze_backbone(False)
    train_loader = build_dataloader(df[df["split"] == "train"], cfg, train=True)
    val_df = df[df["split"] == "val"]
    # 没有 val 时退回 train：能跑通，但指标不可信，只适合冒烟
    val_loader = build_dataloader(val_df if len(val_df) else df[df["split"] == "train"], cfg, train=False)

    epochs = int(get_value(cfg, "train.epochs", 20))
    lr = float(get_value(cfg, "train.lr", 1e-3))
    wd = float(get_value(cfg, "train.weight_decay", 1e-4))
    patience = int(get_value(cfg, "train.patience", 5))
    smoothing = float(get_value(cfg, "train.label_smoothing", 0.05))
    # AMP 只在 CUDA 上有意义；CPU 的 bf16 前向通常更慢，直接走 fp32
    use_amp = bool(get_value(cfg, "train.amp", False)) and device.type == "cuda"
    # bf16 在 Ampere+（含 RTX 30/40/50 系）数值稳定、无需 GradScaler；
    # fp16 兼容性更广但需要 scaler。默认 fp16，Blackwell 建议改 bf16。
    amp_dtype = str(get_value(cfg, "train.amp_dtype", "fp16")).lower()
    if amp_dtype not in ("fp16", "bf16"):
        log.warning("未知 train.amp_dtype=%s，回退 fp16", amp_dtype)
        amp_dtype = "fp16"
    # 关键：Scaler 只为 fp16 服务。bf16 的指数位与 fp32 相同，梯度不会下溢，
    # 强行套 scaler 反而多一次乘除、还可能改变数值行为。
    use_scaler = use_amp and amp_dtype == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler) if use_scaler else None

    # 差异化学习率：主干只做轻微适配（0.1×），分类头从头学（1×），
    # 避免预训练特征被小数据集的大梯度迅速破坏
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": lr * 0.1},
        {"params": model.head.parameters(), "lr": lr},
    ], weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    # 直接从 manifest 取全量训练标签算权重，不依赖 loader 的采样顺序
    y_all = df[df["split"] == "train"]["label_idx"].to_numpy()
    criterion = nn.CrossEntropyLoss(
        weight=_class_weights(y_all, n_cls, device), label_smoothing=smoothing,
    )

    best_loss, best_epoch, best_state, stale = float("inf"), -1, None, 0
    best_row: dict = {"val_top1": 0.0, "val_top5": 0.0}
    history: list[dict] = []
    log.warning("端到端微调在纯 CPU 上会非常慢，建议先用 feature 模式验证数据链路")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, seen, correct = 0.0, 0, 0
        for images, ys, _ in train_loader:
            images, ys = images.to(device), ys.to(device)
            optimizer.zero_grad(set_to_none=True)
            # autocast 只包住前向与 loss；反向的缩放由 scaler 负责
            amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16 if amp_dtype == "bf16" else torch.float16)
            ctx = amp_ctx if use_amp else nullcontext()  # 不开 AMP 时用空上下文，保持代码路径统一
            with ctx:
                logits = model(images)
                loss = criterion(logits, ys)
            if scaler is not None:
                # fp16 路径：先放大 loss 再反向，防止小梯度在 fp16 下变成 0
                scaler.scale(loss).backward()
                # 裁剪前必须 unscale_ 还原真实梯度量级，否则裁的是放大后的数值
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), float(get_value(cfg, "train.grad_clip", 5.0)))
                # unscale_ 已执行，scaler.step 内部会跳过重复 unscale，直接做 inf/nan 检查后更新
                scaler.step(optimizer)
                scaler.update()
            else:
                # bf16 / fp32 路径：正常反向即可，不需要 scaler
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), float(get_value(cfg, "train.grad_clip", 5.0)))
                optimizer.step()
            total_loss += loss.item() * len(ys)
            seen += len(ys)
            correct += int((logits.argmax(1) == ys).sum())

        model.eval()
        vloss, vcorrect, vtop5, vseen = 0.0, 0, 0.0, 0
        with torch.no_grad():
            for images, ys, _ in val_loader:
                images, ys = images.to(device), ys.to(device)
                # 验证不启用 autocast：用 fp32 计算，指标更稳定、可跨次运行比较
                logits = model(images)
                vloss += float(nn.functional.cross_entropy(logits, ys).item()) * len(ys)
                vcorrect += int((logits.argmax(1) == ys).sum())
                vtop5 += _topk(logits, ys, 5) * len(ys)
                vseen += len(ys)
        val_loss = vloss / max(vseen, 1)
        row = {
            "epoch": epoch, "train_loss": round(total_loss / max(seen, 1), 6),
            "train_top1": round(correct / max(seen, 1), 6), "val_loss": round(val_loss, 6),
            "val_top1": round(vcorrect / max(vseen, 1), 6),
            "val_top5": round(vtop5 / max(vseen, 1), 6),
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        log.info("epoch %3d | train %.4f | val %.4f | val_top1 %.3f",
                 epoch, row["train_loss"], val_loss, row["val_top1"])
        scheduler.step(val_loss)
        # 微调每轮代价高，这里用固定 1e-4 作为改善阈值（不读 min_delta），
        # 既过滤浮点噪声，又不至于因为阈值过大而过早停止
        if val_loss < best_loss - 1e-4:
            best_loss, best_epoch = val_loss, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_row = row
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                log.info("早停，最佳 epoch=%d", best_epoch)
                break

    if best_state:  # 载回最佳权重，保证保存/评估的是验证 loss 最低的那一版
        model.load_state_dict(best_state)
    metrics = {
        "mode": "finetune", "best_epoch": best_epoch, "best_val_loss": round(best_loss, 6),
        "val_top1": best_row["val_top1"], "val_top5": best_row["val_top5"],
        "feature_dim": model.feature_dim, "n_classes": n_cls,
    }
    return {"model": model, "metrics": metrics, "history": history}


# --------------------------------------------------------------------- 统一入口
def run_training(cfg: dict, mode: str | None = None) -> dict:
    """训练统一入口：解析模式 → 选设备 → 载清单 → 训练 → 落盘 checkpoint 与报告。

    Args:
        cfg: 配置字典（已应用 --set 覆盖项）。
        mode: "feature" / "finetune"；None 时读 model.mode，缺省为 feature。

    Returns:
        dict: 训练报告，包含 project / mode / device / deep（最佳指标）/
        label_map / checkpoint；feature 模式且 classical.enabled 为真时，
        还包含 sklearn_baseline 与两者的 val_top1 差值对比。

    Raises:
        ValueError: mode 不属于 feature / finetune。
        FileNotFoundError: manifest 或 label_map 缺失。
        RuntimeError: 训练集为空。
        OSError: 输出目录不可写。
    """
    # 延迟导入：sklearn 启动慢，且纯 finetune/推理环境可能没装它；
    # 放在函数内可避免模块 import 阶段就因缺依赖失败
    from .models.classical import train_baseline

    mode = mode or str(get_value(cfg, "model.mode", "feature"))
    if mode not in ("feature", "finetune"):
        raise ValueError(f"未知训练模式：{mode}")

    # 固定随机种子要在建模型、建 DataLoader 之前，否则打乱顺序仍然不可复现
    seed_everything(int(get_value(cfg, "project.seed", 42)))
    device = resolve_device(
        str(get_value(cfg, "train.device", "auto")),
        int(get_value(cfg, "train.cpu_threads", 8)),
    )
    device_info = log_device(device, int(get_value(cfg, "train.cpu_threads", 8)))
    df, label_map = load_manifest(cfg)
    log.info("加载清单：%d 帧 / %d 类", len(df), len(label_map))

    # 两种模式签名一致，直接按字符串选函数，避免重复 if/else 分支
    result = (train_feature_mode if mode == "feature" else train_finetune_mode)(cfg, df, label_map, device)

    out_dir = cfg_path(cfg, "train.output_dir", mkdir=True)
    ckpt_path = out_dir / "best.pt"
    # checkpoint 是训练与推理之间唯一的契约：predict.py/evaluate.py 按这些键还原网络。
    # 新增字段可以，重命名/删除必须同步修改读取方，否则旧模型会加载失败。
    torch.save({
        "version": 1,
        "mode": mode,
        "model_state": result["model"].state_dict(),
        "backbone": result["model"].backbone_name,
        "feature_dim": result["model"].feature_dim,
        "num_classes": len(label_map),
        "label_map": label_map,
        "config": cfg,
        "metrics": result["metrics"],
        "device_info": device_info,
    }, ckpt_path)
    log.info("checkpoint → %s", ckpt_path)

    # 逐 epoch 历史单独存 CSV，便于画 loss 曲线或定位过拟合起点
    _write_history(result["history"], cfg_path(cfg, "train.history_file"))

    report = {
        "project": get_value(cfg, "project.name", "CineEye"),
        "mode": mode,
        "device": device_info,
        "deep": result["metrics"],
        "label_map": label_map,
        "checkpoint": str(ckpt_path),
    }
    # 用传统 sklearn 特征做对照基线：深度模型若没有明显胜出，说明数据或划分有问题
    if bool(get_value(cfg, "classical.enabled", True)) and mode == "feature":
        report["sklearn_baseline"] = train_baseline(cfg, df)
        delta = report["deep"]["val_top1"] - report["sklearn_baseline"]["val_top1"]
        log.info("深度特征 vs sklearn 基线：val_top1 %+.4f", delta)
    else:
        report["sklearn_baseline"] = None

    report_path = ROOT / "outputs" / "reports" / "train_report.json"
    save_json(report, report_path)
    log.info("训练报告 → %s", report_path)
    return report
