"""scikit-learn 传统机器学习基线：HOG + 颜色直方图 + LogisticRegression/SVM/RF。

用途：提供一条**完全不依赖深度学习**的分类链路——手工特征（HOG + HSV 直方图
+ 缩略灰度图）→ 标准化 → 线性/树模型。产出 ``classical_baseline.joblib``
与一组 val 指标，写入训练报告供对照。

为什么保留这个基线（改之前请先读完）：
    1. **标尺作用**：它的作用不是当主力，而是当「标尺」。
       如果深度特征模型跑不赢这个几十行就能写完的基线，说明数据划分或标注出了问题，
       而不是模型不够大。这在图片识别电影这类任务里是很有价值的 sanity check。
    2. **兜底能力**：本机没有 GPU 时，若 ImageNet 权重无法下载（离线/内网），
       深度链路会降级为随机初始化主干，效果可能还不如 HOG。
       基线保证「任何环境下都有一个可用的、指标可解释的分类器」。
    3. **OpenCV 版本锁定**：``cv2.HOGDescriptor`` 在 OpenCV 4.x 可用，
       但项目锁 ``opencv-python-headless>=4.9,<5``（requirements.txt），
       因为 OpenCV 5 移除了 HOGDescriptor 且部分 API 不兼容。为了不在升级 OpenCV
       时整个包导入失败，这里同时保留了纯 NumPy 的后备实现 :func:`_hog_numpy`。

被谁使用：
    - ``cineeye/train.py`` 在 ``classical.enabled=true`` 且 ``model.mode=feature``
      时调用 :func:`train_baseline`，结果放进训练报告；
    - ``tests/test_pipeline.py`` 测试 :func:`image_to_features` 的维度与确定性。

依赖：opencv-python-headless（cv2）、scikit-learn、joblib、numpy、pandas。
纯 CPU，且比深度模型更省资源；本机 /tmp 与 /dev/shm 只有 10MB，
但这里只在内存里处理几 MB 的特征矩阵，不写临时文件，因此不受影响。
（若未来把 ``classical.estimator`` 换成会 memmap 到 /dev/shm 的大规模模型，
需要重新评估这个前提。）

运维要点：
    1. 手工特征维度必须稳定：改了 :func:`image_to_features` 的实现，
       已缓存的 ``outputs/features/classical_*.npz`` 必须删除（缓存只校验样本数，
       不校验特征语义），否则会用旧维度的特征去 fit，报维度不匹配。
    2. ``classical.output`` 会被覆盖写入，历史基线结果不会自动留存，
       需要对比实验请手动备份 joblib。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from ..config import ROOT, get_value
from ..utils import ensure_dir, get_logger

log = get_logger()

def _make_hog():
    """OpenCV 5 起移除了 HOGDescriptor；缺失时返回 None，改用 NumPy 后备实现。

    返回值在模块级只算一次（见 ``_HOG``），避免每张图都做一次 hasattr 检查。

    Returns:
        ``cv2.HOGDescriptor`` 实例；当前 OpenCV 版本没有该 API 时返回 None。

    Raises:
        无。参数是常量，正常不会失败；只是 OpenCV 5 缺少该 API 时返回 None，
        由调用方（:func:`image_to_features`）决定是否走 NumPy 后备。
    """
    if hasattr(cv2, "HOGDescriptor"):
        # 参数：winSize=(64,64)、blockSize=(16,16)、blockStride=(8,8)、cellSize=(8,8)、9 bins
        # 输出维度 = 7*7 blocks * 2*2 cells * 9 bins = 1764
        return cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
    return None


_HOG = _make_hog()


def _hog_numpy(gray: np.ndarray) -> np.ndarray:
    """与 ``cv2.HOGDescriptor((64,64),(16,16),(8,8),(8,8),9)`` 近似等价的后备实现。

    8x8 cell / 9 个方向 bin / 16x16 block + L2-Hys 归一化，输出 1764 维。
    纯 NumPy 实现，避免 OpenCV 版本差异导致整个包无法导入。

    与 OpenCV 实现的细微差别：OpenCV 默认使用 L2-Hys（截断 0.2 后再归一化）
    且带三线性插值；这里用「按幅值比例分到相邻两个 bin」的线性插值近似，
    没有做 block 之间的高斯加权。数值不完全相同，但判别能力足够，
    且只在 OpenCV 缺少 HOG 时才启用，不会影响现有实验的可比性。

    Args:
        gray: 灰度图（任意尺寸，内部会 resize 到 64x64）。

    Returns:
        ``(1764,)`` 的 float32 特征向量。

    Raises:
        cv2.error: 输入为空或非单通道时（resize / Sobel 抛出）。
    """
    g = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=1)
    mag = np.hypot(gx, gy)                                     # (64, 64)
    ang = ((np.degrees(np.arctan2(gy, gx)) + 180.0) % 180.0) / 20.0  # 0..9

    cells = np.zeros((8, 8, 9), np.float32)
    for cy in range(8):
        for cx in range(8):
            sl = (slice(cy * 8, (cy + 1) * 8), slice(cx * 8, (cx + 1) * 8))
            m, a = mag[sl].ravel(), ang[sl].ravel()
            i0 = np.floor(a).astype(np.int64) % 9
            frac = a - np.floor(a)
            # 软分箱：一个像素的梯度幅值按角度比例分给相邻两个方向 bin，
            # 比硬分箱（直接 round）对小角度扰动更稳健。
            np.add.at(cells[cy, cx], i0, m * (1 - frac))
            np.add.at(cells[cy, cx], (i0 + 1) % 9, m * frac)

    blocks = []
    for by in range(7):
        for bx in range(7):
            v = cells[by:by + 2, bx:bx + 2].ravel()
            v = v / (np.linalg.norm(v) + 1e-8)
            v = np.clip(v, 0, 0.2)                            # L2-Hys
            # Hys 的第二步：截断后重新归一化，抑制强边缘主导整个描述子
            blocks.append(v / (np.linalg.norm(v) + 1e-8))
    return np.asarray(blocks, np.float32).reshape(-1)


def image_to_features(bgr: np.ndarray, use_hog: bool = True) -> np.ndarray:
    """单张图 → 一维特征向量（约 2.8k 维）。

    三段特征拼接，各自负责不同的判别线索：
        1. HOG（1764 维）—— 构图/纹理/人物姿态，对黑白片同样有效；
        2. HSV 直方图（64 维）—— 调色风格，是区分不同电影最有效的低层线索
           （很多片子的统一色调比剧情画面更稳定）；
        3. 32x32 缩略灰度（1024 维）—— 保留粗略的空间布局，
           相当于极低成本的颜色/结构模板。
    每一段都做了 L1/L2 归一化，避免某一段数值量级过大而独占标准化后的方差。

    Args:
        bgr: BGR 图像（cv2 读取的原始格式）。
        use_hog: 是否计算 HOG；关闭后维度减少 1764，可用于快速对比实验。
            注意开关必须与缓存文件对应，混用会导致维度不一致。

    Returns:
        一维 float32 特征向量：``use_hog=True`` 时 1764+64+1024=2852 维，
        ``False`` 时 1088 维。

    Raises:
        cv2.error: 输入为空或通道数不为 3 时。
    """
    # 统一缩到 64x64：尺寸一致才能让 HOG 与缩略图两段特征对齐
    img = cv2.resize(bgr, (64, 64), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blocks: list[np.ndarray] = []
    if use_hog:
        # OpenCV 可用就用 C++ 实现，否则走 NumPy 后备（见 _hog_numpy）
        h = (_HOG.compute(gray).reshape(-1).astype(np.float32)
             if _HOG is not None else _hog_numpy(gray))             # 1764 维
        blocks.append(h / (np.linalg.norm(h) + 1e-8))

    # HSV 而不是 RGB：H/S/V 分离后，直方图对亮度变化的鲁棒性更好。
    # H 取 32 bin（0-180），S/V 各 16 bin（0-256），合计 64 维。
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = np.concatenate([
        cv2.calcHist([hsv], [0], None, [32], [0, 180]).reshape(-1),
        cv2.calcHist([hsv], [1], None, [16], [0, 256]).reshape(-1),
        cv2.calcHist([hsv], [2], None, [16], [0, 256]).reshape(-1),
    ]).astype(np.float32)
    blocks.append(hist / (hist.sum() + 1e-8))  # L1 归一化：变成「颜色占比」分布

    tiny = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    blocks.append(tiny.reshape(-1))
    return np.concatenate(blocks).astype(np.float32)


def extract_handcrafted(df: pd.DataFrame, cfg: dict, split: str, cache: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """把某个 split 的图片转成特征矩阵，并缓存为 .npz，避免重复解码。

    Args:
        df: 含 ``split`` / ``frame_path`` / ``label_idx`` 的清单。
        cfg: 全局配置，读取 ``classical.hog``、``features.cache_dir``。
        split: ``"train"`` / ``"val"`` / ``"test"``。
        cache: 是否读写缓存；False 时强制重算（改了特征实现就该用它）。

    Returns:
        ``(X, y)``：X 为 ``(N, D)`` float32，y 为 ``(N,)`` int64。

    Raises:
        不抛异常。单张图读失败会用 128 灰图占位（与 FrameDataset 策略一致），
        保证坏图不会中断整条基线流程。
    """
    from ..config import path as cfg_path

    sub = df[df["split"] == split]
    cache_path = cfg_path(cfg, "features.cache_dir", mkdir=True) / f"classical_{split}.npz"
    # 缓存校验只比对样本数 len(y) == len(sub)，不校验特征实现版本。
    # 这是有意的轻量取舍：手工特征重算很快（几十秒），
    # 但没有感知到「换了 HOG 开关 / 改了直方图 bin 数」的情况，
    # 因此改动 image_to_features 后请手动删除 outputs/features/classical_*.npz。
    if cache and cache_path.exists():
        data = np.load(cache_path)
        if len(data["y"]) == len(sub):
            return data["X"], data["y"]

    use_hog = bool(get_value(cfg, "classical.hog", True))
    X, y = [], []
    for i, row in enumerate(sub.itertuples(index=False)):
        p = Path(row.frame_path)
        if not p.is_absolute():
            p = ROOT / p
        bgr = cv2.imread(str(p))
        if bgr is None:
            bgr = np.full((64, 64, 3), 128, dtype=np.uint8)
        X.append(image_to_features(bgr, use_hog))
        y.append(int(row.label_idx))
        if (i + 1) % 500 == 0:
            log.info("  %s 手工特征 %d/%d", split, i + 1, len(sub))
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    if cache:
        ensure_dir(cache_path.parent)
        np.savez_compressed(cache_path, X=X, y=y)
    return X, y


def build_classical_pipeline(cfg: dict):
    """按配置构造 sklearn 估计器（含必要的标准化）。

    三种选择及理由：
        - logreg（默认）：线性、快、可解释，配合 StandardScaler 在手工特征上
          通常就是最强基线；``class_weight="balanced"`` 抵消类别不均。
        - svm：``LinearSVC`` 本身不输出概率，外面套
          ``CalibratedClassifierCV`` 做 sigmoid 校准，才能和深度模型比较 top-k 概率。
          代价是要额外做 3 折交叉拟合，训练时间约为 logreg 的数倍。
        - rf：树模型对特征尺度不敏感，因此**不接** StandardScaler；
          ``max_features="sqrt"`` 是分类任务的常规设置。

    Args:
        cfg: 全局配置，读取 ``classical.estimator`` 与 ``classical.C``。

    Returns:
        可直接 ``fit/predict`` 的 sklearn 估计器（Pipeline 或裸分类器）。

    Raises:
        无。未知 ``estimator`` 值会静默回退到 logreg（保持向后兼容）。
    """
    estimator = str(get_value(cfg, "classical.estimator", "logreg")).lower()
    C = float(get_value(cfg, "classical.C", 1.0))
    if estimator == "svm":
        base = LinearSVC(C=C, class_weight="balanced", max_iter=3000, random_state=42)
        # 固定 random_state 与 cv=3：校准过程也要可复现
        clf = CalibratedClassifierCV(base, cv=3, method="sigmoid")
        return make_pipeline(StandardScaler(), clf)
    if estimator == "rf":
        clf = RandomForestClassifier(
            n_estimators=200, max_features="sqrt", n_jobs=-1,
            class_weight="balanced", random_state=42,
        )
        return clf  # 树模型不需要标准化
    # sklearn >=1.8 起 LogisticRegression 的 n_jobs 已无效，1.10 会移除，故不再传
    # max_iter=2000：手工特征量级差异大，默认 100 轮常常不收敛并刷 ConvergenceWarning
    clf = LogisticRegression(
        C=C, max_iter=2000, class_weight="balanced", random_state=42,
    )
    return make_pipeline(StandardScaler(), clf)


def train_baseline(cfg: dict, df: pd.DataFrame) -> dict:
    """训练基线并返回 val 指标，结果写入 joblib。

    Args:
        cfg: 全局配置，读取 ``classical.*`` 与 ``features.cache_dir``。
        df: 完整清单（含三个 split），内部自行切分。

    Returns:
        指标 dict：``estimator`` / ``val_top1`` / ``val_macro_f1`` /
        ``feature_dim`` / ``n_train``。会被塞进训练报告，用于和深度模型对比。

    Raises:
        ValueError: train 或 val 为空（sklearn ``fit`` 会抛）。
            每类至少有 1 段视频时不会发生，见 ``manifest._allocate`` 的保证。
    """
    from sklearn.metrics import accuracy_score, f1_score

    from ..config import path as cfg_path

    Xtr, ytr = extract_handcrafted(df, cfg, "train")
    Xva, yva = extract_handcrafted(df, cfg, "val")
    log.info("scikit-learn 基线：train=%s val=%s，特征维度=%d", Xtr.shape, Xva.shape, Xtr.shape[1])

    pipe = build_classical_pipeline(cfg)
    pipe.fit(Xtr, ytr)
    # 只在 val 上评估，test 集留到最终评估脚本，避免在这里「偷看」测试集
    pred = pipe.predict(Xva)
    metrics = {
        "estimator": str(get_value(cfg, "classical.estimator", "logreg")),
        "val_top1": float(accuracy_score(yva, pred)),
        # macro-F1 比 accuracy 更能反映类别不均下的真实水平，与深度模型口径一致
        "val_macro_f1": float(f1_score(yva, pred, average="macro", zero_division=0)),
        "feature_dim": int(Xtr.shape[1]),
        "n_train": int(len(ytr)),
    }
    out = cfg_path(cfg, "classical.output")
    ensure_dir(out.parent)
    joblib.dump({"pipeline": pipe, "metrics": metrics}, out)
    log.info("基线完成：val_top1=%.4f macro_f1=%.4f → %s",
             metrics["val_top1"], metrics["val_macro_f1"], out)
    return metrics
