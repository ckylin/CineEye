"""按时间戳从视频里采样帧，并过滤重复/黑屏画面。

用途：把 ``data/videos/<电影名>/*.mp4`` 解码成 JPEG 帧落盘，并返回帧级记录
（相对路径 / 标签 / video_id / 时间戳），供 manifest 阶段汇总成 CSV。
这是训练数据链路的第一个环节，也是「目录约定」的唯一权威定义。

目录约定（训练数据的唯一入口约定）：
    data/videos/<电影名>/<任意视频文件>.mp4
        ↓
    data/frames/<电影名>/<视频文件名去扩展名>/000000.jpg

每个视频单独一个子目录，这样后面可以按 ``video_id`` 分组划分数据集，
避免同一段视频的相邻帧同时出现在训练集和验证集里（这是这类任务最常见的数据泄漏）。

被谁使用：
    - ``scripts/01_extract_frames.py`` 调用 :func:`extract_all` 做离线批量抽帧；
    - ``cineeye/predict.py``、``cineeye/identify.py`` 只调用
      :func:`sample_frames_from_video` 做在线推理（帧留在内存，不落盘）。

依赖：opencv-python（cv2）、numpy、``cineeye.config``、``cineeye.utils``。
抽帧本身是纯 CPU 的 I/O + 解码操作，本机无 GPU 也完全不受影响；
若在装有 RTX 5070（Blackwell sm_120）的机器上跑，此模块同样只用 CPU。

运维要点：
    1. 本机 ``/tmp`` 与 ``/dev/shm`` 都只有 10MB，而帧是写到项目内的
       ``data/frames/``（真实磁盘），所以抽帧不会撞到共享内存上限；
       但**不要**把 ``extract`` 的中间产物或缓存目录指到 /tmp、/dev/shm。
    2. 改 ``extract.fps`` / ``max_frames`` / ``resize_long_edge`` 会改变帧的数量、
       尺寸与 ``frame_index`` 编号，必须重新跑 01 → 02 → 特征抽取，
       否则旧 manifest 与旧特征缓存会对不上。
    3. ``skip_similar`` / ``skip_black`` 属于「内容过滤」，阈值调大调小会直接
       改变每个视频最终保留的帧数，进而影响类别平衡，改完建议看一遍抽帧日志。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from ..config import ROOT, get_value
from ..config import path as cfg_path
from ..utils import ensure_dir, get_logger

log = get_logger()

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _relative_to_root(p: Path) -> str:
    """把绝对路径转成「相对项目根目录」的 POSIX 字符串。

    清单里统一存相对路径（如 ``data/frames/xxx/000000.jpg``），
    这样整个项目目录可以整体搬移/备份，manifest.csv 不会失效。

    Args:
        p: 任意路径（通常是刚写出的帧文件）。

    Returns:
        相对 ``ROOT`` 的 POSIX 风格路径；若 ``p`` 不在项目根目录下
        （例如用户把 frames 指到了外部磁盘），则退化为绝对路径字符串。
    """
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return p.resolve().as_posix()


def _thumbnail(frame: np.ndarray, size: int = 32) -> np.ndarray:
    """把 BGR 帧压成 32x32 灰度缩略图，作为「画面指纹」用于黑屏/重复判断。

    只用于判定，不参与训练。缩到 32x32 后单帧只有 1024 个 float，
    比较成本极低；用 INTER_AREA 而非 INTER_LINEAR 是为了在缩小时做区域平均，
    抗混叠、对噪声不敏感，避免高压缩视频的块效应干扰相似度。

    Args:
        frame: BGR 图像（H, W, 3），uint8。
        size: 缩略图边长，默认 32。

    Returns:
        ``(size, size)`` 的 float32 灰度图。

    Raises:
        cv2.error: 当 ``frame`` 为空、通道数非法或尺寸为 0 时。
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    return small


def _is_similar(a: np.ndarray, b: np.ndarray, threshold: float) -> bool:
    """灰度缩略图的余弦相似度；对光照变化不敏感，适合抓静态镜头。

    实现要点：先各自减去均值（零均值化），此时余弦相似度等价于
    Pearson 相关系数。这样「整体变亮/变暗」不会拉低相似度，
    只有画面**结构**变化才会——正好对上电影里渐亮渐暗的转场。
    分母加 1e-8 防止纯色帧（方差为 0、范数为 0）除零。

    Args:
        a: 缩略图 A（一维化前），来自 :func:`_thumbnail`。
        b: 缩略图 B，形状与 ``a`` 相同。
        threshold: 判定为「同一画面」的相似度阈值，配置项
            ``extract.skip_similar_threshold``，默认 0.985。

    Returns:
        True 表示两帧几乎相同，应跳过后者。

    Raises:
        ValueError: 两个数组元素个数不一致时由 NumPy 广播机制抛出。
    """
    va, vb = a.ravel() - a.mean(), b.ravel() - b.mean()
    denom = np.linalg.norm(va) * np.linalg.norm(vb) + 1e-8
    return float(va @ vb / denom) >= threshold


def _resize_long_edge(frame: np.ndarray, long_edge: int) -> np.ndarray:
    """按「长边 = long_edge」等比缩放，只缩小不放大。

    Args:
        frame: BGR 图像。
        long_edge: 目标长边像素；``<= 0`` 表示保持原分辨率（配置里 0 即关闭）。

    Returns:
        缩放后的图像（新数组；未缩放时返回原对象）。

    Raises:
        cv2.error: 尺寸计算异常时。
    """
    if long_edge <= 0:
        return frame
    h, w = frame.shape[:2]
    scale = long_edge / max(h, w)
    if scale >= 1.0:  # 只缩小不放大
        return frame
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def extract_one_video(
    video_path: str | Path,
    label: str,
    frames_dir: str | Path,
    cfg: dict,
) -> list[dict]:
    """抽单个视频，返回帧记录列表（每条含相对路径、标签、video_id、时间戳）。

    采样策略是「按时间戳」而非「按固定帧号」：用源 fps / 目标 fps 得到整数步长，
    因此对 23.976 / 25 / 29.97 等不同帧率的片源都能得到大致相同的每秒帧数。

    Args:
        video_path: 源视频路径。
        label: 电影名（即 ``data/videos/`` 下的一级目录名），同时作为分类标签。
        frames_dir: 帧输出根目录（``data/frames``），实际写到
            ``frames_dir/label/<视频文件名去扩展名>/``。
        cfg: 全局配置 dict，本函数读取 ``extract.*`` 一组键（fps、max_frames、
            resize_long_edge、jpeg_quality、skip_similar、skip_black 等）。

    Returns:
        记录列表，每条为 dict：
            ``frame_path``   相对项目根的图片路径
            ``label``        电影名
            ``video_id``     ``"<label>/<视频 stem>"``，划分数据集时按它分组
            ``frame_index``  该视频内第几张有效帧（从 0 递增）
            ``timestamp_sec`` 该帧在源视频中的时间戳（秒）

    Raises:
        本函数不向上抛异常：打不开视频只记 warning 并返回空列表，
        单帧解码失败会被跳过。这样批量抽帧时坏片不会中断整批任务。
    """
    video_path = Path(video_path)
    fps_target = float(get_value(cfg, "extract.fps", 1.0))
    max_frames = int(get_value(cfg, "extract.max_frames", 200))
    long_edge = int(get_value(cfg, "extract.resize_long_edge", 360))
    quality = int(get_value(cfg, "extract.jpeg_quality", 90))
    skip_similar = bool(get_value(cfg, "extract.skip_similar", True))
    sim_thr = float(get_value(cfg, "extract.skip_similar_threshold", 0.985))
    skip_black = bool(get_value(cfg, "extract.skip_black", True))
    black_thr = float(get_value(cfg, "extract.black_mean_threshold", 12.0))

    # 输出目录按「电影名/视频名」两级拆分，video_id 与目录结构一一对应：
    # 后续 manifest 直接由目录名反推 video_id，两者必须保持同构。
    out_dir = ensure_dir(Path(frames_dir) / label / video_path.stem)
    video_id = f"{label}/{video_path.stem}"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        # 坏文件/不支持的编码：只告警不抛异常，批量任务继续跑
        log.warning("无法打开视频，已跳过：%s", video_path)
        return []

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if src_fps <= 0 or np.isnan(src_fps):
        # 部分容器（如某些 .ts / 流式 mp4）不写 fps 元数据，按 25 兜底；
        # 此时时间戳会有偏差，但不影响训练（我们只用帧序和 video_id）。
        src_fps = 25.0
    # 采样步长：每 step 帧取 1 帧。round 而非 int() 是为了避免 29.97/25 这类
    # 非整数比值被系统性向下取整，导致长时间视频的采样率悄悄偏低。
    step = max(1, int(round(src_fps / max(fps_target, 1e-6))))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    records: list[dict] = []
    # prev_thumb 始终指向「上一张**已保留**的帧」，而不是上一张解码帧。
    # 这样连续多帧画面不动时只会保留第一张；一旦画面真的变了，
    # 比较基准也随之更新，缓慢摇镜不会被误判成静止而全部丢弃。
    prev_thumb: np.ndarray | None = None
    frame_no = 0  # 已 grab 的帧数，用于把采样点换算回原片时间戳
    saved = 0     # 已落盘的有效帧数，同时用作文件名序号
    # grab() 只推进解码器不取像素，retrieve() 才做颜色转换，比逐帧 read() 省 CPU
    while cap.grab():
        take = (frame_no % step == 0)
        frame_no += 1
        if not take:
            continue
        ok, frame = cap.retrieve()
        if not ok or frame is None:
            continue

        thumb = _thumbnail(frame)
        # 黑屏阈值 12 的依据：解码后的纯黑/近黑帧灰度均值通常在 0~5，
        # 片头片尾的极暗渐隐约 8~15。取 12 能滤掉纯黑与接近全黑的过渡帧，
        # 又不会误伤夜景戏（夜景即使暗也有灯光高光，均值一般 > 25）。
        # 注意：这里判的是**均值**而非最大亮度，所以「全黑 + 少量字幕」的帧
        # 只要字幕面积够小仍然会被过滤——这正是我们想要的，因为这类帧
        # 对「识别是哪部电影」几乎不含信息量。
        if skip_black and float(thumb.mean()) < black_thr:
            continue
        # 相似帧去重的作用：电影 1 fps 采样时，一个 10 秒的静态镜头会产出 ~10 张
        # 几乎相同的图。它们既浪费磁盘，也会让模型把「该镜头停留时长」当成特征
        # （某些电影静止镜头多，会形成伪相关）。阈值 0.985 对应 Pearson 相关，
        # 只滤掉肉眼难辨的重复，轻微运镜和人物移动都会被保留。
        if skip_similar and prev_thumb is not None and _is_similar(thumb, prev_thumb, sim_thr):
            continue
        prev_thumb = thumb  # 只有真正保留的帧才更新基准，见上文 prev_thumb 说明

        frame = _resize_long_edge(frame, long_edge)
        # 文件名用 6 位零填充的连续序号：字典序即时间序，
        # 且 manifest 里 frame_index 与文件名一一对应，便于人工抽查。
        out_path = out_dir / f"{saved:06d}.jpg"
        # 显式指定 JPEG 质量（默认 95 会让 data/frames 体积翻倍，且训练用不到那么高）
        cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        records.append({
            "frame_path": _relative_to_root(out_path),
            "label": label,
            "video_id": video_id,
            "frame_index": saved,
            # frame_no 已自增 1，所以当前帧的序号是 frame_no-1；
            # 用源 fps 而不是采样 fps 才能得到真实播放时间。
            "timestamp_sec": round((frame_no - 1) / src_fps, 3),
        })
        saved += 1
        # 单视频帧数上限：长片按 1 fps 抽会到上万张。默认 200 既控制磁盘
        # （本机 /tmp、/dev/shm 仅 10MB，虽然帧写在项目磁盘上，但缓存与临时
        # 文件仍要警惕），也避免同一部电影在训练集里过度主导。
        if saved >= max_frames:
            break
    cap.release()

    log.info("抽帧 %-28s → %4d 帧 (源 %d 帧 @ %.1f fps, 采样步长 %d)",
             f"{label}/{video_path.name}", saved, total or frame_no, src_fps, step)
    return records


def iter_videos(videos_dir: str | Path, cfg: dict) -> Iterator[tuple[Path, str]]:
    """遍历 ``<videos_dir>/<电影名>/<视频>``，产出 (视频路径, 标签)。

    只认「两层」结构：一级目录名 = 电影名（标签），二级文件 = 视频。
    以 ``.`` 开头的目录/文件会被跳过（macOS 的 ``.DS_Store``、编辑器的临时目录等）。

    Args:
        videos_dir: 视频根目录（``data/videos``）。
        cfg: 全局配置，读取 ``extract.video_extensions`` 决定哪些后缀算视频。

    Yields:
        ``(video_path, label)``。顺序为「标签名排序 + 文件名排序」，
        保证同一份数据多次运行得到完全一致的抽帧顺序（可复现）。

    Raises:
        FileNotFoundError: ``videos_dir`` 不存在时立即抛出，避免用户误以为
            「没有视频」而以为抽帧成功。
    """
    videos_dir = Path(videos_dir)
    exts = set(get_value(cfg, "extract.video_extensions", [".mp4"]))
    if not videos_dir.exists():
        raise FileNotFoundError(f"视频目录不存在：{videos_dir}")
    for label_dir in sorted(p for p in videos_dir.iterdir() if p.is_dir() and not p.name.startswith(".")):
        for video in sorted(label_dir.iterdir()):
            if video.is_file() and video.suffix.lower() in exts:
                yield video, label_dir.name


def extract_all(cfg: dict) -> list[dict]:
    """抽全部视频。若 videos 目录为空则不报错，方便用户直接放入已抽好的图片。

    Args:
        cfg: 全局配置。路径取自 ``data.videos_dir`` / ``data.frames_dir``。

    Returns:
        所有视频的帧记录拼接列表（结构同 :func:`extract_one_video`）。

    Raises:
        FileNotFoundError: ``data/videos`` 目录本身不存在（由 :func:`iter_videos` 抛出）。
        注意「目录存在但没有视频」不算异常，只会打 warning。
    """
    videos_dir = cfg_path(cfg, "data.videos_dir")
    frames_dir = cfg_path(cfg, "data.frames_dir", mkdir=True)
    records: list[dict] = []
    n_videos = 0
    for video, label in iter_videos(videos_dir, cfg):
        n_videos += 1
        records.extend(extract_one_video(video, label, frames_dir, cfg))
    if n_videos == 0:
        log.warning("未在 %s 找到视频；若你已有图片帧，可直接放到 %s/<电影名>/ 下并跳到第二步",
                    videos_dir, frames_dir)
    else:
        log.info("共处理 %d 个视频，产出 %d 张有效帧", n_videos, len(records))
    return records


def sample_frames_from_video(video_path: str | Path, num_frames: int = 16) -> list[np.ndarray]:
    """推理用：从一段视频均匀抽 ``num_frames`` 帧到内存（BGR ndarray，不落盘）。

    与 :func:`extract_one_video` 的区别：这里不做去重/黑屏过滤，也不写磁盘。
    推理需要「均匀覆盖全片」的采样，才有利于后续概率聚合（mean_prob 等）；
    过滤掉黑屏反而可能让片头黑场特别长的片子抽不到帧。

    实现上用 ``CAP_PROP_POS_FRAMES`` 随机定位而不是顺序解码，
    对长视频可以只解码需要的十几帧，代价是部分编码（如 H.264 的 B 帧）
    定位会有几帧误差，对「识别是哪部电影」无影响。

    Args:
        video_path: 视频路径。
        num_frames: 采样帧数，默认 16（对应 ``inference.num_frames``）。

    Returns:
        BGR 帧列表，长度 <= ``num_frames``；顺序与时间顺序一致。

    Raises:
        RuntimeError: 视频打不开，或整段视频没有一帧能解码。
            （在线推理场景宁可报错，也不要静默返回空列表。）
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames: list[np.ndarray] = []
    if total > 0:
        # linspace 保证首帧与末帧都被取到，中间等距分布
        indices = np.linspace(0, max(total - 1, 0), num_frames).astype(int).tolist()
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"视频中没有可解码的帧：{video_path}")
    return frames
