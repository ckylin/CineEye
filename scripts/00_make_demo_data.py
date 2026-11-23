"""生成合成演示视频，用于在没有真实片源时跑通/验证整条流水线。

用途：不依赖任何外部素材，用 OpenCV 现场画出一批"电影片段"（每部片一个固定色调 +
一种几何母题），让 01~05 全流程可以在空机器上先跑通一遍，用来验证代码/环境而不是效果。
注意：合成数据的指标（准确率接近 100%）不代表真实电影场景，只能当冒烟测试。

典型调用：
    python scripts/00_make_demo_data.py                 # 4 部 × 5 段
    python scripts/00_make_demo_data.py --movies 6 --clips 6

输入/输出：
    输出：data/videos/<电影名>/clip_XX.mp4（编码器不可用时退回 .avi）
    读取：无（纯合成，不需要联网）

运维要点：
    1. 每段用不同的 phase 起点，保证训练/验证/测试拿到的是不同片段，
       否则同一段视频的相似帧会跨集合出现，指标虚高。
    2. 每帧都会叠时间码与噪点，避免被 01_extract_frames 的 skip_similar 过滤掉。
    3. 输出目录默认 data/videos；重跑会覆盖同名文件，但不会清理多余旧片。
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from cineeye.config import ROOT  # noqa: E402

# 演示用"电影名"：只用于目录名与类别名，与真实片名无关
MOVIE_NAMES = ["星际回响", "雨夜列车", "赤色荒漠", "深海之城", "雾中灯塔", "玻璃森林"]
# 每部片一对 (背景色, 前景色)，OpenCV 是 BGR 顺序 —— 注释里写的颜色名按 BGR 读
PALETTE = [
    ((18, 32, 78), (240, 240, 255)),      # 深蓝 + 白
    ((22, 58, 40), (60, 220, 240)),       # 深绿 + 黄青
    ((28, 28, 110), (80, 220, 230)),      # 暗红(BGR) + 青
    ((92, 40, 24), (255, 220, 180)),      # 深海蓝褐 + 浅蓝
    ((70, 60, 35), (230, 245, 255)),      # 雾灰 + 白
    ((60, 35, 75), (180, 240, 200)),      # 紫 + 浅绿
]


def make_frame(movie_idx: int, t: int, size: int, rng: np.random.Generator) -> np.ndarray:
    """合成单帧：固定色调背景 + 随时间漂移的几何母题 + 噪点与时间码。

    参数 t 既是"时间"也是空间相位来源：同一部片的不同片段只需传不同 t 起点，
    就能得到"像同一部电影、但画面不重复"的帧，满足训练/验证/测试互不重叠的要求。
    """
    bg, fg = PALETTE[movie_idx % len(PALETTE)]
    h = w = size
    img = np.zeros((h, w, 3), np.uint8)
    # 缓慢变化的渐变背景，模拟镜头色调
    for y in range(h):
        k = 0.75 + 0.25 * np.sin(y / h * np.pi + t * 0.05)
        img[y, :, :] = np.clip(np.array(bg, np.float32) * k, 0, 255).astype(np.uint8)

    # 母题中心随时间做正弦漂移（而不是随机跳变），保证帧间是"同一镜头的连续运动"
    cx = int(w * (0.5 + 0.22 * np.sin(t * 0.07 + movie_idx)))
    cy = int(h * (0.5 + 0.20 * np.cos(t * 0.06 + movie_idx)))
    r = int(min(h, w) * 0.18)
    kind = movie_idx % 4  # 4 种母题循环使用：同心圆 / 三角 / 竖条 / 星形
    if kind == 0:
        cv2.circle(img, (cx, cy), r, fg, -1)
        cv2.circle(img, (cx, cy), int(r * 0.55), bg, -1)
        cv2.circle(img, (cx, cy), int(r * 1.6), fg, 3)
    elif kind == 1:
        pts = np.array([[cx, cy - r], [cx - r, cy + r], [cx + r, cy + r]], np.int32)
        cv2.fillPoly(img, [pts], fg)
        cv2.polylines(img, [pts], True, fg, 2)
        cv2.line(img, (0, cy), (w, cy), fg, 2)
    elif kind == 2:
        for i in range(5):
            x = (cx + i * w // 5) % w  # 取模实现横向环绕
            cv2.rectangle(img, (x, cy - r), (x + w // 12, cy + r), fg, -1)
    else:
        cv2.drawMarker(img, (cx, cy), fg, cv2.MARKER_STAR, r * 2, 6)
        for i in range(4):
            cv2.line(img, (cx - r * 2, cy - r + i * r // 2),
                     (cx + r * 2, cy - r + i * r // 2), fg, 1)

    # 每帧加入轻微噪点与时间码，保证相邻帧不完全相同
    # （01_extract_frames 的 skip_similar 会丢弃近似重复帧，全等帧会被整段丢掉）
    noise = rng.integers(-6, 7, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    cv2.putText(img, f"{movie_idx:02d}-{t:03d}", (8, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, fg, 1, cv2.LINE_AA)
    return img


def _writer(path: Path, fps: int, size: int):
    """按 mp4v → XVID 的顺序尝试创建 VideoWriter，返回 (writer, 实际路径)。

    为什么要逐个试：opencv-python-headless 在不同发行版/编译选项下带的编码器不同，
    mp4v 常常不可用；退到 .avi + XVID 基本总能成功。全部失败才报错。
    """
    for fourcc, suffix in (("mp4v", ".mp4"), ("XVID", ".avi")):
        p = path.with_suffix(suffix)  # 先按候选后缀定路径，避免写出 .mp4 却装了 XVID 流
        w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*fourcc), fps, (size, size))
        if w.isOpened():
            return w, p
        w.release()  # 打不开要显式释放，否则句柄泄漏
    raise RuntimeError("当前 OpenCV 没有可用的视频编码器（mp4v / XVID 均不可用）")


def main() -> None:
    """命令行入口：按 --movies/--clips/--frames 批量生成合成视频。"""
    ap = argparse.ArgumentParser(description="影瞳 · 生成合成演示视频")
    ap.add_argument("--movies", type=int, default=4, help="电影（类别）数量")
    ap.add_argument("--clips", type=int, default=5, help="每部电影的片段数，>=3 才能合理划分")
    ap.add_argument("--frames", type=int, default=80, help="每个片段的帧数")
    ap.add_argument("--size", type=int, default=256, help="画面边长")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    out = Path(args.out) if args.out else ROOT / "data" / "videos"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)  # 固定种子：同样的参数得到同样的数据，便于复现
    total = 0
    for m in range(min(args.movies, len(MOVIE_NAMES))):
        # 一部电影 = 一个类别 = 一个子目录；目录名即标签（02_build_manifest 靠目录名反推）
        label_dir = out / MOVIE_NAMES[m]
        label_dir.mkdir(parents=True, exist_ok=True)
        for c in range(args.clips):
            writer, path = _writer(label_dir / f"clip_{c + 1:02d}", args.fps, args.size)
            # 每个片段随机一个时间相位：让各片段画面不同，从而能划分到不同 split
            phase = rng.integers(0, 1000)
            for t in range(args.frames):
                writer.write(make_frame(m, int(t + phase), args.size, rng))
            writer.release()
            total += 1
    print(f"已生成 {total} 段合成视频 → {out}")
    print("提示：这是仅为验证流水线可用的合成数据，识别指标不代表真实电影场景。")


if __name__ == "__main__":
    main()
