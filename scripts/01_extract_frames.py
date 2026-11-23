"""第 1 步：把 data/videos/<电影名>/*.mp4 抽成帧。

用途：视频是"流"，训练需要"帧"。本步按时间戳均匀采样帧、缩放并落盘成
data/frames/<电影名>/<视频名>/000001.jpg，同时跳过全黑帧与近似重复帧，
既降磁盘占用，也避免静态镜头在训练集里被过度计数。

典型调用：
    python scripts/01_extract_frames.py
    python scripts/01_extract_frames.py --set extract.fps=2 --set extract.max_frames=400

输入/输出：
    读取：data/videos/**/*.{mp4,mkv,avi,...}（目录名即类别标签）
    输出：data/frames/<标签>/<video_id>/*.jpg，以及控制台统计；
          不写 manifest（那是第 2 步的事）

运维要点：
    1. 目录结构约定：<videos_dir>/<电影名>/<视频文件>，标签由一级目录名决定，
       改目录名等于改类别名，必须重跑 02 及之后所有步骤（旧 manifest 会对不上）。
    2. 输出目录 "<视频文件名去扩展名>" 就是后续的 video_id，是分组划分的依据，
       所以不要把两个不同视频命名成同一名字。
    3. 本机 /tmp 与 /dev/shm 只有 10MB：抽帧全程用 OpenCV 流式读取，不落临时文件；
       但仍建议保持 TMPDIR 指向项目 .tmp。
    4. --device 与 --set a.b.c=value 都走 cineeye.cli 的公共参数：
       --set 逐项覆盖配置（后写覆盖先写），--device 最后写入 cfg.train.device，
       优先级最高，便于同一份配置在 CPU 笔记本与 GPU 机器间切换。
"""

import sys
from pathlib import Path

# 直接执行脚本时 sys.path[0] 是 scripts/，必须补上项目根才能 import cineeye
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cineeye.cli import base_parser, load_cfg  # noqa: E402
from cineeye.data.extract_frames import extract_all


def main() -> None:
    """解析参数 → 读配置 → 批量抽帧 → 打印帧数与下一步提示。"""
    # base_parser 提供 --config / --device / --set 三个通用参数
    args = base_parser("影瞳 · 视频抽帧").parse_args()
    cfg = load_cfg(args)  # --set 覆盖 → --device 覆盖 train.device
    records = extract_all(cfg)
    print(f"\n完成：共 {len(records)} 张帧。下一步运行 scripts/02_build_manifest.py")


if __name__ == "__main__":
    main()
