"""第 2 步：扫描帧目录，按「视频」分组划分 train/val/test，生成标注清单。

用途：把 data/frames 下散落的 jpg 汇总成一张 data/manifest.csv（每行一帧：
frame_path / label / label_idx / video_id / split），并写出 label_map.json。
划分按 video_id 分组而不是按帧随机——同一段视频的相邻帧高度相似，随机划分会让
验证集"见过"训练集画面，准确率虚高（数据泄漏）。

典型调用：
    python scripts/02_build_manifest.py
    # 需要换划分比例/设备时（--device 对本步无实际影响，但脚本保持参数一致）
    python scripts/02_build_manifest.py --set data.split.train=0.8

输入/输出：
    读取：data/frames/<标签>/<video_id>/*.jpg
    输出：data/manifest.csv、outputs/label_map.json（类别名 → 编号 的映射）

运维要点：
    1. 类别（label）集合在每次运行时重新扫描生成，label_map.json 会被重写；
       训练好的权重与 label_map 是绑定的，换数据后必须重跑 03 及之后步骤。
    2. 划分比例来自 cfg.data.split（默认 0.70/0.15/0.15），分组键是 data.split.group_by。
    3. 帧路径写成相对项目根的路径，整个项目目录可以整体搬移而不失效。
    4. 公共参数：--set 覆盖配置，--device 写入 cfg.train.device（本步不训练，仅保持一致）。
"""

import sys
from pathlib import Path

# 直接执行脚本时需手工把项目根加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cineeye.cli import base_parser, load_cfg  # noqa: E402
from cineeye.data.manifest import build_manifest


def main() -> None:
    """解析参数 → 读配置 → 扫描并分组划分 → 写 manifest.csv / label_map.json。"""
    args = base_parser("影瞳 · 构建标注清单（分组划分，防数据泄漏）").parse_args()
    cfg = load_cfg(args)
    # build_manifest 返回 (DataFrame, {类别名: 编号})，内部还会做一次泄漏自检
    df, label_map = build_manifest(cfg)
    print("\n类别映射：", label_map)
    print("下一步运行 scripts/03_train.py")


if __name__ == "__main__":
    main()
