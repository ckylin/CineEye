"""把外部目录里的长电影切成片段并抽帧，组织成影瞳项目的训练数据布局。

用途：真实片源通常是"一部长片一个文件"，直接抽帧只会得到单一 video_id，
无法在 train/val/test 中都留出独立视频（按视频分组划分时会整部片只落进一个集合）。
本脚本按时间把每部长片切成若干段，每段成为独立 video_id，抽帧后落成项目标准布局。

典型调用：
    python scripts/10_prepare_real_movies.py \\
        --src /home/kylin/movie_train \\
        --segments 8 --frames-per-seg 40
    python scripts/10_prepare_real_movies.py --src /data/raw --no-backup --long-edge 320

输入/输出：
    读取：--src 目录下的视频文件（.mp4/.mkv/.avi/.mov/.webm/.ts/.m4v），需系统有 ffmpeg/ffprobe
    输出：data/frames/movie_01/seg_00/000001.jpg ...
          outputs/reports/movie_id_map.json（中性编号 → 原始文件名，唯一含真实片名的文件）

设计要点：
1. 中性标签 movie_01..06：类别名不进清单/报告，演示截图、评估报告、接口返回都只有
   movie_01 这样的编号；真实片名只写在 movie_id_map.json 里（.gitignore 覆盖 outputs/），
   对外分享项目时不泄露版权片单。
2. 8 段 × 40 帧：（默认 --segments 8）每部片切 8 段，（--frames-per-seg 40）每段抽 40 帧，
   合计约 320 帧/部。8 段足以让 70/15/15 的划分在 train/val/test 里各有独立 video_id；
   每段 40 帧则控制单段重复度，避免同一镜头贡献过多相似帧。
3. fps 由段长反推（frames_per_seg / seg_len），因此无论片长多少，每段都恰好约 40 帧。
4. 抽帧前把旧的演示数据与训练产物移到 .tmp/backup_demo（--no-backup 可跳过），
   绝不删除——万一切错了还能回滚。

运维要点：
- 本机 /tmp 只有 10MB，ffmpeg 会写临时文件，所以脚本内部强制把 TMPDIR 指向项目 .tmp。
- 跑完必须重跑 02_build_manifest.py（manifest / label_map / 特征缓存都会和旧数据对不上）。
- --src 里少于 2 个视频会直接退出：单类别无法训练，也不构成划分。
- 备份是"整体 move"，如果 data/frames 里有你手工补充的图片，会被一起移走。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# 直接执行脚本时 sys.path[0] 是 scripts/，需补项目根
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 与 cfg.data.video_extensions 保持一致的兜底列表（本脚本不读配置，避免依赖 manifest）
VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".ts", ".m4v"}


def probe_duration(path: Path) -> float:
    """用 ffprobe 读取视频总时长（秒）。

    用 ffprobe 而不是 OpenCV：CV_CAP_PROP_FRAME_COUNT/fps 对 mkv、可变帧率、
    损坏索引的文件经常算出离谱的值，导致分段位置错乱。
    """
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ], text=True)
    return float(out.strip())


def backup_existing() -> None:
    """把旧的演示帧与训练产物移到 .tmp 备份，避免混合进真实数据。

    注意是 move 而不是 copy/delete：真实数据重跑时磁盘占用不翻倍，
    同时保留回滚能力（.tmp/backup_demo 下）。
    """
    stamp = "backup_demo"
    backup = ROOT / ".tmp" / stamp
    backup.mkdir(parents=True, exist_ok=True)

    frames = ROOT / "data" / "frames"
    if frames.exists():
        dest = backup / "frames"
        if dest.exists():
            # 二次重跑时旧备份已存在，直接清掉（备份目录内部数据可丢弃）
            shutil.rmtree(dest)
        shutil.move(str(frames), str(dest))
        print(f"[备份] 旧帧目录 → {dest.relative_to(ROOT)}")

    # 训练产物必须一起备份：否则残留的 label_map/best.pt 会与新的 movie_XX 类别错位
    for rel in ["outputs/checkpoints", "outputs/features", "outputs/reports",
                "outputs/label_map.json"]:
        src = ROOT / rel
        if src.exists():
            dest = backup / Path(rel).name
            if dest.exists():
                # 文件与目录的清理方式不同：目录 rmtree，文件 unlink
                shutil.rmtree(dest) if dest.is_dir() else dest.unlink()
            shutil.move(str(src), str(dest))
            print(f"[备份] {rel} → {dest.relative_to(ROOT)}")

    manifest = ROOT / "data" / "manifest.csv"
    if manifest.exists():
        dest = backup / "manifest.csv"
        if dest.exists():
            dest.unlink()
        shutil.move(str(manifest), str(dest))
        print(f"[备份] data/manifest.csv → {dest.relative_to(ROOT)}")


def main() -> None:
    """命令行入口：扫描片源 → （可选）备份 → 逐部切段抽帧 → 写 movie_id_map.json。"""
    ap = argparse.ArgumentParser(description="影瞳 · 真实电影切段抽帧")
    ap.add_argument("--src", default="/home/kylin/movie_train", help="原始视频目录")
    ap.add_argument("--segments", type=int, default=8, help="每部电影切成几段")
    ap.add_argument("--frames-per-seg", type=int, default=40, help="每段抽取帧数")
    ap.add_argument("--long-edge", type=int, default=256, help="落盘长边缩放")
    ap.add_argument("--no-backup", action="store_true", help="不备份旧数据（覆盖式重跑时用）")
    args = ap.parse_args()

    src = Path(args.src)
    # 只取 --src 顶层的视频文件，不递归：子目录留给"按类别组织"的其他数据布局
    videos = sorted(p for p in src.iterdir()
                    if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
    if len(videos) < 2:
        # 单部电影无法划分 train/val/test，直接失败比生成坏数据好
        raise SystemExit(f"在 {src} 只找到 {len(videos)} 个视频，至少需要 2 个")

    if not args.no_backup:
        backup_existing()

    frames_root = ROOT / "data" / "frames"
    frames_root.mkdir(parents=True, exist_ok=True)

    # 临时目录指向项目盘：本机 /tmp 仅 10MB，ffmpeg/系统临时写入会爆
    env = os.environ.copy()
    env["TMPDIR"] = str(ROOT / ".tmp")

    id_map: dict[str, dict] = {}
    fps_base = args.frames_per_seg  # 由每段时长反推 fps

    for idx, video in enumerate(videos, 1):
        # 中性编号：1 开始、两位补零，排序稳定；与真实片名无关，不泄露片单
        label = f"movie_{idx:02d}"
        duration = probe_duration(video)
        seg_len = duration / args.segments  # 每段秒数
        # 反推采样率：段长 seg_len 秒内要恰好出 frames_per_seg 帧
        fps = fps_base / seg_len if seg_len > 0 else 1.0
        total_frames = 0
        print(f"\n>> {label}  {video.name}")
        print(f"   时长 {duration:.0f}s → {args.segments} 段 × 约 {args.frames_per_seg} 帧 "
              f"(段长 {seg_len:.0f}s, fps={fps:.5f})")

        for s in range(args.segments):
            # 每段一个目录 = 一个独立 video_id（02 步按目录名反推 video_id 分组划分）
            seg_dir = frames_root / label / f"seg_{s:02d}"
            seg_dir.mkdir(parents=True, exist_ok=True)
            start = s * seg_len
            # -ss 放在 -i 之前是快速定位（关键帧跳转），比解码到位置快得多；
            # scale=-2:long_edge 保持宽高比，-2 保证宽是偶数（H.264/JPEG 友好）
            cmd = [
                "ffmpeg", "-nostdin", "-v", "error",
                "-ss", f"{start:.3f}", "-t", f"{seg_len:.3f}",
                "-i", str(video),
                "-vf", f"fps={fps:.6f},scale=-2:{args.long_edge}",
                "-frames:v", str(args.frames_per_seg),
                "-q:v", "4", "-y", str(seg_dir / "%06d.jpg"),
            ]
            subprocess.run(cmd, check=True, env=env)
            # 以实际落盘文件数计数：ffmpeg 可能因片尾不足而少出几帧
            n = len(list(seg_dir.glob("*.jpg")))
            total_frames += n
            print(f"   seg_{s:02d}: {n} 帧", flush=True)

        # 只在映射表里记录原始文件名与片长，供后期追溯；训练流程完全不读这个表
        id_map[label] = {
            "source_file": video.name,
            "duration_sec": round(duration, 1),
            "segments": args.segments,
            "frames": total_frames,
        }

    map_path = ROOT / "outputs" / "reports" / "movie_id_map.json"
    map_path.parent.mkdir(parents=True, exist_ok=True)
    map_path.write_text(json.dumps(id_map, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n完成：{len(videos)} 部电影 / {sum(v['frames'] for v in id_map.values())} 帧")
    print(f"类别映射（仅此文件含原始片名）→ {map_path.relative_to(ROOT)}")
    print("下一步：python scripts/02_build_manifest.py")


if __name__ == "__main__":
    main()
