"""一键跑通：合成数据 → 抽帧 → 清单 → 训练 → 评估 → 推理演示。

用途：把 00~05 六个脚本按顺序串起来，用于"新机器/新环境先确认整条链路可用"。
每一步都是独立进程（subprocess），因此任一步失败会当场停下并保留该步的报错，
不会出现"用半份数据继续往下跑"的脏状态。

典型调用：
    python scripts/run_all.py
    python scripts/run_all.py --epochs 20 --set model.backbone=mobilenet_v3_small
    python scripts/run_all.py --skip-demo      # 已有自己的数据，只跑后半段

输入/输出：
    读取：无（--skip-demo 时读取已有的 data/videos 或 data/frames）
    输出：data/videos/（合成数据）、data/frames/、data/manifest.csv、
          outputs/checkpoints/best.pt、outputs/reports/*（与单独跑各步完全一致）

运维要点：
    1. --skip-demo 只跳过"造数据"，后续步骤仍会覆盖 manifest 与 checkpoint；
       用真实数据做正式训练时，建议逐步手工执行 01~04，便于逐段检查。
    2. 传给各步的参数只有 --set 系列（外加 04 的 --split），所以 --device 这类参数
       这里没有暴露；需要时请改用 --set 或直接单独调用某个脚本。
    3. 全程把 TMPDIR 指向项目 .tmp：本机 /tmp 只有 10MB，pip/ffmpeg/DataLoader
       的临时写入会失败。
    4. 最后一步的推理演示会从 manifest 里取第一张帧、从 data/videos 里取最后一个视频，
       属于"证明链路通"的最小验证，不代表模型效果。
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# 直接执行脚本时 sys.path[0] 是 scripts/，需补项目根（下面 import pandas 之前就要生效）
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run(script: str, extra: list[str]) -> None:
    """跑一个子脚本：统一环境变量、打印分隔标题、失败即抛 CalledProcessError。

    用 sys.executable 而不是 "python"：保证子进程用的是当前虚拟环境的解释器，
    避免在 ./.venv 里跑却调用到系统 python 导致依赖缺失。
    """
    env = os.environ.copy()
    (ROOT / ".tmp").mkdir(exist_ok=True)
    # setdefault：若运维已显式设置 TMPDIR（例如指向更大的数据盘），就尊重其选择
    env.setdefault("TMPDIR", str(ROOT / ".tmp"))
    print("\n" + "=" * 72)
    print(f">> {script}  {' '.join(extra)}")
    print("=" * 72, flush=True)
    # cwd 固定为项目根：所有脚本都按"配置里的相对路径基于项目根"约定工作
    subprocess.run([sys.executable, str(ROOT / "scripts" / script), *extra],
                   cwd=str(ROOT), env=env, check=True)


def main() -> None:
    """命令行入口：拼装每一步的参数，顺序执行流水线，最后做一次推理演示。"""
    ap = argparse.ArgumentParser(description="影瞳 · 端到端流水线")
    ap.add_argument("--skip-demo", action="store_true", help="跳过合成数据生成（使用自己的数据）")
    ap.add_argument("--movies", type=int, default=4)
    ap.add_argument("--clips", type=int, default=5)
    ap.add_argument("--frames", type=int, default=80)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--set", action="append", default=[], dest="overrides", metavar="a.b.c=VALUE")
    args = ap.parse_args()

    # 统一成 "--set=a.b.c=value" 的形式，原样透传给每一步。
    # 优先级：显式 --set train.epochs=N > --epochs N。若用户在 --set 里已经指定了
    # train.epochs，就不再追加 --epochs，否则后追加的值会覆盖用户的手动设置。
    sets = [f"--set={s}" for s in args.overrides]
    if not any(s.strip().startswith("train.epochs=") for s in args.overrides):
        sets.append(f"--set=train.epochs={args.epochs}")

    if not args.skip_demo:
        # 造数据不带 --set：它不读 cineeye 配置，只认自己的参数
        run("00_make_demo_data.py", [
            "--movies", str(args.movies), "--clips", str(args.clips), "--frames", str(args.frames),
        ])
    run("01_extract_frames.py", sets)
    run("02_build_manifest.py", sets)
    run("03_train.py", sets)
    # 评估固定用 test 划分，避免"用验证集调参又用验证集汇报"的自我欺骗
    run("04_evaluate.py", [*sets, "--split", "test"])

    # 推理演示：挑一张真实存在的帧和一段视频
    import pandas as pd

    manifest = ROOT / "data" / "manifest.csv"
    # 取 manifest 第一行的帧路径（相对项目根），拼成绝对路径交给 05_predict
    frame = str(ROOT / pd.read_csv(manifest)["frame_path"].iloc[0])
    # 演示数据可能落成 mp4 或 avi（取决于 OpenCV 可用的编码器），两种都找
    videos = sorted((ROOT / "data" / "videos").rglob("*.mp4")) + sorted((ROOT / "data" / "videos").rglob("*.avi"))
    pred_args = [*sets, "--image", frame]
    if videos:
        # 取最后一个：固定选法，保证多次运行的演示对象一致、便于对比
        pred_args += ["--video", str(videos[-1])]
    run("05_predict.py", pred_args)
    print("\n全流程跑通。产物见 outputs/ 目录。")


if __name__ == "__main__":
    main()
