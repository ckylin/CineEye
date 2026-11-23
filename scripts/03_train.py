"""第 3 步：训练（默认 feature 模式，CPU 友好）。

用途：读 manifest.csv 训练电影分类器，产出 outputs/checkpoints/best.pt（含
model_state / label_map / backbone / mode / 训练时的 config 快照）与训练历史 CSV。
两种模式：
    feature  —— 冻结 ImageNet 主干，先把所有帧的 feature 缓存到 outputs/features，
                再只训练轻量分类头；CPU 上几分钟可跑完，是默认路线。
    finetune —— 端到端微调整网；GPU 上效果好，CPU 上极慢。

典型调用：
    python scripts/03_train.py
    python scripts/03_train.py --mode finetune --set train.epochs=5   # 端到端微调，CPU 很慢
    python scripts/03_train.py --set model.backbone=mobilenet_v3_small

输入/输出：
    读取：data/manifest.csv、outputs/label_map.json、预训练权重缓存（~/.cache/torch）
    输出：outputs/checkpoints/best.pt、outputs/checkpoints/classical_baseline.joblib、
          outputs/reports/history.csv、outputs/features/*（feature 模式的特征缓存）

运维要点：
    1. --mode 只在命令行给定时覆盖配置；不传则用 cfg.model.mode（默认 feature）。
    2. 特征缓存与 backbone/图像尺寸强绑定：换了 --set model.backbone 或
       data.image_size 后，旧缓存会失效（代码按指纹校验），需要重新提特征。
    3. 公共参数优先级：--set a.b.c=value 逐项覆盖配置 → --device 最后写入
       cfg.train.device，所以「配置文件写 cpu、命令行 --device cuda」会以 cuda 为准。
    4. 本机无 GPU 但支持 CUDA：用 --device cuda 前先跑 00_check_env.py 确认
       cuda_kernel_ok=True（RTX 50 系需 cu128 wheel）。
    5. /tmp 与 /dev/shm 只有 10MB：train.num_workers 保持 0，否则 DataLoader 会 ENOSPC。
"""

import sys
from pathlib import Path

# 直接执行脚本时 sys.path[0] 是 scripts/，需补项目根
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cineeye.cli import base_parser, load_cfg  # noqa: E402
from cineeye.train import run_training


def main() -> None:
    """解析参数 → 读配置 → 训练 → 打印训练摘要与 sklearn 基线对比。"""
    p = base_parser("影瞳 · 监督训练")
    # 默认 None 而不是 "feature"：让「没传参数」时沿用配置文件里的 model.mode
    p.add_argument("--mode", choices=["feature", "finetune"], default=None,
                   help="feature=冻结主干提特征+训练分类头；finetune=端到端微调")
    args = p.parse_args()
    cfg = load_cfg(args)
    # mode 为 None 时 run_training 内部回退到 cfg.model.mode
    report = run_training(cfg, mode=args.mode)
    print("\n训练摘要：", report["deep"])
    if report.get("sklearn_baseline"):
        # 传统特征 + sklearn 基线，用来判断深度模型是否真的带来提升
        print("sklearn 基线：", report["sklearn_baseline"])
    print("下一步运行 scripts/04_evaluate.py")


if __name__ == "__main__":
    main()
