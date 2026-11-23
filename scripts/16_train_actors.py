"""扩展步骤 5：训练演员识别模型（分类头 + 参考检索库）。

用途：用 15_import_actors.py 产出的 manifest 训练演员分类头。
流程：读 manifest → 人脸裁剪 → 冻结主干抽特征（带缓存）→ 训练线性分类头
→ 在 val/test 上评估 → 估计 unknown 拒识阈值 → 落盘 checkpoint 与检索库。

典型调用：
    # 全量训练（默认读 data/actors/manifest.csv）
    python scripts/16_train_actors.py

    # 调参：200 轮、学习率 0.02、只保留最后报告
    python scripts/16_train_actors.py --epochs 300 --lr 0.02

    # 照片刚改过，强制重算特征缓存
    python scripts/16_train_actors.py --rebuild-cache

    # 只评估已有模型（不训练）
    python scripts/16_train_actors.py --eval-only

输入/输出：
    读取：data/actors/manifest.csv（或 --manifest 指定）、
          照片文件、outputs/cineeye.db（评估后不做额外写入）
    输出：outputs/checkpoints/actor_head.pt     模型（含 label_map/阈值/指标）
          outputs/reports/actor_train_report.json 训练报告
          outputs/features/actor_features.npz   特征缓存
          outputs/actors/embeddings.npy + names.json + encoder.json  检索库

运维要点：
    1. 增删演员或照片后必须重跑本脚本，否则分类头不认识新演员；
    2. 特征缓存按「manifest 内容 + 编码器指纹 + 裁剪参数」做签名，
       任何一项变化都会自动重算；照片只改文件名不改内容时缓存仍有效；
    3. 阈值优先取 actors.unknown_threshold（配置写死），否则训练时自动估计：
       有 unknown_dir 就真实校准，没有就用验证集置信度分位数做经验值；
    4. 换 actors.encoder / backbone / image_size 后，旧模型会被识别器判定
       指纹不匹配并忽略，必须重训。
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cineeye.actors.train import run_training  # noqa: E402
from cineeye.cli import base_parser, load_cfg  # noqa: E402


def main() -> None:
    """命令行入口：解析训练参数 → 调用 run_training → 打印报告摘要。"""
    p = base_parser("影瞳 · 训练演员识别模型")
    p.add_argument("--manifest", default=None, help="演员清单路径（默认 actors.manifest）")
    p.add_argument("--epochs", type=int, default=None, help="分类头训练轮数")
    p.add_argument("--lr", type=float, default=None, help="学习率")
    p.add_argument("--batch-size", type=int, default=None, help="特征训练批大小")
    p.add_argument("--crop", choices=["detect", "center", "whole"], default=None,
                   help="人脸裁剪方式（训练与推理必须一致，会写进 checkpoint）")
    p.add_argument("--rebuild-cache", action="store_true", help="强制重算特征缓存")
    p.add_argument("--eval-only", action="store_true", help="只评估已有模型，不训练")
    args = p.parse_args()

    cfg = load_cfg(args)  # --device 已写入 cfg.train.device，编码阶段生效
    report = run_training(
        cfg,
        manifest=args.manifest,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        crop_mode=args.crop,
        rebuild_cache=args.rebuild_cache,
        device=args.device,
        eval_only=args.eval_only,
    )
    # 控制台只打关键信息，完整报告看 outputs/reports/actor_train_report.json
    brief = {
        "checkpoint": report["checkpoint"],
        "actors": report["actors"],
        "split_counts": report["split_counts"],
        "face_detected_rate": report["face_detected_rate"],
        "threshold": report["threshold"],
        "metrics": report["metrics"],
        "cache_hit": report["cache_hit"],
    }
    print(json.dumps(brief, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
