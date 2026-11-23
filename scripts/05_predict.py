"""第 5 步：识别一张图片或一段视频属于哪部电影。

用途：加载 best.pt 做封闭集分类，是命令行侧的"试一下模型效果"入口。
图片走单帧预测；视频走「均匀抽帧 → 逐帧预测 → 聚合」，后者稳定得多，
因为单帧可能恰好是黑屏、片头字幕或通用空镜。

典型调用：
    python scripts/05_predict.py --image data/frames/星际穿越/clip_01/000010.jpg
    python scripts/05_predict.py --video data/videos/星际穿越/clip_03.mp4
    python scripts/05_predict.py --video xxx.mp4 --set inference.num_frames=32 --set inference.aggregation=majority

输入/输出：
    读取：--image / --video 指定的文件、权重（--checkpoint 或 cfg.inference.checkpoint）
    输出：仅标准输出（人读摘要，或 --json 机器可读）；不落盘

运维要点：
    1. 三种聚合方式：
       mean_prob（默认）—— 概率求均值，最稳；
       log_prob        —— 对数概率求均值，对高置信帧更敏感；
       majority        —— 少数服从多数投票，抗个别异常帧。
    2. 置信度低于 cfg.inference.unknown_threshold 时输出「未知影片」；
       默认阈值为 0 表示关闭拒识，生产上建议设 0.5~0.7。
    3. --image 可传多张（nargs="*"），每张独立出结果，不做跨图聚合。
    4. --device 经 base_parser 写入 cfg.train.device，优先级高于配置与 --set。
"""

import json
import sys
from pathlib import Path

# 直接执行脚本时补项目根到 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cineeye.cli import base_parser, load_cfg  # noqa: E402
from cineeye.predict import Predictor  # noqa: E402


def main() -> None:
    """解析参数 → 建 Predictor → 逐图/逐视频预测 → 打印或输出 JSON。"""
    p = base_parser("影瞳 · 图片/视频识别电影")
    p.add_argument("--image", nargs="*", default=[], help="一张或多张图片路径")
    p.add_argument("--video", default=None, help="视频路径（多帧聚合，比单帧稳定得多）")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--num-frames", type=int, default=None)
    p.add_argument("--aggregation", choices=["mean_prob", "log_prob", "majority"], default=None)
    p.add_argument("--json", action="store_true", help="以 JSON 输出，便于接入其他程序")
    args = p.parse_args()

    if not args.image and not args.video:
        # 用 p.error（argparse 的错误路径）而非 assert：退出码 2 且打印 usage
        p.error("至少提供 --image 或 --video")

    cfg = load_cfg(args)
    # num_frames/aggregation 不传时由 Predictor 用配置默认值兜底
    predictor = Predictor(args.checkpoint, cfg=cfg)
    results = predictor.predict_images(args.image)
    if args.video:
        # 图片与视频结果放在同一列表里输出，保持"多来源一次调用"的体验
        results.append(predictor.predict_video(
            args.video, num_frames=args.num_frames, aggregation=args.aggregation,
        ))

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return
    for r in results:
        # unknown=True 时 label 不可信，显式标成"未知影片"提醒使用者
        tag = "未知影片（低置信度）" if r.get("unknown") else r["label"]
        print(f"\n来源: {r.get('source')}")
        print(f"识别结果: {tag}   置信度: {r['confidence']:.4f}")
        if r.get("num_frames"):
            # frame_agreement = 得票最多的类别占采样帧的比例，用来判断"整段是否一致"
            print(f"聚合方式: {r['aggregation']} | 采样帧数: {r['num_frames']} | 帧一致性: {r['frame_agreement']:.2%}")
        print("Top-K: " + ", ".join(f"{x['label']} {x['confidence']:.3f}" for x in r["top_k"]))


if __name__ == "__main__":
    main()
