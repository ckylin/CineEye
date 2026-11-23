"""第 4 步：在 test 集上评估（帧级 + 视频级），产出混淆矩阵与预测明细。

用途：加载 best.pt，在指定 split 上分别给出
    帧级指标 —— 每张图单独判定，衡量分类头本身的能力；
    视频级指标 —— 同一 video_id 的所有帧概率取均值再判定，贴近真实使用方式，
                  也是汇报给业务方时应引用的口径。
同时把混淆矩阵、逐帧预测明细写进 outputs/reports。

典型调用：
    python scripts/04_evaluate.py
    python scripts/04_evaluate.py --split val
    python scripts/04_evaluate.py --checkpoint outputs/checkpoints/best.pt --device cpu

输入/输出：
    读取：data/manifest.csv（按 split 过滤）、权重（--checkpoint 或 cfg.inference.checkpoint）
    输出：outputs/reports/confusion_matrix_<split>.{csv,png}、
          classification_report_<split>.txt、predictions_<split>.csv、
          eval_<split>.json（验收与回归对比的依据）

运维要点：
    1. 默认在 test 上评估；train 只用于自查是否过拟合，不要拿 train 的分数对外汇报。
    2. 视频级准确率一般高于帧级——如果两者差距过大，说明单帧噪声重，
       真机推理应优先用视频聚合（05_predict --video 或 /predict/video）。
    3. 换了 checkpoint 或 label_map 后必须重新评估，二者必须同时更新。
    4. --device 由 base_parser 提供并写入 cfg.train.device，优先级高于配置文件与 --set。
"""

import sys
from pathlib import Path

# 直接执行脚本时补项目根到 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cineeye.cli import base_parser, load_cfg  # noqa: E402
from cineeye.evaluate import evaluate


def main() -> None:
    """解析参数 → 读配置 → 评估 → 打印帧级/视频级两组指标。"""
    p = base_parser("影瞳 · 模型评估")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--checkpoint", default=None)
    args = p.parse_args()
    # evaluate 内部自己建 Predictor；checkpoint 为空时用配置里的 inference.checkpoint
    result = evaluate(load_cfg(args), split=args.split, checkpoint=args.checkpoint)
    print("\n帧级：", result["frame_level"])
    print("视频级：", result["video_level"])


if __name__ == "__main__":
    main()
