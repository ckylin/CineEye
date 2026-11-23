"""扩展步骤 3：演员参考图库管理。

用途：用"每人几张参考照"的方式做人脸级识别，不需要训练人脸模型。
参考照按 outputs/actors/<演员名>/<图片> 放好后，--build 会用 OpenCV Haar 检测
最大人脸 → 裁切 → 用训练好的主干编码成向量，存成 embeddings.npy + names.json；
之后 --image 就能把一张剧照里的人脸比对到具体演员。

典型调用：
    # 按 <演员名>/<图片> 放好参考照后，构建向量库
    mkdir -p outputs/actors/演员甲
    cp 演员照片.jpg outputs/actors/演员甲/
    python scripts/13_actor_reference.py --build

    # 查看参考库状态
    python scripts/13_actor_reference.py --stats

    # 识别一张图里的人脸
    python scripts/13_actor_reference.py --image 照片.jpg

输入/输出：
    读取：outputs/actors/<演员名>/*.jpg（cfg.actors.reference_dir）、
          outputs/checkpoints/best.pt（当编码器主干）、outputs/cineeye.db（用于补"参演电影"）
    输出：outputs/actors/embeddings.npy（N×D 归一化向量）、
          outputs/actors/names.json（逐行对应的演员名）

运维要点：
    1. 目录名就是演员名，改名等于换身份，必须重新 --build。
    2. --build 是全量重建（每次清空重算），参考照很少时秒级完成，无需增量。
    3. 人脸检测走 Haar 级联，侧脸/遮挡/小脸容易漏检；漏检时按
       actors.fallback_whole_image 决定是否用中心裁剪兜底。
    4. 这是"不训练"的轻量路线：放图即用。若要更高准确率，请改用
       scripts/15_import_actors.py 打标签 + scripts/16_train_actors.py 训练，
       详见 docs/ACTOR_GUIDE.md；训练会覆盖本脚本产出的检索库。
    5. --device 写入 cfg.train.device，决定编码走 CPU 还是 GPU（GPU 上建库快很多）。
"""

import argparse
import json
import sys
from pathlib import Path

# 直接执行脚本时 sys.path[0] 是 scripts/，需补项目根
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cineeye.actors import ActorRecognizer, FaceDetector  # noqa: E402
from cineeye.cli import base_parser, load_cfg  # noqa: E402
from cineeye.gallery import ImageEncoder  # noqa: E402
from cineeye.utils import ensure_dir, get_logger  # noqa: E402

log = get_logger("cineeye.actors.build")


def main() -> None:
    """命令行入口：四种互斥动作 —— --stats 巡检 / --detect-only 只看人脸框 /
    --build 重建向量库 / --image 做识别；都没给就打印帮助。"""
    p = base_parser("影瞳 · 演员参考图库")
    p.add_argument("--build", action="store_true", help="从参考图目录构建向量库")
    p.add_argument("--stats", action="store_true", help="打印参考库状态")
    p.add_argument("--image", default=None, help="识别图片里的人脸")
    p.add_argument("--detect-only", action="store_true", help="只做人脸检测，不做识别")
    args = p.parse_args()

    cfg = load_cfg(args)
    # 参考图目录从配置读，允许相对路径（相对项目根）；不存在就自动建空目录
    ref_dir = Path(cfg.get("actors", {}).get("reference_dir", "outputs/actors"))
    if not ref_dir.is_absolute():
        ref_dir = Path(__file__).resolve().parents[1] / ref_dir
    ensure_dir(ref_dir)

    if args.stats:
        # 只扫描目录 + 检查索引文件是否存在，不加载模型、不编码，适合巡检
        rec = ActorRecognizer(cfg=cfg)
        pairs = rec.iter_reference_images(ref_dir)
        actor_names = sorted({n for n, _ in pairs})
        print(json.dumps({
            "reference_dir": str(ref_dir),
            "build_index_exists": rec.emb_path.exists(),
            "images": len(pairs),
            "actors": actor_names,
        }, ensure_ascii=False, indent=2))
        return

    if args.detect_only:
        # 调参用：只输出人脸框坐标，可快速判断"是检测不到"还是"认错了"
        if not args.image:
            p.error("--detect-only 需要 --image")
        detector = FaceDetector(cfg)
        enc = ImageEncoder(cfg=cfg)
        img = enc.load_image(args.image)
        boxes = detector.detect(img)
        print(json.dumps({"image": args.image, "faces": [
            {"x": b.x, "y": b.y, "w": b.w, "h": b.h} for b in boxes]}, ensure_ascii=False, indent=2))
        return

    if args.build:
        # 全量重建：遍历 <演员名>/ 子目录，每张参考照裁最大人脸后编码
        rec = ActorRecognizer(cfg=cfg)
        n = rec.build_reference()
        print(f"演员参考库构建完成：{n} 张参考图")
        return

    if args.image:
        # 识别：比对参考库得到演员，并用元数据库补上"参演电影"等上下文
        rec = ActorRecognizer(cfg=cfg)
        enc = ImageEncoder(cfg=cfg)
        img = enc.load_image(args.image)
        result = rec.recognize_with_movies(img)
        print(json.dumps(result or {"message": "未识别到演员（参考库为空或无人脸）"},
                         ensure_ascii=False, indent=2))
        return

    # 没有任何动作时给帮助信息，而不是报错（便于人肉探索）
    p.print_help()


if __name__ == "__main__":
    main()
