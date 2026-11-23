"""扩展步骤 4：统一识别（图片/视频 → 完整电影档案）。

用途：命令行版的"一站式识别"。它调用 Identifier，把
    图库向量检索（开放集：海报/剧照/演员照/路演照都行）
    + 封闭集分类器打分（仅 frame 类型，作为强证据）
    + 元数据库补全（主片名/原名/别名/年份/导演/演员/海报）
    + 可选演员人脸识别
合成一个结果，是 05_predict.py（纯封闭集分类）的升级版。

典型调用：
    # 识别一张海报/剧照/演员照
    python scripts/14_identify.py --image 照片.jpg

    # 指定图片类型（只在该类型的图库中检索）
    python scripts/14_identify.py --image 海报.jpg --kind poster

    # 视频（多帧聚合）
    python scripts/14_identify.py --video 片段.mp4 --num-frames 16

    # 只做演员识别（合影/生活照，返回每张脸是谁）
    python scripts/14_identify.py --actor-image 合影.jpg

    # 按片名/别名搜索（含多语言译名）
    python scripts/14_identify.py --search "Example Movie"

    # 查看电影档案
    python scripts/14_identify.py --movie movie_01

    # 系统状态
    python scripts/14_identify.py --status

输入/输出：
    读取：outputs/gallery/*（图库索引）、outputs/cineeye.db（元数据库）、
          outputs/actors/*（演员参考库）、outputs/checkpoints/best.pt（分类器）
    输出：仅标准输出 JSON（default=str 兼容 numpy/时间类型）；不落盘

运维要点：
    1. 依赖前三步的产物：11 导入元数据 → 12 建图库索引 → 13 建演员参考库；
       缺任何一环不会崩，但对应能力会降级（只有一个分类器/只有图库检索）。
    2. --kind 会限制检索范围：类型填错（例如拿 poster 去检索 frame 库）会得到空结果，
       报 "未识别" 时先确认 kind 与图库里的实际类型一致。
    3. 演员识别默认开启（cfg.actors.enabled），--no-actors 可关闭以提速。
    4. --device 由 base_parser 提供并写入 cfg.train.device，Identifier 内部会把它
       同时传给图库编码器与分类器，避免设备割裂。
    5. 与 05_predict.py 的区别：05 只能在训练过的类别里选，14 能泛化到图库里
       任何新增电影（加图即可，无需重训）。
"""

import argparse
import json
import sys
from pathlib import Path

# 直接执行脚本时 sys.path[0] 是 scripts/，需补项目根
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cineeye.cli import base_parser, load_cfg  # noqa: E402
from cineeye.identify import Identifier  # noqa: E402


def main() -> None:
    """命令行入口：按参数选择 status/search/movie/video/image 五个分支之一。

    分支顺序即优先级：同时给多个参数时，靠前的先生效（避免歧义）。
    """
    p = base_parser("影瞳 · 统一电影档案识别")
    p.add_argument("--image", nargs="*", default=[], help="一张或多张图片路径")
    p.add_argument("--video", default=None, help="视频路径")
    p.add_argument("--search", default=None, help="按片名/别名搜索")
    p.add_argument("--movie", default=None, help="查看电影完整档案")
    p.add_argument("--actor-image", nargs="+", default=None,
                   help="只做演员识别（合照/生活照/剧照均可，不需要电影模型）")
    p.add_argument("--status", action="store_true", help="系统状态")
    p.add_argument("--kind", default=None, choices=["frame", "poster", "actor", "still", "roadshow", "other"],
                   help="图片类型过滤")
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--no-actors", action="store_true", help="跳过演员识别")
    p.add_argument("--with-actors", action="store_true", help="强制启用演员识别")
    args = p.parse_args()

    cfg = load_cfg(args)  # --device 已由 base_parser 提供并写入 cfg.train.device

    if args.actor_image:
        # 演员识别单独走一条轻量路径：不需要 best.pt，也不需要图库/元数据以外的产物。
        # 放在 Identifier 之前，是为了让"只装了演员数据"的机器也能用这个入口。
        import cv2
        from cineeye.actors import ActorRecognizer

        rec = ActorRecognizer(cfg=cfg)
        results = []
        for path in args.actor_image:
            pth = Path(path)
            if not pth.is_absolute():
                pth = Path(__file__).resolve().parents[1] / pth
            img = cv2.imread(str(pth))
            if img is None:
                results.append({"source": path, "error": "图片无法读取或不存在"})
                continue
            detail = rec.recognize_all(img)
            for face in detail.get("faces", []):
                person = rec.store.find_person(face["actor"]) if face.get("actor") else None
                face["person_id"] = person["id"] if person else None
                face["movies"] = person["movies"] if person else []
            results.append({"source": path, **detail})
        print(json.dumps({"recognizer": rec.status(), "results": results},
                         ensure_ascii=False, indent=2, default=str))
        return

    # Identifier 构造时会加载元数据库/图库/演员库/分类器，比较重，所以只建一次复用
    ident = Identifier(cfg=cfg)

    if args.status:
        # 巡检：元数据库统计 + 图库统计 + 编码器与设备信息
        result = ident.status()
    elif args.search:
        # 纯文本检索，不碰模型，最快
        result = ident.search_movies(args.search, top_k=args.top_k or 10)
    elif args.movie:
        result = ident.get_movie(args.movie) or {"error": "电影不存在"}
    elif args.video:
        # 视频：均匀抽帧 → 多帧检索聚合；kind 默认 frame
        # with_actors 三态与图片分支保持一致：--with-actors 强制开 > --no-actors 强制关 > 配置默认
        with_actors = True if args.with_actors else (False if args.no_actors else None)
        result = ident.identify_video(
            args.video, num_frames=args.num_frames, kind=args.kind or "frame",
            top_k=args.top_k, with_actors=with_actors)
    elif args.image:
        # 图片：先解码成 BGR ndarray 再交给 Identifier（保持与视频/HTTP 入口一致）
        imgs = [ident.encoder.load_image(p_) for p_ in args.image]
        # 三方优先级：--with-actors 强制开 > --no-actors 强制关 > 配置默认（开）
        with_actors = True if args.with_actors else (False if args.no_actors else True)
        result = ident.identify_images(
            imgs, kind=args.kind, top_k=args.top_k, with_actors=with_actors)
        result["sources"] = args.image  # 回填来源，便于对照多张输入
    else:
        p.print_help()
        return

    # default=str：结果里可能混入 numpy 标量/Path 等不可直接序列化的对象
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
