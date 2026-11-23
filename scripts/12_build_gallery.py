"""扩展步骤 2：构建/更新图库向量索引。

用途：把"图片 → 向量"这一步离线做掉，运行时只做一次前向编码 + 余弦相似度检索。
图库（海报/剧照/演员照/路演照）是开放集识别的依据：模型新增一部片，只需往图库
里加图，不用重新训练分类器。

典型调用：
    # 从元数据库登记的图片构建（推荐，先跑 11_import_metadata.py）
    python scripts/12_build_gallery.py

    # 全量重建（换了模型或编码器时）
    python scripts/12_build_gallery.py --rebuild

    # 追加一个目录下的图片作为指定类型
    python scripts/12_build_gallery.py --add-dir data/posters/电影名 --kind poster --movie-id movie_01

    # 查看索引状态
    python scripts/12_build_gallery.py --stats

输入/输出：
    读取：outputs/cineeye.db 里登记的图片路径（默认模式）、或 --add-dir 指定的目录；
          outputs/checkpoints/best.pt（复用训练好的主干做编码器）
    输出：outputs/gallery/embeddings.npy（N×D，已 L2 归一化）、
          outputs/gallery/meta.json（逐行对应的 movie_id/kind/path）、
          outputs/gallery/index.meta.json（维度、条数、编码器指纹、更新时间）

增量索引原理（运维必须懂这一条）：
    GalleryIndex.add() 会先用「(movie_id, kind, path) 三元组」做集合比对，
    已存在的图片直接跳过、不重复编码，只把新增的图编码后 vstack 到旧矩阵上——
    因此日常追加图片只花"新增那一部分"的时间，几千张规模下是秒级。
    而 --rebuild（rebuild()）会清空矩阵后 force 全量重编码，只在下面两种情况用：
      a) 换了 backbone / 权重，向量空间变了，旧向量与新向量不可比；
      b) index.meta.json 里的 fingerprint 与当前编码器不一致（add 时维度不一致会直接报错）。
    fingerprint 记录的是编码器身份（主干名+特征维度等），是"该不该重建"的唯一线索。

运维要点：
    1. 默认走元数据库：图片必须先在 11_import_metadata.py 里登记过（store.add_image）。
    2. --limit N 只处理前 N 张，用于快速验证链路，正式建库别带。
    3. 索引是"整体重写"而不是追加文件：中途 Ctrl+C 可能留下不完整索引，重跑一次即可。
    4. --device 写入 cfg.train.device 后由 ImageEncoder 读取，决定编码走 CPU 还是 GPU；
       在 GPU 机器上建库会快一个数量级。
"""

import argparse
import sys
from pathlib import Path

# 直接执行脚本时 sys.path[0] 是 scripts/，需补项目根
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cineeye.cli import base_parser, load_cfg  # noqa: E402
from cineeye.gallery import GalleryIndex, ImageEncoder  # noqa: E402
from cineeye.metadata import MetaStore  # noqa: E402
from cineeye.utils import get_logger  # noqa: E402

log = get_logger("cineeye.gallery.build")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    """命令行入口：--stats 只读；--add-dir 追加目录；默认从元数据库取图并增量建索引。"""
    p = base_parser("影瞳 · 构建图库索引")
    p.add_argument("--rebuild", action="store_true", help="全量重建索引")
    p.add_argument("--add-dir", default=None, help="追加一个目录的图片")
    p.add_argument("--kind", default="poster", help="追加图片的类型（配合 --add-dir）")
    p.add_argument("--movie-id", default=None, help="追加图片所属电影（配合 --add-dir）")
    p.add_argument("--stats", action="store_true", help="只打印索引状态")
    p.add_argument("--limit", type=int, default=0, help="最多处理图片数（0=不限），用于快速验证")
    args = p.parse_args()

    cfg = load_cfg(args)
    # 构造时会自动从 outputs/gallery 加载已有索引（没有则视为空），这是增量更新的基础
    index = GalleryIndex(cfg=cfg)

    if args.stats:
        # 只读操作：不解编码器、不改文件，可随时执行（巡检用）
        import json
        print(json.dumps(index.stats(), ensure_ascii=False, indent=2))
        return

    if args.add_dir:
        if not args.movie_id:
            p.error("--add-dir 需要同时提供 --movie-id")
        d = Path(args.add_dir)
        if not d.is_absolute() and not d.exists():
            # 相对路径先按当前目录找，找不到再按项目根找，兼容两种使用习惯
            d = Path(__file__).resolve().parents[1] / d
        # rglob 递归收集（海报目录常按子目录分组），后缀过滤掉非图片文件
        files = sorted(f for f in d.rglob("*") if f.suffix.lower() in IMG_EXTS)
        if args.limit:
            files = files[:args.limit]
        # 统一成 add() 需要的记录格式；kind 决定检索时能否按类型过滤
        records = [{"movie_id": args.movie_id, "kind": args.kind, "path": str(f)} for f in files]
        log.info("追加 %d 张 %s 图片 → %s", len(records), args.kind, args.movie_id)
    else:
        # 默认来源：元数据库里已登记的图片（11 步写入），带 movie_id/kind/path 三要素
        store = MetaStore(cfg=cfg)
        records = store.images_for_index()
        if args.limit:
            records = records[:args.limit]
        log.info("从元数据库读取 %d 张已登记图片", len(records))

    # 显式构造编码器并注入索引：这样 --device 参数（经 load_cfg 写入 train.device）才能生效，
    # 否则索引会自己延迟创建一个按配置默认设备跑的编码器
    enc = ImageEncoder(cfg=cfg)
    index.encoder = enc
    # rebuild = 清空后全量重编码；默认 add = 跳过已存在项、只编码新增图片
    added = index.rebuild(records) if args.rebuild else index.add(records)

    import json
    print(f"\n处理完成：新增 {added} 张")
    print(json.dumps(index.stats(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
