"""扩展步骤 1：把电影档案导入元数据库。

用途：训练只产出 movie_XX 这样的类别编号，要让系统回答"这是哪部电影、演了谁"，
就必须有一份档案库。本脚本把片名/别名/年份/导演/演员/海报写进 SQLite
（outputs/cineeye.db），供 14_identify.py 与 /movies /search 接口查询。

典型调用：
    # 本地表格（推荐，离线可控）
    python scripts/11_import_metadata.py --source local --file my_movies.csv

    # TMDB（需 TMDB_API_KEY 环境变量或 --set metadata.providers.tmdb.api_key=xxx）
    python scripts/11_import_metadata.py --source tmdb --titles "电影A" "电影B"

    # 豆瓣（无 key，受反爬影响，请调大 request_interval）
    python scripts/11_import_metadata.py --source douban --titles "电影A" "电影B"

    # 从已有训练产物自动生成占位档案（movie_01~06，无片名，供图库检索先跑通）
    python scripts/11_import_metadata.py --source labelmap

    # 导出完整档案
    python scripts/11_import_metadata.py --export outputs/metadata/export.json

输入/输出：
    读取：--file 本地表格、outputs/label_map.json、data/manifest.csv、
          outputs/reports/movie_id_map.json（三者供 labelmap 模式用）
    输出：outputs/cineeye.db（SQLite：movies/aliases/people/credits/images 等表）；
          --export 时另写一个 JSON 快照

运维要点：
    1. 四种来源可混用，按 movie_id 匹配，后者覆盖前者。也就是说：
       先用 labelmap 建占位档案 → 再用 local/tmdb 覆盖同一个 movie_id，即可补齐真实信息。
    2. labelmap 模式故意不写真实片名：占位档案的标题就是 movie_01，
       真实片名必须由 local/tmdb/douban 显式提供，便于控制对外暴露的信息。
    3. 豆瓣抓取无官方 API，容易被封；批量抓请把 request_interval 调到 2~3 秒。
    4. 元数据库是幂等 upsert，可反复执行；重跑不会产生重复电影。
    5. --device 对本步无实际意义（不跑模型），但 base_parser 统一提供；
       --set 可用于覆盖 providers 的 key / 间隔等配置，优先级高于配置文件。
"""

import argparse
import json
import sys
from pathlib import Path

# 直接执行脚本时 sys.path[0] 是 scripts/，需补项目根
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cineeye.cli import base_parser, load_cfg  # noqa: E402
from cineeye.metadata import MetaStore  # noqa: E402
from cineeye.metadata.providers import build_provider  # noqa: E402
from cineeye.utils import ensure_dir, get_logger  # noqa: E402

log = get_logger("cineeye.import")


def import_labelmap(store: MetaStore, cfg: dict) -> int:
    """从 label_map.json + movie_id_map.json 生成占位档案。

    训练产物只有类别编号，这里给每个编号建一条"空壳"电影记录（标题=编号本身），
    并把 data/frames/<编号>/ 下的帧登记为图库图片，好让 12_build_gallery.py
    与 14_identify.py 在没有外部元数据时也能先跑通端到端。
    返回处理的类别数。
    """
    import pandas as pd

    n = 0
    label_map_path = Path(cfg["data"]["label_map"])
    if not label_map_path.is_absolute():
        # 配置里的路径都相对项目根，这里手工解析，避免依赖 cineeye.config.path 的副作用
        label_map_path = Path(__file__).resolve().parents[1] / label_map_path
    if not label_map_path.exists():
        # 没训练过就没有类别可用，属于正常情况，只告警不报错
        log.warning("找不到 %s，跳过占位档案", label_map_path)
        return 0
    label_map = json.loads(label_map_path.read_text(encoding="utf-8"))

    # movie_id_map 仅用于追溯原始片名，缺失也不影响建档；这里读它只是保持与 10 步的联动
    id_map_path = Path(__file__).resolve().parents[1] / "outputs/reports/movie_id_map.json"
    id_map = json.loads(id_map_path.read_text(encoding="utf-8")) if id_map_path.exists() else {}

    manifest_path = Path(__file__).resolve().parents[1] / "data/manifest.csv"
    year_by_label: dict[str, int | None] = {}
    if manifest_path.exists():
        df = pd.read_csv(manifest_path)
        for label in label_map:
            sub = df[df["label"] == label]
            year_by_label[label] = None  # 训练产物不含年份，留空由外部数据源补充

    for label in label_map:
        # 占位档案保持中性编号：不从源文件名提取片名。
        # 真实片名/演员请通过 local / tmdb / douban 提供者显式导入，
        # 便于控制对外暴露的信息。
        title = label
        # aliases 里塞一条 alias=label / lang=id，让 /search?q=movie_01 也能命中
        store.upsert_movie(
            label, title,
            original_title=None,
            aliases=[{"alias": label, "lang": "id", "source": "labelmap"}],
            provider="labelmap",
        )
        # 把已有帧登记为图库图片（frame 类型）
        frames_dir = Path(__file__).resolve().parents[1] / "data/frames" / label
        if frames_dir.exists():
            count = 0
            for img in sorted(frames_dir.rglob("*.jpg")):
                if count >= 40:  # 每部最多登记 40 张，避免索引过大
                    break
                # 只登记不编码：真正算向量在 12_build_gallery.py（关注点分离）
                store.add_image(label, img, kind="frame", source="labelmap")
                count += 1
        n += 1
    return n


def main() -> None:
    """命令行入口：按 --source 选择来源，写入元数据库，或只做 --export。"""
    p = base_parser("影瞳 · 导入电影元数据")
    p.add_argument("--source", choices=["local", "tmdb", "douban", "labelmap"], default="local")
    p.add_argument("--file", default=None, help="本地表格路径（source=local）")
    p.add_argument("--titles", nargs="*", default=[], help="要抓取的片名列表（source=tmdb/douban）")
    p.add_argument("--export", default=None, help="导出完整档案到 JSON 后退出")
    args = p.parse_args()

    cfg = load_cfg(args)
    store = MetaStore(cfg=cfg)

    if args.export:
        # 导出是只读操作，导出完直接返回，不做任何导入
        out = store.export_json(args.export)
        print(f"档案已导出：{out}")
        return

    if args.source == "labelmap":
        # 无外部数据源时最快的跑通方式：占位档案 + 登记已有帧
        n = import_labelmap(store, cfg)
    elif args.source == "tmdb" or args.source == "douban":
        provider = build_provider(args.source, cfg)
        if args.source == "tmdb" and not getattr(provider, "available", False):
            # TMDB 必须配 key，否则所有请求都会 401；提前失败并给出设置方法
            p.error("未配置 TMDB_API_KEY：请设置环境变量或 --set metadata.providers.tmdb.api_key=你的key")
        if not args.titles:
            p.error("请提供 --titles 片名列表")
        n = 0
        for title in args.titles:
            # 逐个片名抓取；单条失败只告警，不中断整批
            rec = provider.fetch_by_title(title)
            if rec is None:
                log.warning("未找到：%s", title)
                continue
            # to_kwargs() 把 provider 的统一记录结构摊平成 upsert_movie 的关键字参数
            store.upsert_movie(rec.movie_id, **rec.to_kwargs())
            log.info("导入 %s：%s（%d 别名 / %d 演员）",
                     args.source, rec.primary_title, len(rec.aliases), len(rec.cast))
            n += 1
    else:
        # local：读整张表格批量导入（CSV/JSON 由 provider 自己识别）
        provider = build_provider("local", cfg)
        if args.file:
            provider = build_provider("local", cfg)
            provider.path = Path(args.file)  # 覆盖配置里的默认表格路径
        records = provider.fetch_all()
        for rec in records:
            store.upsert_movie(rec.movie_id, **rec.to_kwargs())
        n = len(records)

    print(f"\n导入完成：{n} 部电影")
    print("库内统计：", json.dumps(store.stats(), ensure_ascii=False))


if __name__ == "__main__":
    main()
