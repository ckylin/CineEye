"""扩展步骤 4：导入演员照片并打标签，生成可训练的数据集。

用途：把「一批演员照片」变成「带标签、去重、划分好的数据集」。
照片不限场合——剧照、生活照、路透、写真、红毯照都可以，模型只学"这张脸是谁"。
支持两种打标签方式：

    方式 A：目录即标签（最省事）
        data/actors/刘德华/001.jpg
        data/actors/刘德华/2024/活动照.jpg
        data/actors/梁朝伟/xxx.jpg
        python scripts/15_import_actors.py --src data/actors

    方式 B：CSV 清单（照片可以散落在任何位置）
        # photos.csv
        path,actor,name_en
        /data/photos/a1.jpg,刘德华,Andy Lau
        /data/photos/b2.jpg,梁朝伟,Tony Leung
        python scripts/15_import_actors.py --csv photos.csv

输入/输出：
    读取：--src 目录或 --csv 文件
    输出：--dest（默认 data/actors）下的照片归档 + manifest.csv；
          manifest 列：path, actor, person_id, split, sha1, face_ok, source
    副作用：在 outputs/cineeye.db 的 persons 表登记演员（可用 --no-db 关闭）

运维要点：
    1. 每人建议 ≥ 8 张、至少 3 张；少于 3 张的演员会被全部划入 train，
       只能"学"不能"验"，摘要里会标出来。
    2. 去重按文件内容 MD5，同一张图换名字重复导入只保留一份。
    3. 重复导入是安全的：manifest 会按本次输入整体重写（不是追加），
       所以增删照片后重跑本脚本即可，随后重跑 16_train_actors.py。
    4. --dry-run 只统计不落盘，适合先看看标签有没有打错。
    5. 演员名要和元数据库里的人名一致，否则识别出来查不到参演电影。
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cineeye.actors.dataset import add_face_flags, import_photos, read_manifest, write_manifest  # noqa: E402
from cineeye.actors.face import FaceDetector  # noqa: E402
from cineeye.cli import base_parser, load_cfg  # noqa: E402


def main() -> None:
    """命令行入口：校验参数 → 导入 → 可选的人脸检出率统计 → 打印摘要。"""
    p = base_parser("影瞳 · 演员照片导入与打标签")
    src_group = p.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--src", default=None,
                           help="照片根目录，子目录名即演员名（如 data/actors/刘德华/*.jpg）")
    src_group.add_argument("--csv", dest="csv_path", default=None,
                           help="CSV 标签文件，列：path,actor[,name_en]")
    p.add_argument("--dest", default=None, help="照片归档目录（默认取 actors.data_dir）")
    p.add_argument("--no-copy", action="store_true",
                   help="不复制照片，只登记原始路径（照片位置长期不变时使用）")
    p.add_argument("--val-ratio", type=float, default=None)
    p.add_argument("--test-ratio", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no-db", action="store_true", help="不往 persons 表登记演员")
    p.add_argument("--no-face-check", action="store_true",
                   help="跳过人脸检出率统计（大数据集可省时间）")
    p.add_argument("--dry-run", action="store_true", help="只统计，不复制、不写盘")
    args = p.parse_args()

    cfg = load_cfg(args)
    actors_cfg = cfg.get("actors", {})
    dest = args.dest or actors_cfg.get("data_dir", "data/actors")

    store = None
    if not args.no_db:
        from cineeye.metadata import MetaStore
        store = MetaStore(cfg=cfg)

    summary = import_photos(
        src=args.src,
        csv_path=args.csv_path,
        dest_dir=dest,
        copy=not args.no_copy,
        store=store,
        cfg=cfg,
        val_ratio=args.val_ratio if args.val_ratio is not None
        else float(actors_cfg.get("train", {}).get("val_ratio", 0.15)),
        test_ratio=args.test_ratio if args.test_ratio is not None
        else float(actors_cfg.get("train", {}).get("test_ratio", 0.15)),
        seed=args.seed if args.seed is not None
        else int(cfg.get("train", {}).get("seed", cfg.get("seed", 42))),
        dry_run=args.dry_run,
    )

    # 人脸检出率：帮助判断"照片质量够不够"（侧脸多、脸太小都会拉低这个数）
    manifest = Path(summary["manifest"])
    if not args.dry_run and not args.no_face_check and manifest.exists():
        detector = FaceDetector(cfg)
        records = add_face_flags(read_manifest(manifest), detector)
        write_manifest(records, manifest)
        ok = sum(1 for r in records if r.face_ok == "1")
        summary["face_detected"] = ok
        summary["face_detected_rate"] = round(ok / max(1, len(records)), 4)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
