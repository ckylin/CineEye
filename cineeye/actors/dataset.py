"""演员照片数据集：导入、打标签、去重、按演员分层划分、清单读写。

用途：把「一堆演员照片」变成可训练的数据集。支持两种打标签方式：
    1. 目录即标签：``data/actors/刘德华/*.jpg``，子目录名就是演员名；
    2. CSV 清单：``photo_path,actor_name[,name_en]``，适合照片散落在不同目录、
       或一个目录里混着多位演员的情况。
    照片既可以是剧照，也可以是生活照、路透、写真、红毯照——模型只关心
    「这张脸是谁」，不关心拍摄场合；真正影响效果的是清晰度、正脸比例和
    每人样本量（建议 ≥ 8 张，最少 3 张）。

被谁使用：scripts/15_import_actors.py（导入）、scripts/16_train_actors.py
    （读取 manifest 训练）、actors/recognizer.py（构建检索库时读同一份清单）。
依赖：OpenCV（读图与尺寸校验）、metadata.MetaStore（登记 persons）、
    actors/face.py（人脸裁剪）。
运维要点（改这里之前必须知道的事）：
    1. manifest 是训练的唯一数据源，手工改它等于改数据集；改动后必须重训。
    2. 去重按文件内容 MD5（不是文件名）：同一张图换个名字重复导入只会保留一份，
       避免刷高某个演员的样本数导致分类头偏向。
    3. 划分按「演员内分层」：每位演员都按 val/test 比例切分，保证验证集里
       出现的演员在训练集里也出现过（否则无法评估「认人」能力）。
       样本太少的演员（<3 张）全部进 train，并在摘要里标记，避免验证集为空。
    4. 路径以「相对项目根」的 POSIX 形式写盘，整个项目换机器/挪目录仍可用。
"""

from __future__ import annotations

import csv
import hashlib
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from ..config import ROOT
from ..utils import ensure_dir, get_logger

log = get_logger("cineeye.actors.data")

# 支持的图片扩展名（与图库、参考库保持一致）
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# manifest 的列顺序是落盘契约：训练脚本按列名读取，新增列请追加到末尾
MANIFEST_COLUMNS = ["path", "actor", "person_id", "split", "sha1", "face_ok", "source"]


@dataclass
class PhotoRecord:
    """一张已打标签的演员照片。

    Attributes:
        path: 图片路径（相对项目根的 POSIX 字符串；外部图片为绝对路径）。
        actor: 演员名（标签），必须与 persons.name 一致才能反查参演电影。
        person_id: 元数据库里的 person_id；未登记时为空串。
        split: train / val / test。
        sha1: 文件内容 MD5（字段名沿用 sha1 兼容旧清单，实际是 md5）。
        face_ok: "1" 表示人脸检测成功，"0" 表示回退整图/中心裁剪。
        source: 来源标记（dir / csv / manual），便于排查数据出处。
    """

    path: str
    actor: str
    person_id: str = ""
    split: str = ""
    sha1: str = ""
    face_ok: str = ""
    source: str = ""

    def to_row(self) -> dict[str, str]:
        """转成 manifest 的一行（全部字符串，避免 CSV 类型歧义）。"""
        d = asdict(self)
        return {k: str(d.get(k, "")) for k in MANIFEST_COLUMNS}


def file_md5(path: str | Path, block: int = 1 << 20) -> str:
    """计算文件内容 MD5（分块读取，避免大图占满内存）。

    Args:
        path: 文件路径。
        block: 每次读取的字节数，默认 1MB。

    Returns:
        32 位十六进制字符串；读取失败返回空串（调用方按"无法去重"处理）。
    """
    h = hashlib.md5()
    try:
        with Path(path).open("rb") as f:
            for chunk in iter(lambda: f.read(block), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError as exc:  # 权限/文件被占用等
        log.warning("读取文件失败，无法计算指纹：%s（%s）", path, exc)
        return ""


def rel_path(path: str | Path) -> str:
    """把路径转成「相对项目根」的 POSIX 字符串；项目外路径保留绝对形式。

    这样 manifest 在项目整体迁移后仍然有效（和 movie_images 的存储策略一致）。
    """
    p = Path(path)
    try:
        rp = p.resolve()
    except OSError:
        return str(p)
    if rp.is_relative_to(ROOT):
        return rp.relative_to(ROOT).as_posix()
    return str(rp)


def abs_path(path: str | Path) -> Path:
    """把 manifest 里的路径还原成绝对路径（相对路径基于项目根解析）。"""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def iter_actor_dirs(src: str | Path) -> list[tuple[str, Path]]:
    """扫描「子目录即演员名」的目录结构，返回 (演员名, 图片路径) 列表。

    只遍历一级子目录（跳过 . 开头的隐藏目录），子目录内部允许再分层
    （例如 ``刘德华/2020/xxx.jpg``），方便按年份/来源整理。
    结果按演员名、图片名排序，保证同样的目录内容产出稳定顺序（可复现）。

    Args:
        src: 数据根目录，例如 ``data/actors``。

    Returns:
        [(actor, image_path), ...]；目录不存在返回空列表。
    """
    out: list[tuple[str, Path]] = []
    root = Path(src)
    if not root.exists():
        log.warning("演员照片目录不存在：%s", root)
        return out
    for actor_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        for img in sorted(p for p in actor_dir.rglob("*") if p.suffix.lower() in IMG_EXTS):
            out.append((actor_dir.name, img))
    return out


def read_label_csv(csv_path: str | Path, path_col: str = "path",
                   actor_col: str = "actor", name_en_col: str = "name_en"
                   ) -> list[tuple[str, Path, str | None]]:
    """读取 CSV 标签文件，返回 (演员名, 图片路径, 英文名) 列表。

    兼容三种写法：
        - 带表头且列名为 path/actor/name_en（大小写不敏感）；
        - 表头是中文（图片路径/演员/英文名）时按列位置解析；
        - 完全无表头时按「第 1 列路径、第 2 列演员、第 3 列英文名」解析。
    相对路径按「CSV 文件所在目录」解析——把照片和 CSV 放一起最省事。

    Args:
        csv_path: CSV 文件路径。
        path_col: 路径列的列名（带表头时用于定位）。
        actor_col: 演员列的列名。
        name_en_col: 英文名列名（可缺省）。

    Returns:
        [(actor, image_path, name_en_or_None), ...]

    Raises:
        FileNotFoundError: CSV 不存在。
        ValueError: 某行缺少路径或演员名。
    """
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"标签文件不存在：{p}")
    base = p.parent
    rows: list[tuple[str, Path, str | None]] = []
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        reader = list(csv.reader(f))
    if not reader:
        return rows

    header = [c.strip().lower() for c in reader[0]]
    has_header = any(h in ("path", "actor", "图片路径", "演员", "文件名") for h in header)
    idx_path, idx_actor, idx_en = 0, 1, 2
    if has_header:
        def find(names: Iterable[str], default: int) -> int:
            for n in names:
                if n in header:
                    return header.index(n)
            return default
        idx_path = find([path_col.lower(), "path", "图片路径", "文件名"], 0)
        idx_actor = find([actor_col.lower(), "actor", "演员", "姓名"], 1)
        idx_en = find([name_en_col.lower(), "name_en", "英文名"], 2)
        data = reader[1:]
    else:
        data = reader

    for lineno, row in enumerate(data, start=2 if has_header else 1):
        if not row or all(not c.strip() for c in row):
            continue  # 跳过空行
        try:
            raw_path = row[idx_path].strip()
            actor = row[idx_actor].strip()
        except IndexError as exc:
            raise ValueError(f"{p} 第 {lineno} 行列数不足：{row}") from exc
        if not raw_path or not actor:
            raise ValueError(f"{p} 第 {lineno} 行缺少路径或演员名：{row}")
        img = Path(raw_path)
        if not img.is_absolute():
            img = base / img  # 相对 CSV 所在目录，而不是 CWD
        en = row[idx_en].strip() if len(row) > idx_en and row[idx_en].strip() else None
        rows.append((actor, img, en))
    return rows


def split_records(records: list[PhotoRecord], val_ratio: float = 0.15,
                  test_ratio: float = 0.15, seed: int = 42) -> list[PhotoRecord]:
    """按演员分层划分 train/val/test（原地写回 split 字段并返回同一列表）。

    设计要点：
        - 分层：每位演员单独切分，保证 val/test 里出现的演员训练时都见过——
          否则「认人」这件事根本没法评估；
        - 保底：每位演员至少留 1 张给 train；不足 3 张的演员全部进 train，
          并在日志里提示（这类演员只能"学"，不能"验"）；
        - 可复现：按演员名排序后各自用固定种子 shuffle，结果稳定。

    Args:
        records: 待划分的记录（需已填 actor）。
        val_ratio: 验证集比例。
        test_ratio: 测试集比例。
        seed: 随机种子。

    Returns:
        同一个 list（split 字段已更新）。
    """
    by_actor: dict[str, list[PhotoRecord]] = {}
    for r in records:
        by_actor.setdefault(r.actor, []).append(r)
    rng = random.Random(seed)  # 独立随机源，不污染全局 random 状态
    for actor in sorted(by_actor):
        items = sorted(by_actor[actor], key=lambda r: r.path)  # 先排序再打乱，保证可复现
        n = len(items)
        if n < 3:
            for r in items:
                r.split = "train"
            log.warning("演员 %s 只有 %d 张照片，全部划入 train（无法评估）", actor, n)
            continue
        n_val = int(round(n * val_ratio))
        n_test = int(round(n * test_ratio))
        # 保底 1 张训练样本：比例切完后若 train 为空，从 val/test 里挪回来
        while n - n_val - n_test < 1:
            if n_val >= n_test and n_val > 0:
                n_val -= 1
            elif n_test > 0:
                n_test -= 1
            else:
                break
        rng.shuffle(items)
        for i, r in enumerate(items):
            if i < n_train(n, n_val, n_test):
                r.split = "train"
            elif i < n_train(n, n_val, n_test) + n_val:
                r.split = "val"
            else:
                r.split = "test"
    return records


def n_train(total: int, n_val: int, n_test: int) -> int:
    """train 的边界下标（抽出来只是为了让上面的循环可读）。"""
    return total - n_val - n_test


def _validate_image(path: Path, min_side: int = 48) -> tuple[bool, str]:
    """校验图片可读且尺寸够大；返回 (是否可用, 说明)。"""
    img = cv2.imread(str(path))
    if img is None:
        return False, "无法解码"
    h, w = img.shape[:2]
    if min(h, w) < min_side:
        return False, f"尺寸过小 {w}x{h}"
    return True, ""


def import_photos(src: str | Path | None = None, csv_path: str | Path | None = None,
                  dest_dir: str | Path = "data/actors", *, copy: bool = True,
                  store: Any | None = None, cfg: dict | None = None,
                  val_ratio: float = 0.15,
                  test_ratio: float = 0.15, seed: int = 42,
                  dry_run: bool = False) -> dict[str, Any]:
    """导入演员照片 → 校验 → 去重 → 登记演员 → 划分 → 写 manifest。

    Args:
        src: 「子目录即演员名」的数据根目录（与 csv_path 二选一）。
        csv_path: CSV 标签文件（与 src 二选一）。
        dest_dir: 照片归档目录（默认 data/actors）；copy=False 时只登记原路径。
        copy: 是否把照片复制进 dest_dir/<演员名>/。复制的好处是数据集自包含，
            迁移/备份只需拷一个目录；不复制适合照片本来就长期不动的场景。
        store: MetaStore 实例；None 时用 cfg 里的 metadata.db 自建
            （用于登记 persons）。传 cfg 与不传的区别只影响写哪个数据库。
        cfg: 全局配置；store 为 None 时用它定位数据库。
        val_ratio / test_ratio / seed: 划分参数。
        dry_run: True 时只统计、不复制、不写盘（预览用）。

    Returns:
        摘要 dict：{images, actors, duplicates, invalid, per_actor: {...},
        manifest, copied, dry_run}。

    Raises:
        ValueError: src 与 csv_path 同时为空。
        FileNotFoundError: 指定的目录/CSV 不存在。
    """
    if src is None and csv_path is None:
        raise ValueError("必须提供 src 或 csv_path 之一")

    pairs: list[tuple[str, Path, str | None, str]] = []
    if src is not None:
        for actor, img in iter_actor_dirs(src):
            pairs.append((actor, img, None, "dir"))
    if csv_path is not None:
        for actor, img, en in read_label_csv(csv_path):
            pairs.append((actor, img, en, "csv"))

    dest = Path(dest_dir)
    if not dest.is_absolute():
        dest = ROOT / dest

    # 延迟导入，避免 dataset 模块在 import 期就拉起 SQLite/配置
    if store is None and not dry_run:
        from ..metadata import MetaStore
        store = MetaStore(cfg=cfg)

    seen: dict[str, PhotoRecord] = {}
    duplicates = 0
    invalid: list[dict[str, str]] = []
    cop_record: list[Path] = []

    for actor, img_path, name_en, source in pairs:
        actor = actor.strip()
        if not actor:
            invalid.append({"path": str(img_path), "reason": "演员名为空"})
            continue
        if not img_path.exists():
            invalid.append({"path": str(img_path), "reason": "文件不存在"})
            continue
        ok, why = _validate_image(img_path)
        if not ok:
            invalid.append({"path": str(img_path), "reason": why})
            continue
        digest = file_md5(img_path)
        if digest and digest in seen:
            duplicates += 1  # 同内容不同文件名：只保留第一次出现的
            continue

        # 归档：同名冲突时用内容前 12 位命名，天然避免覆盖
        target = img_path
        if copy:
            actor_dir = dest / actor
            target = actor_dir / f"{digest[:12]}{img_path.suffix.lower()}"
            if not dry_run:
                ensure_dir(actor_dir)
                if not target.exists():
                    shutil.copy2(img_path, target)
                    cop_record.append(target)

        rec = PhotoRecord(
            path=rel_path(target), actor=actor, sha1=digest, source=source,
            face_ok="",
        )
        if store is not None and not dry_run:
            rec.person_id = store.upsert_person(actor, name_en=name_en)
        seen[digest or rec.path] = rec

    records = list(seen.values())
    split_records(records, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)

    manifest_path = dest / "manifest.csv"
    if not dry_run:
        write_manifest(records, manifest_path)

    per_actor: dict[str, dict[str, int]] = {}
    for r in records:
        d = per_actor.setdefault(r.actor, {"train": 0, "val": 0, "test": 0, "total": 0})
        d[r.split] = d.get(r.split, 0) + 1
        d["total"] += 1

    summary = {
        "images": len(records),
        "actors": len(per_actor),
        "duplicates": duplicates,
        "invalid": len(invalid),
        "invalid_samples": invalid[:10],
        "per_actor": dict(sorted(per_actor.items())),
        "manifest": str(manifest_path),
        "copied": len(cop_record),
        "dry_run": dry_run,
    }
    log.info("演员照片导入完成：%d 张 / %d 位演员（重复 %d，无效 %d）",
             summary["images"], summary["actors"], duplicates, len(invalid))
    return summary


def write_manifest(records: list[PhotoRecord], path: str | Path) -> Path:
    """把记录写成 manifest.csv（UTF-8 with BOM，Excel 直接打开不乱码）。"""
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        w.writeheader()
        for r in records:
            w.writerow(r.to_row())
    return p


def read_manifest(path: str | Path) -> list[PhotoRecord]:
    """读取 manifest.csv，返回 PhotoRecord 列表（缺失列自动补空串）。

    Raises:
        FileNotFoundError: manifest 不存在（提示先跑 15_import_actors.py）。
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"找不到演员清单：{p}（请先运行 scripts/15_import_actors.py）")
    out: list[PhotoRecord] = []
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            out.append(PhotoRecord(
                path=row.get("path", ""), actor=row.get("actor", ""),
                person_id=row.get("person_id", ""), split=row.get("split", "") or "train",
                sha1=row.get("sha1", ""), face_ok=row.get("face_ok", ""),
                source=row.get("source", ""),
            ))
    return out


def add_photo_bytes(actor: str, data: bytes, dest_dir: str | Path = "data/actors", *,
                    store: Any | None = None, cfg: dict | None = None,
                    name_en: str | None = None, split: str = "train",
                    min_side: int = 48) -> dict[str, Any]:
    """把一张"上传的照片"写入数据集并打上标签（增量，不重排已有划分）。

    与 import_photos 的分工：
        - import_photos 面向"一批照片"，会重新分层、整体重写 manifest；
        - add_photo_bytes 面向"一张一张传"，只追加一条记录，split 默认 train。
          传够一批后仍建议重跑 15_import_actors.py 让划分重新均衡。

    Args:
        actor: 演员名（标签），会同步登记到 persons 表。
        data: 图片二进制内容（HTTP base64 解码后的 bytes）。
        dest_dir: 照片归档目录。
        store: MetaStore 实例；None 时用 cfg 的 metadata.db 自建。
        cfg: 全局配置（定位数据库）。
        name_en: 英文名，可选。
        split: 初始划分，默认 train。
        min_side: 最小边像素阈值。

    Returns:
        {added, duplicate, path, actor, sha1}；重复照片 added=False。

    Raises:
        ValueError: 图片无法解码或尺寸过小。
    """
    import numpy as _np

    arr = _np.frombuffer(data, dtype=_np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("上传内容不是可解码的图片")
    h, w = img.shape[:2]
    if min(h, w) < min_side:
        raise ValueError(f"图片尺寸过小（{w}x{h}），最小边需 ≥ {min_side}px")

    actor = (actor or "").strip()
    if not actor:
        raise ValueError("必须提供演员名 actor")

    digest = hashlib.md5(data).hexdigest()
    dest = Path(dest_dir)
    if not dest.is_absolute():
        dest = ROOT / dest
    manifest_path = dest / "manifest.csv"

    # 已存在同内容照片 → 直接返回重复，不重复写盘/写清单
    if manifest_path.exists():
        for r in read_manifest(manifest_path):
            if r.sha1 == digest:
                return {"added": False, "duplicate": True, "path": r.path,
                        "actor": r.actor, "sha1": digest}

    actor_dir = dest / actor
    ensure_dir(actor_dir)
    ext = ".jpg"  # 统一转存为 jpg，避免 png/webp 混装带来的解码差异
    out_path = actor_dir / f"{digest[:12]}{ext}"
    if not out_path.exists():
        cv2.imwrite(str(out_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    if store is None:
        from ..metadata import MetaStore
        store = MetaStore(cfg=cfg)
    person_id = store.upsert_person(actor, name_en=name_en)

    rec = PhotoRecord(path=rel_path(out_path), actor=actor, person_id=person_id,
                      split=split, sha1=digest, face_ok="", source="upload")
    records = read_manifest(manifest_path) if manifest_path.exists() else []
    records.append(rec)
    write_manifest(records, manifest_path)
    log.info("已上传演员照片：%s → %s（累计 %d 张）", actor, rec.path, len(records))
    return {"added": True, "duplicate": False, "path": rec.path,
            "actor": actor, "sha1": digest, "total": len(records)}


def center_square(img: np.ndarray, ratio: float = 1.0) -> np.ndarray:
    """裁出以图片中心为基准的正方形区域。

    用途：人脸检测失败时的降级方案。相比直接缩放整图，中心裁切能减少
    两侧背景/字幕的干扰；ratio<1 时取更小的中心区域（更聚焦主体）。
    """
    h, w = img.shape[:2]
    side = int(min(h, w) * max(0.1, min(1.0, ratio)))
    cy, cx = h // 2, w // 2
    y1, x1 = max(0, cy - side // 2), max(0, cx - side // 2)
    return img[y1:y1 + side, x1:x1 + side]


def prepare_face(img: np.ndarray, detector: Any | None, mode: str = "detect",
                 pad: float = 0.2, fallback_whole: bool = True) -> tuple[np.ndarray | None, bool]:
    """把一张原图变成"用于识别的脸部区域"，训练与推理共用同一套规则。

    规则（顺序很重要，训练/推理必须完全一致，否则特征分布对不上）：
        1. mode="detect" 且检测器可用：取最大人脸并外扩 pad；
        2. 检测失败（或 mode="center"）：中心正方形裁剪；
        3. mode="whole"：直接用整图；
        4. fallback_whole=False 且检测失败：返回 (None, False)，上层"不猜"。

    Args:
        img: BGR 原图。
        detector: FaceDetector 实例；None 表示不做检测。
        mode: detect / center / whole。
        pad: 人脸外扩比例（保留发型、下颌轮廓）。
        fallback_whole: 检测失败时是否降级为中心裁剪。

    Returns:
        (裁剪图或 None, 是否检测到人脸)。第二项会写进 manifest 的 face_ok，
        便于统计"多少样本其实是整图"。
    """
    if img is None or img.size == 0:
        return None, False
    if mode == "whole" or detector is None:
        return (img if mode == "whole" else center_square(img)), False
    if mode == "center":
        return center_square(img), False
    crop = detector.largest_face(img, pad=pad)
    if crop is not None and crop.size > 0:
        return crop, True
    if fallback_whole:
        return center_square(img), False
    return None, False


def add_face_flags(records: list[PhotoRecord], detector: Any | None,
                   pad: float = 0.2) -> list[PhotoRecord]:
    """给记录回填 face_ok（是否检测到人脸），仅用于统计与数据质量排查。

    不改变 manifest 的顺序，也不裁剪图片；真正的裁剪在训练/推理时做。
    """
    for r in records:
        img = cv2.imread(str(abs_path(r.path)))
        if img is None:
            r.face_ok = "0"
            continue
        _, ok = prepare_face(img, detector, mode="detect", pad=pad)
        r.face_ok = "1" if ok else "0"
    return records
