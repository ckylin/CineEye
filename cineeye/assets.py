"""素材资产层：统一登记视频/图片素材及其五层标签，并兼容现有训练流程。

用途：
    为 Web 管理后台提供「素材库」的单一真相源。每一张上传的图片、每一段
    上传的视频都对应 assets 表里一条记录，携带五层标签（基础属性/内容分类/
    语义关联/训练用途/合规采集）。多电影、多演员关联分别存到关联表。
    同时负责把标签「派生写回」到现有流程能读懂的产物（movie_images 表、
    data/actors/manifest.csv），从而不改动已经跑通的训练脚本。

被谁使用：
    - serve.py 的 /admin/upload、/admin/assets 系列接口；
    - cineeye/jobs.py 训练任务（读取 assets 决定抽帧/训练输入）；
    - 运维脚本（老数据反向导入 import_legacy）。

依赖：
    - metadata.MetaStore 的 SQLite 连接约定（同库 outputs/cineeye.db）；
    - configs/tag_schema.json（标签字段与枚举的契约）；
    - stdlib sqlite3/cv2（读图取尺寸、抽首帧）。

运维要点（改这里之前必须知道的事）：
    1. assets 等 5 张表用 CREATE TABLE IF NOT EXISTS 幂等创建，不改动 movies/
       persons 等旧表；新增列请追加并用 ALTER TABLE 迁移旧库。
    2. asset_id 由内容 sha256 派生，天然幂等：同一文件重复上传只会得到同一
       ID（UNIQUE 约束），不会刷高样本数。
    3. 派生写回是「单向覆盖 + 增量」：改标签后重新计算该素材对应的旧产物，
       不会删除用户手工放进 data/frames 的其他文件。
    4. file_path 一律存相对项目根的 POSIX 字符串，跨机器可搬运。
    5. 枚举值以 configs/tag_schema.json 为准；本模块的 ENUMS 仅作兜底，
       validate_tags() 会优先读 JSON。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import cv2

from .config import ROOT, get_value
from .utils import ensure_dir, get_logger

log = get_logger("cineeye.assets")

# 标签契约路径（相对项目根）
SCHEMA_PATH = ROOT / "configs" / "tag_schema.json"

# 枚举兜底：正常从 tag_schema.json 读取，读不到时用它，保证离线/旧部署可用
ENUMS: dict[str, list[str]] = {
    "asset_type": ["video", "image"],
    "source": ["upload", "local", "tmdb", "douban", "manual"],
    "status": ["active", "archived"],
    "media_kind": ["frame", "poster", "still", "behind_scenes", "roadshow",
                   "actor_portrait", "actor_life", "screenshot", "trailer", "clip", "other"],
    "purpose": ["train", "query", "gallery", "reference"],
    "split": ["", "train", "val", "test"],
    "quality": ["", "high", "medium", "low"],
}

# media_kind -> 默认 purpose（与 tag_schema.json 的 media_kind_defaults 保持一致）
KIND_DEFAULT_PURPOSE = {
    "frame": "train", "poster": "gallery", "still": "gallery",
    "behind_scenes": "gallery", "roadshow": "gallery",
    "actor_portrait": "reference", "actor_life": "reference",
    "screenshot": "query", "trailer": "train", "clip": "train", "other": "query",
}

# 哪些 media_kind 属于「电影画面」，需要派生写回 movie_images 表
MOVIE_IMAGE_KINDS = {"frame", "poster", "still", "behind_scenes", "roadshow"}
# 哪些 media_kind 属于「演员照片」，需要派生写回 actors/manifest.csv
ACTOR_IMAGE_KINDS = {"actor_portrait", "actor_life"}

# assets 表的全部可变标签字段（补标接口可写），不含只读的基础属性
MUTABLE_FIELDS = [
    "source", "source_url", "media_kind", "lang", "region", "purpose", "split",
    "label", "quality", "face_ok", "face_boxes", "used_in", "license", "consent",
    "notes", "status",
]

SCHEMA = """
CREATE TABLE IF NOT EXISTS assets (
    asset_id    TEXT PRIMARY KEY,
    asset_type  TEXT NOT NULL,
    file_path   TEXT NOT NULL,
    file_name   TEXT,
    sha256      TEXT NOT NULL UNIQUE,
    size        INTEGER,
    width       INTEGER,
    height      INTEGER,
    duration    REAL,
    ext         TEXT,
    source      TEXT DEFAULT 'upload',
    source_url  TEXT,
    media_kind  TEXT DEFAULT 'other',
    lang        TEXT,
    region      TEXT,
    purpose     TEXT DEFAULT 'query',
    split       TEXT,
    label       TEXT,
    quality     TEXT,
    face_ok     INTEGER DEFAULT 0,
    face_boxes  TEXT,
    used_in     TEXT,
    license     TEXT,
    consent     INTEGER,
    notes       TEXT,
    status      TEXT DEFAULT 'active',
    uploaded_at TEXT DEFAULT (datetime('now')),
    updated_at  TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS asset_movies (
    asset_id TEXT NOT NULL REFERENCES assets(asset_id) ON DELETE CASCADE,
    movie_id TEXT NOT NULL REFERENCES movies(id) ON DELETE CASCADE,
    PRIMARY KEY (asset_id, movie_id)
);
CREATE TABLE IF NOT EXISTS asset_persons (
    asset_id  TEXT NOT NULL REFERENCES assets(asset_id) ON DELETE CASCADE,
    person_id TEXT NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    PRIMARY KEY (asset_id, person_id)
);
CREATE TABLE IF NOT EXISTS asset_tags (
    asset_id TEXT NOT NULL REFERENCES assets(asset_id) ON DELETE CASCADE,
    tag      TEXT NOT NULL,
    PRIMARY KEY (asset_id, tag)
);
CREATE TABLE IF NOT EXISTS jobs (
    id         TEXT PRIMARY KEY,
    kind       TEXT NOT NULL,
    status     TEXT NOT NULL,
    payload    TEXT,
    progress   INTEGER DEFAULT 0,
    log_path   TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    started_at TEXT,
    finished_at TEXT,
    result     TEXT
);
CREATE INDEX IF NOT EXISTS idx_assets_kind ON assets(media_kind);
CREATE INDEX IF NOT EXISTS idx_assets_sha  ON assets(sha256);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
"""


def load_tag_schema() -> dict[str, Any]:
    """读取标签契约 JSON；失败时返回带默认枚举的最小结构，保证系统可用。

    Returns:
        解析后的 dict，至少含 version/layers/enums 键。
    """
    try:
        return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning("标签契约读取失败，使用内置兜底枚举：%s", exc)
        return {"version": "0", "layers": [], "enums": ENUMS,
                "media_kind_defaults": KIND_DEFAULT_PURPOSE}


def file_sha256(path: str | Path, block: int = 1 << 20) -> str:
    """按内容计算 sha256（分块读，避免大视频占满内存）。

    Args:
        path: 文件路径。
        block: 分块字节数，默认 1MiB。

    Returns:
        64 位十六进制字符串；读取失败返回空串。
    """
    h = hashlib.sha256()
    try:
        with Path(path).open("rb") as f:
            for chunk in iter(lambda: f.read(block), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError as exc:
        log.warning("读取文件失败，无法计算 sha256：%s（%s）", path, exc)
        return ""


def validate_tags(tags: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """按标签契约校验并规范化一组标签值。

    只校验契约里声明的枚举字段；未知字段原样保留（前向兼容）。
    bool 字段统一成 0/1；tags/multi_ref 统一成字符串列表。

    Args:
        tags: 待校验的标签 dict（键为字段名）。

    Returns:
        (cleaned, errors)：cleaned 为规范化后的 dict，errors 为错误说明列表；
        errors 为空表示通过。
    """
    schema = load_tag_schema()
    enums = schema.get("enums", ENUMS)
    cleaned: dict[str, Any] = {}
    errors: list[str] = []
    for key, value in tags.items():
        if key in enums and value not in (None, ""):
            if value not in enums[key]:
                errors.append(f"{key}: 非法值 {value!r}，可选 {enums[key]}")
                continue
        if key in ("face_ok", "consent"):
            cleaned[key] = 1 if value in (True, 1, "1", "true", "True") else 0
        elif key in ("free_tags", "used_in"):
            if isinstance(value, str):
                value = [v.strip() for v in value.split(",") if v.strip()]
            cleaned[key] = json.dumps(list(value), ensure_ascii=False)
        elif key in ("movie_ids", "person_ids"):
            cleaned[key] = list(value) if isinstance(value, (list, tuple)) else value
        else:
            cleaned[key] = value
    return cleaned, errors


def _rel(path: str | Path) -> str:
    """把路径转成相对项目根的 POSIX 字符串；外部绝对路径原样保留。"""
    p = Path(path)
    try:
        return p.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


class AssetStore:
    """素材资产库：assets 表及其关联表的增删改查与派生写回。

    Args:
        db_path: SQLite 文件路径；None 时按 cfg.metadata.db 解析，
            默认 ROOT/outputs/cineeye.db（与 MetaStore 同库）。
        cfg: 全局配置 dict。

    运维要点：
        - 与 MetaStore 共用同一个 .db 文件，因此外键能直接引用 movies/persons；
        - 所有写方法都用短连接，天然支持多进程/多线程并发（timeout=30）。
    """

    def __init__(self, db_path: str | Path | None = None, cfg: dict | None = None):
        if db_path is None:
            raw = get_value(cfg or {}, "metadata.db", "outputs/cineeye.db")
            p = Path(raw)
            db_path = p if p.is_absolute() else ROOT / p
        self.path = Path(db_path)
        ensure_dir(self.path.parent)
        with self._conn() as c:
            # 先建电影元数据基础表：assets 的关联表外键引用 movies/persons，
            # 资产库可能在 MetaStore 之前被初始化（如独立部署/测试），必须自包含。
            from .metadata.store import SCHEMA as META_SCHEMA
            c.executescript(META_SCHEMA)
            c.executescript(SCHEMA)  # 幂等建表
        self.cfg = cfg or {}
        log.info("素材库就绪：%s", self.path)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        """短连接上下文，退出时提交，异常时回滚并关闭。"""
        conn = sqlite3.connect(str(self.path), timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ---------------------------------------------------------------- 写入
    def add(self, file_path: str | Path, *, asset_type: str | None = None,
            source: str = "upload", tags: dict[str, Any] | None = None) -> dict[str, Any]:
        """登记一个素材文件（幂等：同内容返回已有记录）。

        自动提取基础属性（sha256/尺寸/时长/扩展名），并按 asset_type 与
        media_kind 推导默认 purpose。重复内容（sha256 相同）不会新建记录。

        Args:
            file_path: 文件路径（项目内相对或绝对均可）。
            asset_type: video/image；None 时按扩展名推断。
            source: 来源标记。
            tags: 附加标签（会与默认值合并，显式值覆盖默认值）。

        Returns:
            素材完整 dict（见 get）。

        Raises:
            FileNotFoundError: 文件不存在。
            ValueError: 无法推断素材类型（扩展名不认识）。
        """
        p = Path(file_path)
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            raise FileNotFoundError(f"素材文件不存在：{p}")

        ext = p.suffix.lower().lstrip(".")
        if asset_type is None:
            if ext in {"mp4", "avi", "mov", "mkv", "flv", "webm", "m4v"}:
                asset_type = "video"
            elif ext in {"jpg", "jpeg", "png", "bmp", "webp", "gif"}:
                asset_type = "image"
            else:
                raise ValueError(f"无法推断素材类型：{ext!r}，请显式指定 asset_type")

        sha = file_sha256(p)
        asset_id = "asset_" + (sha or hashlib.sha256(str(p).encode()).hexdigest())[:16]
        width = height = None
        duration = None
        if asset_type == "image":
            img = cv2.imread(str(p))
            if img is not None:
                height, width = img.shape[:2]
        else:
            cap = cv2.VideoCapture(str(p))
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
                fps = cap.get(cv2.CAP_PROP_FPS) or 0
                frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
                duration = round(frames / fps, 3) if fps > 0 else None
                cap.release()

        tags = tags or {}
        media_kind = tags.get("media_kind", "other")
        defaults = {
            "source": source,
            "media_kind": media_kind,
            "purpose": tags.get("purpose") or KIND_DEFAULT_PURPOSE.get(media_kind, "query"),
            "status": "active",
        }
        defaults.update({k: v for k, v in tags.items() if k in MUTABLE_FIELDS or k in ("movie_ids", "person_ids", "free_tags")})
        cleaned, errors = validate_tags(defaults)
        if errors:
            log.warning("素材 %s 部分标签非法，已忽略：%s", p.name, errors)

        filesize = p.stat().st_size
        now = datetime.now().isoformat(timespec="seconds")
        with self._conn() as c:
            row = c.execute("SELECT asset_id FROM assets WHERE sha256=?", (sha,)).fetchone() if sha else None
            if row:
                log.info("素材已存在（sha256 去重）：%s", asset_id if row["asset_id"] == asset_id else row["asset_id"])
                return self.get(row["asset_id"])

            cols = ["asset_id", "asset_type", "file_path", "file_name", "sha256", "size",
                    "width", "height", "duration", "ext", "uploaded_at", "updated_at"]
            vals = [asset_id, asset_type, _rel(p), p.name, sha, filesize, width, height,
                    duration, ext, now, now]
            for key in MUTABLE_FIELDS:
                if key in cleaned:
                    cols.append(key)
                    vals.append(cleaned[key])
            c.execute(
                f"INSERT INTO assets ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                vals,
            )
            self._write_relations(c, asset_id, cleaned)

        log.info("素材入库：%s（%s，%s）", p.name, asset_type, asset_id)
        return self.get(asset_id)

    def _write_relations(self, c: sqlite3.Connection, asset_id: str, tags: dict[str, Any]) -> None:
        """写入多对多关联（movie_ids/person_ids/free_tags），先清后写。"""
        c.execute("DELETE FROM asset_movies WHERE asset_id=?", (asset_id,))
        c.execute("DELETE FROM asset_persons WHERE asset_id=?", (asset_id,))
        c.execute("DELETE FROM asset_tags WHERE asset_id=?", (asset_id,))
        for mid in tags.get("movie_ids", []) or []:
            c.execute("INSERT OR IGNORE INTO asset_movies (asset_id,movie_id) VALUES (?,?)", (asset_id, str(mid)))
        for pid in tags.get("person_ids", []) or []:
            c.execute("INSERT OR IGNORE INTO asset_persons (asset_id,person_id) VALUES (?,?)", (asset_id, str(pid)))
        raw_tags = tags.get("free_tags", "[]")
        try:
            tag_list = json.loads(raw_tags) if isinstance(raw_tags, str) else raw_tags
        except (ValueError, TypeError):
            tag_list = []
        for tag in tag_list or []:
            c.execute("INSERT OR IGNORE INTO asset_tags (asset_id,tag) VALUES (?,?)", (asset_id, str(tag)))

    def update(self, asset_id: str, changes: dict[str, Any]) -> dict[str, Any] | None:
        """补标/改标：校验后更新可变字段与关联，并触发派生写回。

        Args:
            asset_id: 素材 ID。
            changes: 字段 -> 新值；只处理 MUTABLE_FIELDS 与关联三类。

        Returns:
            更新后的素材 dict；素材不存在返回 None。

        Raises:
            ValueError: 标签校验不通过（枚举非法）。
        """
        relation_keys = {"movie_ids", "person_ids", "free_tags"}
        scalar = {k: v for k, v in changes.items() if k in MUTABLE_FIELDS}
        cleaned, errors = validate_tags(scalar)
        if errors:
            raise ValueError("；".join(errors))
        rel_tags = {k: changes[k] for k in relation_keys if k in changes}

        with self._conn() as c:
            row = c.execute("SELECT asset_id FROM assets WHERE asset_id=?", (asset_id,)).fetchone()
            if not row:
                return None
            if cleaned:
                cleaned["updated_at"] = datetime.now().isoformat(timespec="seconds")
                sets = ",".join(f"{k}=?" for k in cleaned)
                c.execute(f"UPDATE assets SET {sets} WHERE asset_id=?", [*cleaned.values(), asset_id])
            if rel_tags:
                full = dict(rel_tags)
                # 关联写入需要完整的三类，缺失的从库里读现状补齐
                cur = self.get(asset_id)
                for k in relation_keys:
                    if k not in full and cur:
                        full[k] = cur.get(k)
                self._write_relations(c, asset_id, full)
        self.sync_derived(asset_id)
        return self.get(asset_id)

    def archive(self, asset_id: str) -> bool:
        """软删除：置 status=archived，不物理删除文件与记录。"""
        with self._conn() as c:
            cur = c.execute("UPDATE assets SET status='archived', updated_at=? WHERE asset_id=?",
                            (datetime.now().isoformat(timespec="seconds"), asset_id))
            return cur.rowcount > 0

    # ---------------------------------------------------------------- 读取
    def get(self, asset_id: str) -> dict[str, Any] | None:
        """按 ID 取单个素材完整标签（含关联列表）。不存在返回 None。"""
        with self._conn() as c:
            row = c.execute("SELECT * FROM assets WHERE asset_id=?", (asset_id,)).fetchone()
            if not row:
                return None
            d = dict(row)
            d["movie_ids"] = [r[0] for r in c.execute(
                "SELECT movie_id FROM asset_movies WHERE asset_id=?", (asset_id,))]
            d["person_ids"] = [r[0] for r in c.execute(
                "SELECT person_id FROM asset_persons WHERE asset_id=?", (asset_id,))]
            try:
                d["free_tags"] = json.loads(d.get("free_tags") or "[]")
            except (ValueError, TypeError):
                d["free_tags"] = []
            try:
                d["used_in"] = json.loads(d.get("used_in") or "[]")
            except (ValueError, TypeError):
                d["used_in"] = []
            return d

    def list(self, *, kind: str | None = None, purpose: str | None = None,
             q: str | None = None, status: str = "active",
             limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        """素材库列表，支持按 kind/purpose/关键词过滤与分页。"""
        sql = "SELECT * FROM assets WHERE 1=1"
        args: list[Any] = []
        if status:
            sql += " AND status=?"
            args.append(status)
        if kind:
            sql += " AND media_kind=?"
            args.append(kind)
        if purpose:
            sql += " AND purpose=?"
            args.append(purpose)
        if q:
            sql += " AND (file_name LIKE ? OR label LIKE ? OR notes LIKE ?)"
            args.extend([f"%{q}%"] * 3)
        sql += " ORDER BY uploaded_at DESC LIMIT ? OFFSET ?"
        args.extend([limit, offset])
        with self._conn() as c:
            return [dict(r) for r in c.execute(sql, args)]

    def stats(self) -> dict[str, Any]:
        """素材统计：总数、按类型/用途/内容类型分布。"""
        with self._conn() as c:
            total = c.execute("SELECT COUNT(*) FROM assets WHERE status='active'").fetchone()[0]
            by_type = {r[0]: r[1] for r in c.execute(
                "SELECT asset_type, COUNT(*) FROM assets WHERE status='active' GROUP BY asset_type")}
            by_kind = {r[0]: r[1] for r in c.execute(
                "SELECT media_kind, COUNT(*) FROM assets WHERE status='active' GROUP BY media_kind")}
            by_purpose = {r[0]: r[1] for r in c.execute(
                "SELECT purpose, COUNT(*) FROM assets WHERE status='active' GROUP BY purpose")}
        return {"total": total, "by_type": by_type, "by_kind": by_kind, "by_purpose": by_purpose}

    # ---------------------------------------------------------------- 派生
    def sync_derived(self, asset_id: str) -> dict[str, Any]:
        """把一个素材的标签派生写回到现有流程能读懂的产物。

        规则（见 WEB_UI_DESIGN.md 4.1）：
        - 电影画面类 image → movie_images 表（按关联 movie_id 逐个写）；
        - 演员照片类 image → outputs/actors/manifest.csv（素材库专用副本，
          不影响训练主清单，训练任务会重新生成）；
        - video 不在此写回，由 train 任务抽帧处理。

        Returns:
            {"movie_images": n, "actor_manifest": bool} 派生结果摘要。
        """
        a = self.get(asset_id)
        if not a or a.get("status") != "active":
            return {"movie_images": 0, "actor_manifest": False}
        result = {"movie_images": 0, "actor_manifest": False}
        kind = a.get("media_kind")
        if a.get("asset_type") == "image" and kind in MOVIE_IMAGE_KINDS and a.get("movie_ids"):
            with self._conn() as c:
                for mid in a["movie_ids"]:
                    cur = c.execute(
                        "INSERT OR IGNORE INTO movie_images (movie_id,kind,path,source) VALUES (?,?,?,?)",
                        (mid, kind, a["file_path"], a.get("source") or "upload"))
                    result["movie_images"] += cur.rowcount
        if a.get("asset_type") == "image" and kind in ACTOR_IMAGE_KINDS and a.get("person_ids"):
            # 写一份素材库派生清单；真正的训练清单由 15_import_actors.py 生成，
            # 这里只保证信息不丢，便于追溯。
            dest = ROOT / "outputs" / "metadata" / "asset_actors.csv"
            ensure_dir(dest.parent)
            self._append_actor_csv(dest, a)
            result["actor_manifest"] = True
        return result

    def _append_actor_csv(self, dest: Path, a: dict[str, Any]) -> None:
        """把演员照片素材追加/更新到派生 CSV（按 file_path 去重）。"""
        import csv
        existing: dict[str, list[str]] = {}
        if dest.exists():
            with dest.open("r", encoding="utf-8", newline="") as f:
                for row in csv.reader(f):
                    if row:
                        existing[row[0]] = row[1:]
        actors = ",".join(a.get("person_ids") or [])
        existing[a["file_path"]] = [a.get("label") or actors, a.get("split") or "", actors]
        with dest.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "actor", "split", "person_ids"])
            for path, row in existing.items():
                w.writerow([path, *row])

    def sync_all_derived(self) -> int:
        """全量派生（refresh_derived 任务用），返回处理的素材数。"""
        ids = [r[0] for r in self._select_ids()]
        for aid in ids:
            self.sync_derived(aid)
        return len(ids)

    def _select_ids(self) -> list[tuple[str]]:
        with self._conn() as c:
            return c.execute("SELECT asset_id FROM assets WHERE status='active'").fetchall()

    # ---------------------------------------------------------------- 兼容
    def import_legacy(self) -> dict[str, int]:
        """反向导入老数据：把现有 movie_images 与演员清单回填到 assets。

        幂等：靠 sha256 去重，重复执行不会新增记录。老图片 source 记为 'local'，
        media_kind 沿用其原 kind（演员清单的记为 actor_life）。

        Returns:
            {"movie_images": n, "actor_photos": n, "skipped": n}
        """
        counts = {"movie_images": 0, "actor_photos": 0, "skipped": 0}
        with self._conn() as c:
            rows = c.execute("SELECT movie_id, kind, path FROM movie_images").fetchall()
        for r in rows:
            p = ROOT / r["path"]
            if not p.exists():
                counts["skipped"] += 1
                continue
            try:
                before = len(self.list(limit=1))
                self.add(p, asset_type="image", source="local",
                         tags={"media_kind": r["kind"], "movie_ids": [r["movie_id"]],
                               "purpose": KIND_DEFAULT_PURPOSE.get(r["kind"], "gallery")})
                counts["movie_images"] += 1
            except (ValueError, FileNotFoundError):
                counts["skipped"] += 1

        actor_manifest = ROOT / "data" / "actors" / "manifest.csv"
        if actor_manifest.exists():
            import csv
            with actor_manifest.open("r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    p = ROOT / row.get("path", "")
                    if not p.exists():
                        counts["skipped"] += 1
                        continue
                    try:
                        self.add(p, asset_type="image", source="local",
                                 tags={"media_kind": "actor_life", "label": row.get("actor", ""),
                                       "split": row.get("split", ""), "purpose": "reference"})
                        counts["actor_photos"] += 1
                    except (ValueError, FileNotFoundError):
                        counts["skipped"] += 1
        return counts
