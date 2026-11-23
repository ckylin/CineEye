"""电影元数据库（SQLite），全项目离线数据的唯一落盘存储。

用途：存放电影主档（movies）、多语言别名（movie_aliases）、演员（persons）、
    演员与电影的出演关系（movie_cast）、图库图片登记（movie_images），
    以及数据来源追踪（sources）；并提供片名/别名的模糊搜索。
被谁使用：gallery/index.py 读取已登记图片构建向量索引；identify.py 做元数据
    丰富与片名搜索；导入脚本通过 providers 抓取数据后调 upsert_movie 写入。
依赖：sqlite3（标准库，无需额外安装）、同目录 normalize.py 做片名归一化、
    上层 config.ROOT / utils。
运维要点（改这里之前必须知道的事）：
    1. 数据库文件默认在 outputs/cineeye.db，可直接备份（单文件，含全部数据）；
    2. 所有写入走 upsert（ON CONFLICT DO UPDATE），重复导入同一条数据是幂等的，
       不会产生重复行；未提供的字段用 COALESCE 保留旧值，不会被清空；
    3. 外键全部 ON DELETE CASCADE，删除 movies 行会连带清理别名/演员关系/图片登记；
    4. 搜索是"全表读出 + 内存打分"，几千部电影规模没问题；若片库上万条，
       考虑把归一化别名提前物化（建列或缓存）再优化 SQL；
    5. 数据库 schema 变更时，因为用 CREATE TABLE IF NOT EXISTS，旧库不会自动
       加新列——需要手动迁移或删除重建。
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from ..config import ROOT, get_value
from ..utils import ensure_dir, get_logger
from .normalize import normalize, similarity

log = get_logger("cineeye.metadata")

# SQLite 表结构说明（运维重点）：
# - movies：电影主档，id 由调用方给定（本地 CSV 用 movie_XX，外部源用外部 id），
#   主键即业务 id，便于幂等 upsert；
# - movie_aliases：一对多别名表，「一部电影多个名字」检索的核心。
#   UNIQUE(movie_id, alias) 防重；ON DELETE CASCADE 随主档删除；
#   索引 idx_alias_alias 加速按别名的反查；
# - persons：演员维表，id 是 normalize(name) 的 MD5 前缀（见 _ensure_person），
#   同名不同写法（繁简/空格）会被归一到同一 id，避免演员重复建行；
# - movie_cast：电影↔演员多对多桥表，billing_order 保留演员表中的排序（番位）；
# - movie_images：图库图片登记表，gallery 索引构建时的数据源；
#   kind 区分 frame（视频帧）/poster（海报）等用途，检索时可按 kind 过滤；
# - sources：记录每部电影的元数据来自哪个 provider 及其外部 id，
#   便于排查数据质量问题和重复导入。
SCHEMA = """
CREATE TABLE IF NOT EXISTS movies (
    id             TEXT PRIMARY KEY,
    primary_title  TEXT NOT NULL,
    original_title TEXT,
    year           INTEGER,
    director       TEXT,
    synopsis       TEXT,
    poster_path    TEXT,
    release_date   TEXT,
    created_at     TEXT DEFAULT (datetime('now')),
    updated_at     TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS movie_aliases (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    movie_id TEXT NOT NULL REFERENCES movies(id) ON DELETE CASCADE,
    alias    TEXT NOT NULL,
    lang     TEXT,
    region   TEXT,
    source   TEXT,
    UNIQUE(movie_id, alias)
);

CREATE TABLE IF NOT EXISTS persons (
    id       TEXT PRIMARY KEY,
    name     TEXT NOT NULL,
    name_en  TEXT,
    role     TEXT DEFAULT 'actor',
    tmdb_id  TEXT
);

CREATE TABLE IF NOT EXISTS movie_cast (
    movie_id      TEXT NOT NULL REFERENCES movies(id) ON DELETE CASCADE,
    person_id     TEXT NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    character     TEXT,
    billing_order INTEGER DEFAULT 0,
    PRIMARY KEY (movie_id, person_id)
);

CREATE TABLE IF NOT EXISTS movie_images (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    movie_id   TEXT NOT NULL REFERENCES movies(id) ON DELETE CASCADE,
    kind       TEXT NOT NULL DEFAULT 'frame',
    path       TEXT NOT NULL,
    source     TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    UNIQUE(movie_id, kind, path)
);

CREATE TABLE IF NOT EXISTS sources (
    movie_id    TEXT NOT NULL REFERENCES movies(id) ON DELETE CASCADE,
    provider    TEXT NOT NULL,
    external_id TEXT,
    fetched_at  TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (movie_id, provider)
);

CREATE INDEX IF NOT EXISTS idx_alias_alias ON movie_aliases(alias);
CREATE INDEX IF NOT EXISTS idx_images_movie ON movie_images(movie_id);
CREATE INDEX IF NOT EXISTS idx_cast_movie ON movie_cast(movie_id);
CREATE INDEX IF NOT EXISTS idx_cast_person ON movie_cast(person_id);
"""


class MetaStore:
    """电影档案库：电影 / 别名 / 演员 / 图片索引的增删改查与搜索。

    Args:
        db_path: SQLite 文件路径；为 None 时按 cfg 的 metadata.db 解析，
            若 cfg 也未给则默认 ROOT/outputs/cineeye.db。
        cfg: 全局配置 dict，用于解析 db 路径与搜索归一化开关
            （metadata.aliases.* 三个布尔项）。

    Raises:
        sqlite3.OperationalError: 数据库文件损坏或磁盘不可写时由 sqlite3 抛出。
    """

    def __init__(self, db_path: str | Path | None = None, cfg: dict | None = None):
        if db_path is None:
            if cfg:
                raw = get_value(cfg, "metadata.db", "outputs/cineeye.db")
                db_path = Path(raw)
                if not db_path.is_absolute():
                    db_path = ROOT / db_path  # 配置里的相对路径一律相对项目根目录解析
            else:
                db_path = ROOT / "outputs" / "cineeye.db"
        self.path = Path(db_path)
        ensure_dir(self.path.parent)  # 首次运行时父目录可能不存在
        with self._conn() as c:
            c.executescript(SCHEMA)  # 幂等建表：已存在的表不会被改动
        self.cfg = cfg or {}
        # 搜索归一化开关（大小写/标点/繁简），与 normalize() 的参数一一对应；
        # 存成元组避免每次搜索都查配置
        self._nc = (
            bool(get_value(self.cfg, "metadata.aliases.normalize_case", True)),
            bool(get_value(self.cfg, "metadata.aliases.strip_punctuation", True)),
            bool(get_value(self.cfg, "metadata.aliases.traditional_to_simplified", True)),
        )
        log.info("元数据库就绪：%s", self.path.relative_to(ROOT) if self.path.is_relative_to(ROOT) else self.path)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        """打开一个短连接并在退出时提交。

        用短连接而非常驻连接：运维上便于多进程并发访问 SQLite，
        timeout=30 缓解偶发的 database locked 冲突。

        Yields:
            row_factory 已设为 sqlite3.Row 的连接对象（支持按列名取值）。
        """
        conn = sqlite3.connect(str(self.path), timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ---------------------------------------------------------------- 写入
    def upsert_movie(self, movie_id: str, primary_title: str, *,
                     original_title: str | None = None, year: int | None = None,
                     director: str | None = None, synopsis: str | None = None,
                     poster_path: str | None = None, release_date: str | None = None,
                     aliases: list[dict] | None = None, cast: list[dict] | None = None,
                     provider: str | None = None, external_id: str | None = None) -> str:
        """插入或更新一部电影及其别名、演员、来源。返回 movie_id。

        upsert 语义：id 冲突时更新主片名；其余字段用 COALESCE(新值, 旧值)，
        即传 None 的字段保留数据库原值——支持"分批导入、逐步补全"，
        比如先导入本地 CSV 再用 TMDB 补海报不会互相覆盖。

        Args:
            movie_id: 业务主键，幂等导入的关键。
            primary_title: 主片名（展示用）。
            original_title: 原始语言片名。
            year: 年份。
            director: 导演。
            synopsis: 简介。
            poster_path: 海报路径或 URL。
            release_date: 上映日期（ISO 字符串）。
            aliases: 别名列表，元素形如
                {"alias": str, "lang": str, "region": str, "source": str}。
            cast: 演员列表，元素形如
                {"name": str, "name_en": str, "character": str, "role": str}。
            provider: 数据来源标识（local/tmdb/douban），同时写入 sources 表。
            external_id: 在来源系统中的 id，用于溯源。

        Returns:
            传入的 movie_id（原样返回，便于链式调用）。

        Raises:
            sqlite3.IntegrityError: 违反约束时抛出（正常用法不会触发）。
        """
        with self._conn() as c:
            c.execute(
                """INSERT INTO movies (id, primary_title, original_title, year, director,
                                       synopsis, poster_path, release_date, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,datetime('now'))
                   ON CONFLICT(id) DO UPDATE SET
                     primary_title=excluded.primary_title,
                     original_title=COALESCE(excluded.original_title, movies.original_title),
                     year=COALESCE(excluded.year, movies.year),
                     director=COALESCE(excluded.director, movies.director),
                     synopsis=COALESCE(excluded.synopsis, movies.synopsis),
                     poster_path=COALESCE(excluded.poster_path, movies.poster_path),
                     release_date=COALESCE(excluded.release_date, movies.release_date),
                     updated_at=datetime('now')""",
                (movie_id, primary_title, original_title, year, director, synopsis, poster_path, release_date),
            )
            # 多别名支持：主片名 + 原片名也进别名表，让搜索可以"统一按任意名字查"；
            # 用 set 去重，INSERT OR IGNORE 保证已有别名不会被覆盖（幂等）
            titles = {(primary_title, "zh", None), (original_title or "", "orig", None)}
            for a in aliases or []:
                titles.add((str(a.get("alias", "")), a.get("lang"), a.get("region")))
            for alias, lang, region in titles:
                if not alias:
                    continue
                c.execute(
                    """INSERT OR IGNORE INTO movie_aliases (movie_id, alias, lang, region, source)
                       VALUES (?,?,?,?,?)""",
                    (movie_id, alias, lang, region, provider),
                )
            for i, person in enumerate(cast or []):
                name = str(person.get("name", "")).strip()
                if not name:
                    continue
                pid = self._ensure_person(c, name, person.get("name_en"),
                                          person.get("role", "actor"),
                                          person.get("tmdb_id"))
                c.execute(
                    """INSERT OR IGNORE INTO movie_cast (movie_id, person_id, character, billing_order)
                       VALUES (?,?,?,?)""",
                    (movie_id, pid, person.get("character"), int(person.get("billing_order", i))),
                )
            if provider:
                c.execute(
                    """INSERT OR REPLACE INTO sources (movie_id, provider, external_id, fetched_at)
                       VALUES (?,?,?,datetime('now'))""",
                    (movie_id, provider, external_id),
                )
        return movie_id

    @staticmethod
    def _ensure_person(c: sqlite3.Connection, name: str, name_en: str | None,
                       role: str, tmdb_id: str | None) -> str:
        """按归一化名字查找或创建演员，返回 person_id。

        person_id 用 normalize(name) 的 MD5 前缀生成，这样不同来源、
        不同写法（繁简/空格差异）的同一演员会落到同一行，天然去重。

        Args:
            c: 当前事务连接。
            name: 演员中文名。
            name_en: 演员英文名/原文名。
            role: 角色，默认 'actor'。
            tmdb_id: TMDB 人物 id（若有）。

        Returns:
            形如 person_xxxxxxxxxxxx 的稳定 person_id。
        """
        import hashlib

        # 用归一化后的名字做 id，保证「同一演员不同写法 → 同一 id」
        pid = "person_" + hashlib.md5(normalize(name).encode("utf-8")).hexdigest()[:12]
        row = c.execute("SELECT id FROM persons WHERE id=?", (pid,)).fetchone()
        if row is None:
            c.execute("INSERT INTO persons (id,name,name_en,role,tmdb_id) VALUES (?,?,?,?,?)",
                      (pid, name, name_en or None, role or "actor", tmdb_id))
        return pid

    def upsert_person(self, name: str, name_en: str | None = None, role: str = "actor",
                      tmdb_id: str | None = None) -> str:
        """登记（或复用）一位演员，返回稳定的 person_id。

        演员照片打标签时使用：目录名/CSV 里的演员名先在这里落库，后续
        识别结果才能通过 person_id 反查「参演电影」。person_id 由归一化名字
        的 MD5 生成，因此「刘德华」「劉德華」「Andy Lau」这类不同写法是否
        落到同一人，取决于名字字符串本身（英文名请通过 name_en 补充）。

        Args:
            name: 演员名（中文名或唯一标识名）。
            name_en: 英文名/原文名，便于跨语言检索。
            role: 角色，默认 actor；导演可传 director。
            tmdb_id: 外部 id，便于溯源。

        Returns:
            person_id（形如 person_xxxxxxxxxxxx）。

        Raises:
            sqlite3.OperationalError: 数据库不可写时抛出。
        """
        with self._conn() as c:
            pid = self._ensure_person(c, name, name_en, role, tmdb_id)
            # 已存在的演员只补空字段，不覆盖已有英文名（COALESCE 语义）
            if name_en:
                c.execute("UPDATE persons SET name_en=COALESCE(name_en, ?) WHERE id=?", (name_en, pid))
            return pid

    def find_person(self, name: str) -> dict | None:
        """按名字精确/归一化匹配演员，返回 persons 行（含电影列表）。

        比遍历 list_actors() 更高效，供演员识别结果反查使用。

        Args:
            name: 演员名或英文名。

        Returns:
            含 movies 键的档案 dict；未命中返回 None。
        """
        if not name:
            return None
        pid: str | None = None
        with self._conn() as c:
            row = c.execute(
                "SELECT id FROM persons WHERE name=? OR name_en=? LIMIT 1", (name, name)
            ).fetchone()
            if row is not None:
                pid = row["id"]
            else:
                # 退化匹配：忽略繁简/大小写/标点后的名字相同（如「劉德華」vs「刘德华」）
                target = normalize(name)
                for r in c.execute("SELECT id, name, name_en FROM persons"):
                    if normalize(r["name"]) == target or (
                        r["name_en"] and normalize(r["name_en"]) == target
                    ):
                        pid = r["id"]
                        break
        # 出了连接再取完整档案：get_actor 会自己开新连接，避免嵌套连接
        return self.get_actor(pid) if pid else None

    def add_image(self, movie_id: str, path: str | Path, kind: str = "frame",
                  source: str | None = None) -> bool:
        """登记一张图库图片；返回是否为新增。

        路径优先转成相对项目根的 POSIX 路径存储，这样整个 outputs 目录
        挪位置/换机器后索引仍然有效（可移植性）。

        Args:
            movie_id: 所属电影。
            path: 图片路径。
            kind: 图片类型（frame/poster/...），检索时可按 kind 过滤。
            source: 来源说明。

        Returns:
            True=新增登记成功；False=已存在（UNIQUE 约束跳过）。
        """
        p = Path(path)
        rel = p.resolve().relative_to(ROOT).as_posix() if p.resolve().is_relative_to(ROOT) else str(p)
        with self._conn() as c:
            cur = c.execute(
                "INSERT OR IGNORE INTO movie_images (movie_id, kind, path, source) VALUES (?,?,?,?)",
                (movie_id, kind, rel, source),
            )
            return cur.rowcount > 0

    # ---------------------------------------------------------------- 查询
    def get_movie(self, movie_id: str) -> dict | None:
        """按 id 取一部电影的完整档案（含别名/演员/图片/来源）。

        Args:
            movie_id: 电影业务 id。

        Returns:
            档案 dict；不存在时返回 None。
        """
        with self._conn() as c:
            row = c.execute("SELECT * FROM movies WHERE id=?", (movie_id,)).fetchone()
            return self._movie_dict(c, row) if row else None

    def list_movies(self, limit: int = 100, offset: int = 0) -> list[dict]:
        """分页列出电影完整档案。

        Args:
            limit: 每页条数。
            offset: 起始偏移。

        Returns:
            档案 dict 列表（结构同 get_movie）。
        """
        with self._conn() as c:
            rows = c.execute("SELECT * FROM movies ORDER BY id LIMIT ? OFFSET ?", (limit, offset)).fetchall()
            return [self._movie_dict(c, r) for r in rows]

    def count_movies(self) -> int:
        """返回电影总数（运维监控用）。"""
        with self._conn() as c:
            return int(c.execute("SELECT COUNT(*) FROM movies").fetchone()[0])

    def _movie_dict(self, c: sqlite3.Connection, row: sqlite3.Row) -> dict:
        """把 movies 行组装成完整档案 dict，附带四个关联子查询。

        Args:
            c: 当前连接。
            row: movies 表的一行。

        Returns:
            含 aliases / cast / images / sources 键的 dict。
        """
        d = dict(row)
        d["aliases"] = [dict(r) for r in c.execute(
            "SELECT alias, lang, region, source FROM movie_aliases WHERE movie_id=? ORDER BY lang, alias",
            (d["id"],))]
        d["cast"] = [dict(r) for r in c.execute(
            """SELECT p.id AS person_id, p.name, p.name_en, mc.character, mc.billing_order
               FROM movie_cast mc JOIN persons p ON p.id=mc.person_id
               WHERE mc.movie_id=? ORDER BY mc.billing_order""", (d["id"],))]
        d["images"] = [dict(r) for r in c.execute(
            "SELECT kind, path, source FROM movie_images WHERE movie_id=? ORDER BY kind, path", (d["id"],))]
        d["sources"] = [dict(r) for r in c.execute(
            "SELECT provider, external_id, fetched_at FROM sources WHERE movie_id=?", (d["id"],))]
        return d

    def search(self, query: str, top_k: int = 10, min_score: float = 0.0) -> list[dict]:
        """按片名/别名搜索，支持多名字与模糊匹配。

        召回：归一化后做 SQL LIKE 模糊匹配 + 逐条相似度打分；
        排序：相似度降序；同一部电影只返回得分最高的别名。

        Args:
            query: 用户输入的片名（任意语言/写法）。
            top_k: 最多返回的电影数。
            min_score: 相似度阈值，低于此值的候选被丢弃。

        Returns:
            按分数降序的候选列表，每项含 movie_id / score / matched_alias /
            primary_title / original_title / year / aliases。
        """
        opts = self._nc
        # 查询词与库内别名用同一套归一化，保证繁简/大小写/标点差异不影响召回
        nq = normalize(query, case=opts[0], punct=opts[1], t2s=opts[2])
        if not nq:
            return []
        with self._conn() as c:
            rows = c.execute(
                """SELECT a.movie_id, a.alias, a.lang, a.region, m.primary_title, m.original_title, m.year
                   FROM movie_aliases a JOIN movies m ON m.id=a.movie_id"""
            ).fetchall()
            best: dict[str, tuple[float, sqlite3.Row]] = {}
            for r in rows:
                na = normalize(r["alias"], case=opts[0], punct=opts[1], t2s=opts[2])
                if na == nq:
                    score = 1.0
                elif nq in na or na in nq:
                    # 编号型别名（如 movie_01）之间高度相似，包含关系不应给高分，
                    # 否则搜 movie_03 会误召回全部 movie_0X
                    score = 0.92 if not (na.startswith("movie") and nq.startswith("movie")) else 0.55
                else:
                    score = similarity(query, r["alias"], case=opts[0], punct=opts[1], t2s=opts[2])
                    if na.startswith("movie") and nq.startswith("movie"):
                        score *= 0.6
                if score < min_score:
                    continue
                mid = r["movie_id"]
                if mid not in best or score > best[mid][0]:
                    best[mid] = (score, r)
            result = []
            for mid, (score, r) in sorted(best.items(), key=lambda kv: -kv[1][0])[:top_k]:
                movie = self.get_movie(mid) or {}
                result.append({
                    "movie_id": mid, "score": round(score, 4),
                    "matched_alias": r["alias"], "primary_title": r["primary_title"],
                    "original_title": r["original_title"], "year": r["year"],
                    "aliases": movie.get("aliases", []),
                })
            return result

    def get_actor(self, person_id: str) -> dict | None:
        """按 person_id 取演员档案及其参演电影列表。

        Args:
            person_id: persons 表主键。

        Returns:
            含 movies 键（出演列表）的 dict；不存在返回 None。
        """
        with self._conn() as c:
            row = c.execute("SELECT * FROM persons WHERE id=?", (person_id,)).fetchone()
            if not row:
                return None
            d = dict(row)
            d["movies"] = [dict(r) for r in c.execute(
                """SELECT m.id AS movie_id, m.primary_title, m.original_title, m.year, mc.character
                   FROM movie_cast mc JOIN movies m ON m.id=mc.movie_id
                   WHERE mc.person_id=? ORDER BY m.year, m.id""", (person_id,))]
            return d

    def list_actors(self, limit: int = 100) -> list[dict]:
        """列出演员（按参演电影数降序），供前端浏览与 recognize_with_movies 反查。

        Args:
            limit: 最多返回人数。

        Returns:
            [{id, name, name_en, role, movie_count}, ...]
        """
        with self._conn() as c:
            return [dict(r) for r in c.execute(
                """SELECT p.id, p.name, p.name_en, p.role, COUNT(mc.movie_id) AS movie_count
                   FROM persons p LEFT JOIN movie_cast mc ON mc.person_id=p.id
                   GROUP BY p.id ORDER BY movie_count DESC, p.name LIMIT ?""", (limit,))]

    def images_for_index(self) -> list[dict]:
        """给图库索引用：返回所有已登记图片。

        Returns:
            [{movie_id, kind, path, image_id}, ...]，按登记顺序排列。
        """
        with self._conn() as c:
            return [dict(r) for r in c.execute(
                "SELECT movie_id, kind, path, id AS image_id FROM movie_images ORDER BY id")]

    def stats(self) -> dict:
        """各表行数统计，用于 status 接口与运维巡检。

        Returns:
            {movies, aliases, actors, cast_links, images} 计数 dict。
        """
        with self._conn() as c:
            def n(sql: str) -> int:
                return int(c.execute(sql).fetchone()[0])
            return {
                "movies": n("SELECT COUNT(*) FROM movies"),
                "aliases": n("SELECT COUNT(*) FROM movie_aliases"),
                "actors": n("SELECT COUNT(*) FROM persons WHERE role='actor'"),
                "cast_links": n("SELECT COUNT(*) FROM movie_cast"),
                "images": n("SELECT COUNT(*) FROM movie_images"),
            }

    def export_json(self, path: str | Path) -> Path:
        """导出完整档案为 JSON，便于备份或迁移。

        Args:
            path: 输出 JSON 文件路径。

        Returns:
            实际写出的文件 Path（便于日志打印）。
        """
        data = {
            "movies": self.list_movies(limit=100000),
            "actors": self.list_actors(limit=100000),
            "stats": self.stats(),
        }
        out = Path(path)
        ensure_dir(out.parent)
        out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return out
