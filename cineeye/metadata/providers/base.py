"""元数据提供者统一接口。

用途：定义 MovieRecord（标准化电影档案）与 BaseProvider（provider 抽象），
    让本地 CSV / TMDB API / 豆瓣抓取三种数据源产出统一结构后统一入库。
被谁使用：providers/ 下三个具体实现继承本接口；导入脚本经 build_provider
    分发到具体实现。
依赖：仅标准库 dataclasses。
运维要点：新增数据源时继承 BaseProvider、实现 fetch_all / fetch_by_title，
    并在 providers/__init__.py 的 build_provider 注册；MovieRecord 的字段
    结构与 store.upsert_movie 的参数一一对应，改任一侧都要同步另一侧。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MovieRecord:
    """标准化电影档案，三个 provider 都返回这个结构。

    上层拿它直接喂给 MetaStore.upsert_movie（经 to_kwargs 展开），
    因此字段名与 upsert_movie 的关键字参数保持一致。

    Attributes:
        movie_id: 业务主键（外部源用外部 id，本地 CSV 用 movie_XX）。
        primary_title: 主片名。
        original_title: 原始语言片名。
        year: 年份。
        director: 导演。
        synopsis: 简介。
        poster_path: 海报路径或 URL。
        release_date: 上映日期。
        aliases: 别名列表 [{"alias","lang","region","source"}]。
        cast: 演员列表 [{"name","name_en","character","role",...}]。
        external_id: 来源系统中的 id。
        provider: 来源标识（local/tmdb/douban）。
    """

    movie_id: str
    primary_title: str
    original_title: str | None = None
    year: int | None = None
    director: str | None = None
    synopsis: str | None = None
    poster_path: str | None = None
    release_date: str | None = None
    aliases: list[dict[str, Any]] = field(default_factory=list)
    cast: list[dict[str, Any]] = field(default_factory=list)
    external_id: str | None = None
    provider: str = "unknown"

    def to_kwargs(self) -> dict[str, Any]:
        """展开为 upsert_movie 可直接消费的关键字参数 dict。

        注意：movie_id 不在其中（调用方单独传），保持与 upsert_movie
        的第一个位置参数语义一致。

        Returns:
            与 MetaStore.upsert_movie 关键字参数同名的 dict。
        """
        return {
            "primary_title": self.primary_title,
            "original_title": self.original_title,
            "year": self.year,
            "director": self.director,
            "synopsis": self.synopsis,
            "poster_path": self.poster_path,
            "release_date": self.release_date,
            "aliases": self.aliases,
            "cast": self.cast,
            "provider": self.provider,
            "external_id": self.external_id,
        }


class BaseProvider:
    """所有 provider 的抽象接口。

    约定：三个具体实现（本地 CSV / TMDB API / 豆瓣抓取）的取舍：
    - local_table：零依赖、完全离线、字段可控，是默认与兜底来源；
    - tmdb：字段质量最好（多语言/演职员表），但需要 API key 且依赖网络；
    - douban：中文别名最丰富，但无公开 API，靠 HTML 抓取，易受反爬影响。
    统一要求：任何失败（无 key、断网、解析失败）都必须安全返回 None / 空列表，
    不允许向上抛异常阻塞导入主流程——这是失败降级的基本纪律。
    """

    name = "base"

    def fetch_all(self) -> list[MovieRecord]:
        """批量拉取全部电影档案；不支持批量列举的来源返回空列表。"""
        raise NotImplementedError

    def fetch_by_title(self, title: str) -> MovieRecord | None:
        """按片名精确/模糊查询单部电影；查不到返回 None（不抛异常）。"""
        raise NotImplementedError

    @staticmethod
    def _as_int(value: Any) -> int | None:
        """宽容地把各种年份写法转 int，失败返回 None。

        只取前 4 个字符，兼容 '2024-05-01' / 2024 / '约2024' 等脏数据。

        Args:
            value: 任意来源的原始字段值。

        Returns:
            年份 int；无法解析时返回 None。
        """
        if value is None or value == "":
            return None
        try:
            return int(str(value)[:4])
        except (TypeError, ValueError):
            return None
