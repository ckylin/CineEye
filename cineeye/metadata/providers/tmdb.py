"""TMDB 提供者：多语言片名、演员、海报、简介。

用途：从 TMDB 官方 API 拉取高质量电影档案（多语言别名、完整演职员表、
    海报 URL），是字段质量最好的在线来源。
被谁使用：providers/__init__.py 的 build_provider（名称 tmdb）、导入脚本。
依赖：仅标准库 urllib，零新增依赖。
运维要点（改这里之前必须知道的事）：
    1. API key 优先级：配置 metadata.providers.tmdb.api_key >
       环境变量 TMDB_API_KEY；两者都没有时 available=False，
       所有方法安全返回空，不阻塞主流程（失败降级纪律）；
    2. 已知小问题：timeout 复用了 metadata.providers.douban.timeout
       配置项（键名疑似笔误，见 __init__），但默认值一致所以行为正常；
    3. fetch_all 恒返回空列表——TMDB 无法"列出全部"，只能按标题逐个查，
       全量导入请配合本地表格给片名清单。
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from typing import Any

from ...config import get_value
from ...utils import get_logger
from .base import BaseProvider, MovieRecord

log = get_logger("cineeye.metadata.tmdb")


class TMDBProvider(BaseProvider):
    """TMDB API 提供者（v3 REST）。

    Args:
        cfg: 全局配置，读取 metadata.providers.tmdb.* 各项。
    """

    name = "tmdb"

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        # key 双通道：配置文件优先，其次环境变量；都没有则整个 provider 静默降级
        self.api_key = str(get_value(cfg, "metadata.providers.tmdb.api_key", "") or os.environ.get("TMDB_API_KEY", ""))
        self.language = str(get_value(cfg, "metadata.providers.tmdb.language", "zh-CN,en-US"))
        self.region = str(get_value(cfg, "metadata.providers.tmdb.region", "CN"))
        self.base = str(get_value(cfg, "metadata.providers.tmdb.base_url", "https://api.themoviedb.org/3")).rstrip("/")
        # 优先读 tmdb 自己的 timeout；为兼容老配置文件，回退到 douban.timeout，最后 15 秒
        self.timeout = int(get_value(
            cfg, "metadata.providers.tmdb.timeout",
            get_value(cfg, "metadata.providers.douban.timeout", 15)))
        self.available = bool(self.api_key)

    def _get(self, endpoint: str, params: dict[str, Any] | None = None) -> dict | None:
        """发起一次 GET 请求并解析 JSON。

        失败降级：未配置 key、网络异常、非 200 响应一律返回 None，
        由调用方按"查不到"处理，绝不向上抛异常阻塞导入流程。

        Args:
            endpoint: API 路径，如 /search/movie。
            params: 额外查询参数。

        Returns:
            解析后的 JSON dict；失败返回 None。
        """
        if not self.available:
            return None
        p = {"api_key": self.api_key, "language": self.language, "region": self.region}
        p.update(params or {})
        url = f"{self.base}{endpoint}?{urllib.parse.urlencode(p)}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "CineEye/1.0"})
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            log.warning("TMDB 请求失败 %s：%s", endpoint, exc)
            return None

    def fetch_by_title(self, title: str) -> MovieRecord | None:
        """按片名搜索并取第一部结果的完整详情。

        Args:
            title: 查询片名。

        Returns:
            MovieRecord（含演职员表与别名）；未配置 key 或查不到返回 None。
        """
        if not self.available:
            log.info("未配置 TMDB_API_KEY，跳过 TMDB 提供者")
            return None
        found = self._get("/search/movie", {"query": title, "page": 1})
        results = (found or {}).get("results") or []
        if not results:
            return None
        mid = str(results[0]["id"])
        # 详情接口拉全量字段；失败时退回用搜索结果本身组装的简化档案
        return self.fetch_by_id(mid) or self._record(results[0], mid)

    def fetch_by_id(self, movie_id: str) -> MovieRecord | None:
        """按 TMDB id 拉取电影详情（含 credits 演职员表与多语言别名）。

        Args:
            movie_id: TMDB 数字 id。

        Returns:
            完整 MovieRecord；请求失败返回 None。
        """
        if not self.available:
            return None
        detail = self._get(f"/movie/{movie_id}", {"append_to_response": "credits,release_dates"})
        if not detail:
            return None
        rec = self._record(detail, str(movie_id))
        # credits：导演与演员
        credits = detail.get("credits") or {}
        for person in credits.get("crew") or []:
            if person.get("job") == "Director" and not rec.director:  # 只取第一位导演
                rec.director = person.get("name")
        for i, person in enumerate(credits.get("cast") or []):
            rec.cast.append({
                "name": person.get("name"),
                "name_en": person.get("original_name"),
                "character": person.get("character"),
                "role": "actor",
                "tmdb_id": str(person.get("id")) if person.get("id") else None,
                "billing_order": i,  # TMDB 返回顺序即番位
            })
        # 多语言别名：中文/英文/原名
        titles = detail.get("alternative_titles", {}).get("titles", []) if isinstance(detail.get("alternative_titles"), dict) else []
        seen = {a["alias"] for a in rec.aliases}
        for t in titles:
            alias = t.get("title")
            if alias and alias not in seen:  # 各地区译名去重后并入别名表
                rec.aliases.append({"alias": alias, "lang": t.get("iso_3166_1", ""), "region": t.get("iso_3166_1"), "source": "tmdb"})
                seen.add(alias)
        return rec

    def fetch_all(self) -> list[MovieRecord]:
        """TMDB 没有「列出我所有电影」的接口，只能按标题逐个查。

        Returns:
            恒为空列表（设计如此，不是故障）。
        """
        return []

    @staticmethod
    def _record(data: dict, mid: str) -> MovieRecord:
        """把 TMDB 原始响应（搜索结果或详情）组装成 MovieRecord。

        Args:
            data: TMDB 返回的原始 JSON dict。
            mid: TMDB 电影 id。

        Returns:
            基础档案（不含 cast/alternative_titles，由 fetch_by_id 补全）。
        """
        aliases: list[dict] = []
        for key, lang in (("title", "zh"), ("original_title", "orig")):
            v = data.get(key)
            if v and all(a["alias"] != v for a in aliases):
                aliases.append({"alias": v, "lang": lang, "region": None, "source": "tmdb"})
        poster = data.get("poster_path")
        return MovieRecord(
            movie_id=mid,
            primary_title=str(data.get("title") or data.get("name") or mid),
            original_title=data.get("original_title"),
            year=BaseProvider._as_int(data.get("release_date") or data.get("first_air_date")),
            synopsis=data.get("overview"),
            # TMDB 返回的是相对路径，这里拼成可直接下载的 w500 海报 URL
            poster_path=f"https://image.tmdb.org/t/p/w500{poster}" if poster else None,
            release_date=data.get("release_date"),
            aliases=aliases,
            external_id=mid,
            provider="tmdb",
        )
