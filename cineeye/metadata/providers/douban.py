"""豆瓣提供者：中文片名、别名、演员、海报（HTML 抓取，零依赖）。

用途：从豆瓣抓取中文最丰富的电影档案——大陆译名/港台译名/又名列表
    是其他来源没有的，对「国外电影多个名字」的检索召回价值最大。
被谁使用：providers/__init__.py 的 build_provider（名称 douban）、导入脚本。
依赖：仅标准库 urllib / re / html，零新增依赖。
运维要点（改这里之前必须知道的事）：
    1. 豆瓣没有公开 API，本提供者解析搜索页与条目页的 HTML——豆瓣改版
       （class 名/结构变化）会直接导致字段抓不到，返回 None 不阻塞主流程；
    2. 请求间隔默认 1.5 秒（metadata.providers.douban.request_interval），
       既避免给对方造成压力，也降低被封 IP 的概率；抓取失败优先调大间隔；
    3. 遇到验证码页/登录墙时搜索正则匹配不到 sid，表现为"查不到"而非报错；
    4. 可用 metadata.providers.douban.enabled=false 一键停用此来源。
"""

from __future__ import annotations

import html
import json
import re
import time
import urllib.parse
import urllib.request
from typing import Any

from ...config import get_value
from ...utils import get_logger
from .base import BaseProvider, MovieRecord

log = get_logger("cineeye.metadata.douban")

_UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36"}


class DoubanProvider(BaseProvider):
    """豆瓣 HTML 抓取提供者。

    Args:
        cfg: 全局配置，读取 metadata.providers.douban.* 各项。
    """

    name = "douban"

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.enabled = bool(get_value(cfg, "metadata.providers.douban.enabled", True))
        self.interval = float(get_value(cfg, "metadata.providers.douban.request_interval", 1.5))
        self.timeout = int(get_value(cfg, "metadata.providers.douban.timeout", 12))
        self._last = 0.0  # 上次请求时间戳，用于实现最小间隔限速

    def _get(self, url: str) -> str | None:
        """带限速的 GET 请求，返回页面 HTML 文本。

        限速：两次请求之间强制至少间隔 self.interval 秒——这是对目标站点的
        礼貌，也是防封号的自我保护，不要随意调小。

        Args:
            url: 完整 URL。

        Returns:
            HTML 文本；被停用或请求失败返回 None（降级不抛异常）。
        """
        if not self.enabled:
            return None
        wait = self.interval - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)  # 强制最小请求间隔，避免触发反爬
        self._last = time.time()
        try:
            req = urllib.request.Request(url, headers=_UA)
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001
            log.warning("豆瓣请求失败：%s (%s)", url.split("?")[-1][:60], exc)
            return None

    def fetch_by_title(self, title: str) -> MovieRecord | None:
        """先搜索拿 subject id，再抓条目页详情。

        Args:
            title: 查询片名。

        Returns:
            MovieRecord；搜索无结果或页面解析失败返回 None。
        """
        url = "https://www.douban.com/search?" + urllib.parse.urlencode({"q": title, "cat": "1002"})
        page = self._get(url)
        if not page:
            return None
        # 搜索结果里的 sid= 即为电影 id（两种页面结构都要试，豆瓣改版兜底）
        m = re.search(r"sid:\s*(\d+)", page) or re.search(r"/subject/(\d+)/", page)
        if not m:
            return None
        return self.fetch_by_id(m.group(1))

    def fetch_by_id(self, movie_id: str) -> MovieRecord | None:
        """抓取条目页并按 HTML 结构解析出档案字段。

        解析完全依赖豆瓣页面当前的 HTML 结构（microdata 属性 v:itemprop /
        v:originaltitle 等 + class 名正则），豆瓣改版时此处最先受影响。

        Args:
            movie_id: 豆瓣 subject 数字 id。

        Returns:
            MovieRecord（含又名列表与演员表）；页面不可用返回 None。
        """
        page = self._get(f"https://movie.douban.com/subject/{movie_id}/")
        if not page:
            return None

        def meta(name: str) -> str | None:
            # 取 microdata 属性 <span property="v:XXX"> 的文本，去掉内嵌标签
            m = re.search(rf'<span property="v:{name}"[^>]*>(.*?)</span>', page, re.S)
            return html.unescape(re.sub(r"<[^>]+>", "", m.group(1)).strip()) if m else None

        title = meta("itemprop") or self._title(page)
        if not title:
            return None
        original = meta("original_title")
        year = BaseProvider._as_int(meta("initialReleaseDate") or self._year(page))
        poster_m = re.search(r'<img src="(https://img\d+\.doubanio\.com/[^"]+)"', page)
        aliases: list[dict[str, Any]] = []
        # 「又名」行按 / 、 ， 切成多个别名，是中文译名的最大来源
        for alt in re.findall(r'<span class="pl">又名:</span>(.*?)<br', page, re.S):
            for part in re.split(r"/|、|,|，", html.unescape(re.sub(r"<[^>]+>", "", alt))):
                part = part.strip()
                if part and all(a["alias"] != part for a in aliases):  # 逐条去重
                    aliases.append({"alias": part, "lang": "zh", "region": "CN", "source": "douban"})
        if original and all(a["alias"] != original for a in aliases):
            aliases.append({"alias": original, "lang": "orig", "region": None, "source": "douban"})
        cast: list[dict[str, Any]] = []
        # 只取前 20 位主演，避免超长演职员表拖慢导入
        for name in re.findall(r'<span class="actor">\s*<a[^>]*>([^<]+)</a>', page)[:20]:
            cast.append({"name": html.unescape(name.strip()), "role": "actor", "billing_order": len(cast)})
        director = None
        dm = re.search(r'<span class="pl">导演:</span>\s*<a[^>]*>([^<]+)</a>', page)
        if dm:
            director = html.unescape(dm.group(1).strip())
        return MovieRecord(
            movie_id=movie_id,
            primary_title=title,
            original_title=original,
            year=year,
            director=director,
            poster_path=poster_m.group(1) if poster_m else None,
            aliases=aliases,
            cast=cast,
            external_id=movie_id,
            provider="douban",
        )

    def fetch_all(self) -> list[MovieRecord]:
        """豆瓣无批量列举接口。

        Returns:
            恒为空列表（设计如此）。
        """
        return []

    @staticmethod
    def _title(page: str) -> str | None:
        """从条目页提取「中文片名 (原片名)」格式的完整标题。

        Args:
            page: 条目页 HTML。

        Returns:
            标题文本；匹配失败返回 None。
        """
        m = re.search(r'<span property="v:itemreviewed">(.*?)</span>', page, re.S)
        return html.unescape(re.sub(r"<[^>]+>", "", m.group(1)).strip()) if m else None

    @staticmethod
    def _year(page: str) -> int | None:
        """从条目页提取上映年份（class="year" 的 4 位数字）。

        Args:
            page: 条目页 HTML。

        Returns:
            年份 int；匹配失败返回 None。
        """
        m = re.search(r'<span class="year">\((\d{4})\)</span>', page)
        return int(m.group(1)) if m else None
