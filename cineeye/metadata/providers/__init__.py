"""可插拔的电影元数据提供者：本地表格 / TMDB / 豆瓣。

用途：把三种数据源（本地 CSV / TMDB API / 豆瓣 HTML 抓取）统一到
    BaseProvider 接口，导入脚本按名称选择来源。
被谁使用：导入脚本、运维 CLI。
依赖：本目录下 base / local_table / tmdb / douban 四个模块。
运维要点（改这里之前必须知道的事）：
    1. 三种来源的取舍：local 完全离线是默认兜底；tmdb 字段质量最好但
       需要 API key；douban 中文别名最全但依赖页面结构、易受反爬影响；
    2. 所有 provider 失败时都安全返回 None/空列表（降级不阻塞主流程），
       因此可以多来源串联导入——先 local 建骨架，再 tmdb/douban 补全；
    3. 新增来源：继承 BaseProvider 后在 build_provider 注册一个分支。
"""

from .base import BaseProvider, MovieRecord
from .local_table import LocalTableProvider
from .tmdb import TMDBProvider
from .douban import DoubanProvider

__all__ = [
    "BaseProvider", "MovieRecord",
    "LocalTableProvider", "TMDBProvider", "DoubanProvider",
    "build_provider",
]


def build_provider(name: str, cfg: dict) -> BaseProvider:
    """按名称构建 provider（简单工厂）。

    Args:
        name: 提供者名称，大小写不敏感；local/local_table/csv/json 均指向
            本地表格实现。
        cfg: 全局配置 dict，原样传给具体实现。

    Returns:
        对应的 BaseProvider 实例。

    Raises:
        ValueError: 名称不在支持列表中。
    """
    name = name.lower()
    if name in ("local", "local_table", "csv", "json"):
        return LocalTableProvider(cfg)
    if name == "tmdb":
        return TMDBProvider(cfg)
    if name == "douban":
        return DoubanProvider(cfg)
    raise ValueError(f"未知元数据提供者：{name}，可选 local/tmdb/douban")
