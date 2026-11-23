"""本地表格提供者：CSV / JSON，零依赖、完全离线。

用途：从运维手工维护的本地表格导入电影档案，是三个 provider 中的
    默认与兜底来源——不联网、无 API key、字段完全可控，适合演示环境
    或内网部署。
被谁使用：providers/__init__.py 的 build_provider（名称 local/local_table/csv/json）。
依赖：仅标准库 csv / json。
运维要点：
    1. 表头大小写不敏感（统一转小写处理）；CSV 需 UTF-8（带 BOM 也可）；
    2. aliases / cast 两列是 JSON 字符串，写错 JSON 会被忽略并告警（不中断）；
    3. movie_id 缺失时按行号自动生成 movie_NN，若后续行顺序变化会导致
       id 不稳定——正式数据务必显式提供 movie_id。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from ...config import ROOT, get_value
from ...utils import get_logger
from .base import BaseProvider, MovieRecord

log = get_logger("cineeye.metadata.local")


def _json_field(value: Any, default: Any) -> Any:
    """把 CSV 单元格里的 JSON 字符串解析为对象，失败时返回 default 并告警。

    取舍：aliases/cast 塞在单个单元格里用 JSON 编码，是为了保持表格
    「一行一部电影」的简单结构；解析失败降级为空列表，不让一行脏数据
    拖垮整批导入。

    Args:
        value: 原始单元格值（字符串/None/已是 list 或 dict）。
        default: 解析失败或为空时的返回值（通常为 []）。

    Returns:
        解析后的对象，或 default。
    """
    if value in (None, ""):
        return default
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(str(value))
    except (TypeError, ValueError):
        log.warning("字段不是合法 JSON，已忽略：%s", str(value)[:80])
        return default


class LocalTableProvider(BaseProvider):
    """本地 CSV / JSON 表格提供者。

    Args:
        cfg: 全局配置，读取 metadata.providers.local_table 路径。
        path: 显式指定表格路径；优先级高于 cfg。
    """

    name = "local"

    def __init__(self, cfg: dict | None = None, path: str | Path | None = None):
        if path is None:
            raw = get_value(cfg or {}, "metadata.providers.local_table", "outputs/metadata/movies.csv")
            path = Path(raw)
            if not path.is_absolute():
                path = ROOT / path  # 相对路径一律相对项目根目录
        self.path = Path(path)

    def fetch_all(self) -> list[MovieRecord]:
        """读取整个表格并转为 MovieRecord 列表。

        文件不存在时告警并返回空列表（降级不报错），支持 .json 与
        其他后缀（按 CSV 处理）。

        Returns:
            MovieRecord 列表；表格为空或缺失时为空列表。

        Raises:
            json.JSONDecodeError / csv.Error: 文件内容严重损坏时向上抛出
            （这是唯一的例外情况——文件存在但整体不可解析说明是运维失误，
            应当显式暴露而不是静默吞掉）。
        """
        if not self.path.exists():
            log.warning("本地表格不存在：%s（可跳过）", self.path)
            return []
        if self.path.suffix.lower() == ".json":
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            rows = raw if isinstance(raw, list) else raw.get("movies", [])
        else:
            # utf-8-sig 兼容 Windows Excel 导出的带 BOM CSV
            with self.path.open("r", encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
        records: list[MovieRecord] = []
        for i, row in enumerate(rows):
            # 表头统一去空格转小写，容忍人工维护时的大小写/空格不一致
            row = {(k or "").strip().lower(): v for k, v in row.items()}
            # movie_id 缺失时按行号兜底生成；正式数据应显式提供以保证 id 稳定
            mid = str(row.get("movie_id") or f"movie_{i+1:02d}")
            title = str(row.get("primary_title") or row.get("title") or mid)
            records.append(MovieRecord(
                movie_id=mid,
                primary_title=title,
                original_title=row.get("original_title") or None,
                year=self._as_int(row.get("year")),
                director=row.get("director") or None,
                synopsis=row.get("synopsis") or None,
                poster_path=row.get("poster_path") or None,
                release_date=row.get("release_date") or None,
                aliases=_json_field(row.get("aliases"), []),
                cast=_json_field(row.get("cast"), []),
                external_id=row.get("external_id") or None,
                provider=self.name,
            ))
        log.info("本地表格读取 %d 部电影：%s", len(records), self.path.name)
        return records

    def fetch_by_title(self, title: str) -> MovieRecord | None:
        """按片名/别名包含关系在本地表格中查找第一部匹配的电影。

        Args:
            title: 查询片名。

        Returns:
            首个匹配的 MovieRecord；无匹配返回 None。
        """
        for rec in self.fetch_all():
            names = {rec.primary_title, rec.original_title or ""}
            names |= {a.get("alias", "") for a in rec.aliases}
            # 双向包含：容忍查询词带/不带前缀修饰的情况
            if any(title in n or n in title for n in names if n):
                return rec
        return None
