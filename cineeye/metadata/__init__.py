"""电影元数据库：多片名、演员、图片索引与三个可插拔数据提供者。

用途：对外只导出 MetaStore——上层（identify.py、导入脚本、serve.py）
    统一从 cineeye.metadata import MetaStore，不感知内部实现。
被谁使用：identify.py / gallery 相关脚本 / 运维 CLI。
依赖：store.py（SQLite）与 providers/ 下的数据提供者。
运维要点：MetaStore 是全项目元数据读写唯一入口；若要换存储后端
    （如 PostgreSQL），只需保持本包对外接口不变。
"""

from .store import MetaStore

__all__ = ["MetaStore"]
