"""图库向量索引：NumPy 余弦相似度（预留 faiss 开关）。

用途：把每部电影的图库图片编码成向量并落盘，提供开放集的
    「图片 → 相似电影」检索（不限于训练类别，加新片只需增量 add）。
被谁使用：identify.py（图库检索主通道）、构建索引的脚本、serve.py 的 status。
依赖：numpy + gallery/encoder.py（ImageEncoder）。
运维要点（改这里之前必须知道的事）：
    1. 索引文件（位于 cfg 的 gallery.dir，默认 outputs/gallery）：
        embeddings.npy   (N, D) float32，行已 L2 归一化
        meta.json        [{movie_id, kind, path, image_id}, ...]
        index.meta.json  {dim, n, fingerprint, updated_at, use_faiss}
    2. embeddings.npy 是设备无关的纯 numpy 数组——编码在 GPU 上算完后
       搬回 CPU 落盘（float32），因此换 GPU/换机器无需重建索引，只要
       ImageEncoder.fingerprint() 没变（backbone/维度/分辨率一致）即可直接加载；
    3. 相似度是余弦：向量已归一化，检索就是一次矩阵乘 X @ q，几千张图
       毫秒级；超过十万张再考虑启用 faiss（gallery.use_faiss 预留开关）；
    4. meta.json 与 embeddings.npy 行号严格一一对应，手工编辑任一文件
       都会造成检索结果错位——改数据请走 add()/rebuild()；
    5. 换了 best.pt 或改了 data.image_size 后必须 rebuild（指纹校验）。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..config import ROOT, get_value, path as cfg_path
from ..utils import ensure_dir, get_logger
from .encoder import ImageEncoder

log = get_logger("cineeye.gallery.index")


class GalleryIndex:
    """电影图库：图片 → 向量 → 相似电影检索。

    Args:
        cfg: 全局配置，读取 gallery.* 各项（dir/top_k/aggregation/
            use_faiss/min_similarity）。
        encoder: 外部注入的编码器（identify.py 共享同一实例）；
            None 时检索前自动创建。
        gallery_dir: 显式指定索引目录；缺省按 cfg 解析。

    Raises:
        FileNotFoundError: 编码器权重缺失时由 ImageEncoder 抛出（懒加载时）。
    """

    def __init__(self, cfg: dict | None = None, encoder: ImageEncoder | None = None,
                 gallery_dir: str | Path | None = None):
        cfg = cfg or {}
        self.cfg = cfg
        self.dir = Path(gallery_dir) if gallery_dir else cfg_path(cfg, "gallery.dir", mkdir=True)
        ensure_dir(self.dir)
        # 三个落盘文件：向量矩阵 / 每行元数据 / 索引自述信息
        self.emb_path = self.dir / "embeddings.npy"
        self.meta_path = self.dir / "meta.json"
        self.info_path = self.dir / "index.meta.json"
        self.top_k = int(get_value(cfg, "gallery.top_k", 5))
        # 多图检索的聚合方式：mean / max / log_sum（见 search_images）
        self.aggregation = str(get_value(cfg, "gallery.aggregation", "mean"))
        self.use_faiss = bool(get_value(cfg, "gallery.use_faiss", False))  # 预留，当前恒走 numpy
        self.min_sim = float(get_value(cfg, "gallery.min_similarity", 0.0))
        self.encoder = encoder
        self.X: np.ndarray | None = None
        self.meta: list[dict[str, Any]] = []
        self._load()

    # ---------------------------------------------------------------- 持久化
    def _load(self) -> None:
        """从磁盘加载已有索引；文件缺失则保持空索引等构建。

        embeddings.npy 与 meta.json 必须成对存在，行号一一对应，
        缺任何一个都按空索引处理（宁可重建也不要错位检索）。
        """
        if self.emb_path.exists() and self.meta_path.exists():
            self.X = np.load(self.emb_path).astype(np.float32)
            self.meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
            log.info("图库索引已加载：%d 张图片 / %d 维", len(self.meta), self.X.shape[1] if self.X.ndim == 2 else 0)
        else:
            log.info("图库索引为空，等待构建")

    def _save(self) -> None:
        """整体重写三个索引文件（增量 add 也是全量落盘，几千张规模秒级）。"""
        if self.X is None:
            return
        # 存 float32 且只存 numpy 数组：设备无关，换 GPU/CPU/换机器都能直接 load
        np.save(self.emb_path, self.X.astype(np.float32))
        self.meta_path.write_text(json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8")
        info = {
            "dim": int(self.X.shape[1]),
            "n": int(self.X.shape[0]),
            "fingerprint": self.encoder.fingerprint() if self.encoder else "unknown",
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "use_faiss": self.use_faiss,
        }
        self.info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    def _ensure_encoder(self) -> ImageEncoder:
        """懒加载编码器：检索/写入前才真正加载权重，加快空索引启动。

        Returns:
            可用的 ImageEncoder 实例。
        """
        if self.encoder is None:
            self.encoder = ImageEncoder(cfg=self.cfg)
        return self.encoder

    # ---------------------------------------------------------------- 写入
    def add(self, records: list[dict[str, Any]], *, force: bool = False) -> int:
        """增量添加图片。records: [{movie_id, kind, path}, ...]。返回实际新增数。

        默认跳过已在索引中的 path；force=True 时全部重新编码。

        Args:
            records: 待登记的图片记录（来自 MetaStore.images_for_index）。
            force: True 时忽略去重全量重编码。

        Returns:
            实际新增的图片数。

        Raises:
            ValueError: 新图片特征维度与已有索引不一致（通常意味着换了
            checkpoint 却没 rebuild，应全量重建而非混搭）。
        """
        if not records:
            return 0
        enc = self._ensure_encoder()
        # 以 (movie_id, kind, path) 三元组判重，与 MetaStore 的登记口径一致
        existing = {(m["movie_id"], m.get("kind"), str(m["path"])) for m in self.meta}
        todo = [r for r in records if force or (r["movie_id"], r.get("kind", "frame"), str(r["path"])) not in existing]
        if not todo:
            log.info("图库无新增图片（%d 张已存在）", len(records))
            return 0
        paths = [r["path"] for r in todo]
        X_new = enc.encode_paths(paths)
        # 过滤掉读取失败的行（norm 为 0 的占位行不能进索引，否则检索会错位）
        valid = [i for i in range(len(todo)) if np.linalg.norm(X_new[i]) > 0]
        X_new = X_new[valid]
        todo = [todo[i] for i in valid]
        if not todo:
            return 0
        rows = [{
            "movie_id": str(r["movie_id"]),
            "kind": str(r.get("kind", "frame")),
            "path": str(r["path"]),
            "image_id": r.get("image_id"),
        } for r in todo]
        if self.X is None or self.X.shape[0] == 0:
            self.X = X_new
            self.meta = rows
        else:
            if self.X.shape[1] != X_new.shape[1]:
                raise ValueError(f"特征维度不一致：索引 {self.X.shape[1]} vs 新图片 {X_new.shape[1]}")
            self.X = np.vstack([self.X, X_new])  # 行号与 meta 追加顺序严格对应
            self.meta.extend(rows)
        self._save()
        log.info("图库新增 %d 张，现有 %d 张", len(rows), self.X.shape[0])
        return len(rows)

    def rebuild(self, records: list[dict[str, Any]]) -> int:
        """全量重建索引（换权重/换分辨率/数据异常时的标准操作）。

        Args:
            records: 与 add 相同的图片记录列表。

        Returns:
            重建后的索引图片总数。
        """
        self.X, self.meta = None, []
        return self.add(records, force=True)

    # ---------------------------------------------------------------- 检索
    def search(self, query_vec: np.ndarray, *, top_k: int | None = None,
               kind: str | None = None, min_similarity: float | None = None) -> list[dict]:
        """单向量检索，返回按电影聚合的候选列表。

        相似度计算：向量均 L2 归一化，X @ q 即余弦相似度，取值 [-1, 1]；
        同一部电影有多张图片命中时只保留最高分（"最像的一张"决定归属）。

        Args:
            query_vec: 查询向量（内部会再归一化一次，容忍未归一化输入）。
            top_k: 返回电影数上限；None 用 cfg 的 gallery.top_k。
            kind: 按 kind 过滤（如只要 "frame" 或 "poster"）；None 不过滤。
            min_similarity: 相似度阈值；None 用 cfg 的 gallery.min_similarity。

        Returns:
            [{movie_id, similarity, kind, matched_path}, ...]，按相似度降序。
        """
        q = np.asarray(query_vec, dtype=np.float32).ravel()
        if self.X is None or self.X.shape[0] == 0:
            return []
        q = q / (np.linalg.norm(q) + 1e-9)  # 再归一化一次，容忍外部传入未归一化向量
        sims = self.X @ q  # 余弦相似度：归一化向量的点积
        thr = self.min_sim if min_similarity is None else min_similarity
        k = self.top_k if top_k is None else top_k

        best: dict[str, dict[str, Any]] = {}
        for i, sim in enumerate(sims):
            if float(sim) < thr:
                continue
            m = self.meta[i]
            if kind and m.get("kind") != kind:
                continue  # 按 kind 过滤：例如视频帧查询只比帧库，不比海报库
            mid = m["movie_id"]
            cur = best.get(mid)
            if cur is None or sim > cur["similarity"]:  # 同片多图只留最高分
                best[mid] = {
                    "movie_id": mid,
                    "similarity": round(float(sim), 6),
                    "kind": m.get("kind"),
                    "matched_path": m.get("path"),
                }
        return sorted(best.values(), key=lambda d: -d["similarity"])[:k]

    def search_images(self, images: list[np.ndarray], *, top_k: int | None = None,
                      kind: str | None = None, aggregation: str | None = None) -> list[dict]:
        """多图检索并聚合：适合海报组、视频抽帧、花絮多图。

        聚合方式（gallery.aggregation）的区别：
        - mean：所有命中相似度的平均，最平滑，抗单张误配，
          是默认选择（多帧同片的平均分远高于偶发误配）；
        - max：取最高分，对"只有一张关键图命中"的场景召回最好，
          但更容易被单张巧合误配抬高；
        - log_sum：log(1+x) 后求和，介于两者之间——命中的图片数量越多
          分越高（奖励多证据），但单张极高分的权重被压缩。

        Args:
            images: 查询图片列表（BGR ndarray）。
            top_k: 返回电影数上限。
            kind: 按 kind 过滤库内图片。
            aggregation: 覆盖配置的聚合方式。

        Returns:
            按聚合分降序的电影候选，每项含 n_evidence（命中图片数）。
        """
        if not images:
            return []
        enc = self._ensure_encoder()
        X = enc.encode(images)
        # 每张图先各自取 top_k*3 候选，扩大召回面，聚合后再截断到 top_k
        per_image = [self.search(X[i], top_k=(top_k or self.top_k) * 3, kind=kind) for i in range(len(X))]
        scores: dict[str, list[float]] = {}
        evidence: dict[str, dict] = {}
        for one in per_image:
            for hit in one:
                mid = hit["movie_id"]
                scores.setdefault(mid, []).append(hit["similarity"])
                # 记录该电影最高相似度的单张命中作为证据展示
                if mid not in evidence or hit["similarity"] > evidence[mid]["similarity"]:
                    evidence[mid] = hit
        agg = aggregation or self.aggregation
        fused: dict[str, float] = {}
        for mid, vals in scores.items():
            if agg == "max":
                fused[mid] = max(vals)
            elif agg == "log_sum":
                fused[mid] = float(np.log1p(np.clip(vals, 0, 1)).sum())
            else:  # mean（默认）
                fused[mid] = float(np.mean(vals))
        out = []
        for mid, score in sorted(fused.items(), key=lambda kv: -kv[1])[:top_k or self.top_k]:
            item = dict(evidence[mid])
            item["similarity"] = round(score, 6)
            item["n_evidence"] = len(scores[mid])  # 命中的图片张数，供上层做证据展示
            out.append(item)
        return out

    def stats(self) -> dict:
        """索引统计信息（status 接口用）。

        Returns:
            含图片数/维度/电影数/按 kind 分布/top 电影/编码器指纹/更新时间。
        """
        info: dict[str, Any] = {}
        if self.info_path.exists():
            info = json.loads(self.info_path.read_text(encoding="utf-8"))
        per_kind: dict[str, int] = {}
        per_movie: dict[str, int] = {}
        for m in self.meta:
            per_kind[m.get("kind", "?")] = per_kind.get(m.get("kind", "?"), 0) + 1
            per_movie[m["movie_id"]] = per_movie.get(m["movie_id"], 0) + 1
        return {
            "images": int(self.X.shape[0]) if self.X is not None else 0,
            "dim": int(self.X.shape[1]) if self.X is not None and self.X.ndim == 2 else 0,
            "movies": len(per_movie),
            "by_kind": per_kind,
            "top_movies": sorted(per_movie.items(), key=lambda kv: -kv[1])[:10],
            "encoder_fingerprint": info.get("fingerprint"),
            "updated_at": info.get("updated_at"),
        }
