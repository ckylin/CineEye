"""统一识别服务：任意视觉线索 → 电影档案（多片名 / 演员 / 置信度）。

用途：面向 serve.py / CLI 的顶层门面——输入图片/视频/base64，输出带
    完整元数据（主片名/原名/别名/演员/年份/导演/海报）的识别结果。
被谁使用：serve.py（HTTP 接口）、CLI 命令。
依赖：gallery（开放集检索）、actors（演员识别）、metadata（档案库）、
    predict.Predictor（封闭集分类器，可选）。
运维要点（改这里之前必须知道的事）：
    1. 编排流程：图片/视频 → 抽帧 → 人脸检测与演员识别（可选）→
       图库向量检索（开放集）→ 封闭集分类器概率融合（帧类输入）→
       元数据库补全 → 统一结果结构。与原有 Predictor（封闭集帧分类）
       互补：Identifier 负责开放集与多类型输入；
    2. 返回门槛（identify.min_gallery_similarity，默认 0.8）：只返回**图库相似度
       ≥ 门槛**的候选，低于门槛的乱猜一律不返回；confidence 与排序都以图库
       相似度为准，做到"所见即所得"。旧版曾把 0.6*图库 + 0.4*分类器的融合分
       写回 similarity，导致训练原图（图库相似度 1.0）被分类器拉低到 0.62、
       甚至被相似度 0.87 的错误候选挤到第二名，表现为"训练图也认不出"。
       现在分类器融合分降级为辅助证据（evidence.fused_score /
       evidence.classifier_confidence），不再覆盖主分数与门槛；
    3. device 统一解析：构造时 resolve_device() 只解析一次，同时传给
       图库编码器与分类器——避免 GPU 机器上出现"图库走 CPU、分类器
       走 GPU"的设备割裂，也保证 serve.py 的 --device 参数全局生效；
    4. 分类器权重缺失只告警不失败（降级为纯图库检索），log 里会出现
       "分类器权重不可用"属正常现象。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .actors import ActorRecognizer
from .config import ROOT, get_value
from .data.extract_frames import sample_frames_from_video
from .device import resolve_device
from .gallery import GalleryIndex, ImageEncoder
from .metadata import MetaStore
from .predict import Predictor
from .utils import get_logger

log = get_logger("cineeye.identify")


class Identifier:
    """电影档案识别器：图库检索 + 演员识别 + 元数据丰富。

    编排者角色：自身不做模型推理，负责把图库索引、封闭集分类器、
    演员识别、元数据库四个组件的输出融合成统一结果。

    Args:
        cfg: 全局配置 dict；None 时加载默认配置。
        with_actors: 是否启用演员识别；None 按 cfg 的 actors.enabled。
        with_classifier: 是否加载封闭集分类器（权重缺失时自动降级跳过）。
        device: 显式设备字符串（如 "cuda"/"cpu"）；None 按 train.device
            自动解析一次，全局共用。
    """

    def __init__(self, cfg: dict | None = None, *, with_actors: bool | None = None,
                 with_classifier: bool = True, device: str | None = None):
        from .config import load_config

        self.cfg = cfg or load_config()
        # 统一解析一次设备，传给图库编码器与分类器，避免 GPU 机器上图库走 CPU、
        # 分类器走 GPU 的设备割裂；也保证 serve.py 的 --device 参数生效。
        self.device = resolve_device(
            str(device or get_value(self.cfg, "train.device", "auto")),
            int(get_value(self.cfg, "train.cpu_threads", 8)),
        )
        self.store = MetaStore(cfg=self.cfg)
        self.encoder = ImageEncoder(cfg=self.cfg, device=self.device)
        self.gallery = GalleryIndex(cfg=self.cfg, encoder=self.encoder)
        if with_actors is None:
            with_actors = bool(get_value(self.cfg, "actors.enabled", True))
        self.actors = ActorRecognizer(cfg=self.cfg, encoder=self.encoder, store=self.store) if with_actors else None
        self.top_k = int(get_value(self.cfg, "gallery.top_k", 5))
        # 图库相似度返回门槛：低于此值的候选不返回（默认 0.8，即 80%）。
        # 只作用于图库检索的 similarity，不受分类器融合影响；设 0 可关闭门槛。
        self.min_similarity = float(get_value(self.cfg, "identify.min_similarity", 0.8))
        # 封闭集分类器作为帧类图片的辅助证据（与图库检索互补）
        self.classifier = None
        if with_classifier:
            try:
                self.classifier = Predictor(cfg=self.cfg, device=self.device)
            except FileNotFoundError as exc:
                # 权重缺失只降级不失败：开放集图库检索依然可用
                log.warning("分类器权重不可用，仅用图库检索：%s", exc)

    # ---------------------------------------------------------------- 分类器证据
    def _classifier_scores(self, images: list[np.ndarray]) -> dict[str, float]:
        """用封闭集分类器给帧类图片打分，返回 {label: confidence}。

        分类器只在训练类别内有效，因此仅作为加权证据，不覆盖图库的开放集能力。

        Args:
            images: 帧类图片列表（BGR ndarray）。

        Returns:
            {movie_id_label: 平均概率}；分类器未加载或打分异常时返回空 dict
            （异常降级，不阻塞主流程）。
        """
        if self.classifier is None or not images:
            return {}
        try:
            tensors = [self.classifier._transform(img) for img in images]
            import torch
            probs = self.classifier.predict_proba(torch.stack(tensors))
            avg = probs.mean(axis=0)  # 多帧平均：降低单帧噪声对分类概率的影响
            return {self.classifier.idx2label[i]: float(avg[i]) for i in range(len(avg))}
        except Exception as exc:  # noqa: BLE001
            log.warning("分类器打分失败，跳过：%s", exc)
            return {}

    # ---------------------------------------------------------------- 丰富档案
    def _enrich(self, hit: dict[str, Any]) -> dict[str, Any]:
        """把图库检索的 movie_id 命中补全成完整电影档案。

        Args:
            hit: GalleryIndex 返回的单条命中。

        Returns:
            统一结果结构：含 title/original_title/aliases/year/director/
            cast/poster 及 evidence 证据块。
        """
        movie = self.store.get_movie(hit["movie_id"]) or {}
        aliases = movie.get("aliases") or []
        # 找不到档案时 title 回退为 movie_id，保证结果结构始终可用
        return {
            "movie_id": hit["movie_id"],
            "confidence": hit.get("similarity", 0.0),
            "title": movie.get("primary_title") or hit["movie_id"],
            "original_title": movie.get("original_title"),
            "aliases": [
                {"alias": a["alias"], "lang": a.get("lang"), "region": a.get("region"), "source": a.get("source")}
                for a in aliases
            ],
            "year": movie.get("year"),
            "director": movie.get("director"),
            "cast": movie.get("cast") or [],
            "poster": movie.get("poster_path"),
            "evidence": {
                "kind": hit.get("kind"),
                # similarity 恒为图库相似度（与返回门槛、前端展示同口径）
                "similarity": hit.get("similarity"),
                "matched_path": hit.get("matched_path"),
                "n_evidence": hit.get("n_evidence", 1),
                # 分类器融合分仅作辅助证据，不再覆盖上面的图库相似度
                "fused_score": hit.get("fused_score"),
            },
        }

    # ---------------------------------------------------------------- 主入口
    def identify_images(self, images: list[np.ndarray], *, kind: str | None = None,
                        top_k: int | None = None,                         aggregation: str | None = None,
                        with_actors: bool | None = True) -> dict[str, Any]:
        """图片列表识别主入口（编排流程见模块 docstring）。

        Args:
            images: BGR ndarray 列表（多图会聚合检索）。
            kind: 按 kind 过滤图库（None/poster/frame/...）。
            top_k: 返回候选电影数；None 用配置默认。
            aggregation: 多图聚合方式（mean/max/log_sum）；None 用配置。
            with_actors: 是否附带演员识别。

        Returns:
            {input_type, num_images, matches, actor, elapsed_ms}，
            matches 内每项为 _enrich 结构：similarity 为图库相似度（≥门槛、
            与 confidence 和排序同口径）；classifier_confidence 与 fused_score
            仅作辅助证据。全部候选低于门槛时 matches 为空数组。
        """
        t0 = time.time()
        k = top_k or self.top_k
        # 多取 3 倍候选，融合分类器重排后再截断到 top_k，
        # 否则真候选可能排在融合前的 top_k 之外、没机会被捞回来
        hits = self.gallery.search_images(images, top_k=k * 3,
                                          kind=kind, aggregation=aggregation)

        # 第一步：图库相似度硬门槛。低于 min_similarity 的候选直接丢弃，
        # 这一步在分类器介入之前完成，保证"返回的一定是画面够像的"。
        # 用 k*3 的召回面先筛，避免真候选恰好落在 top_k 边界外被误杀。
        hits = [h for h in hits if h["similarity"] >= self.min_similarity]

        # 第二步：分类器仅作为辅助证据记录，不再覆盖图库相似度、也不重排。
        # 旧版把融合分写回 similarity，会让相似度 1.0 的训练原图被分类器
        # 拉低到 0.62，出现"训练图也认不出"；现在相似度/排序/展示三者统一。
        cls_scores: dict[str, float] = {}
        if (kind in (None, "frame")) and self.classifier:
            cls_scores = self._classifier_scores(images)
            for hit in hits:
                c = cls_scores.get(hit["movie_id"])
                if c is not None:
                    hit["classifier_confidence"] = round(c, 6)
                    hit["fused_score"] = round(0.6 * hit["similarity"] + 0.4 * c, 6)

        # 第三步：按图库相似度降序截断（第一名即画面最像的，分数与门槛同口径）
        hits.sort(key=lambda d: -d["similarity"])
        hits = hits[:k]
        matches = [self._enrich(h) for h in hits]
        for m in matches:
            m["evidence"]["classifier_confidence"] = cls_scores.get(m["movie_id"])

        # 演员识别只看第一张图（海报/剧照通常主角在首图）
        actor_info = None
        if with_actors and self.actors and images:
            actor_info = self.actors.recognize_with_movies(images[0])

        return {
            "input_type": "images",
            "num_images": len(images),
            "matches": matches,
            "actor": actor_info,
            "elapsed_ms": int((time.time() - t0) * 1000),
        }

    def identify_image(self, image: np.ndarray | str | Path, **kwargs) -> dict[str, Any]:
        """单张图片识别，支持直接传图片路径。

        Args:
            image: BGR ndarray 或图片路径（相对路径回退项目根解析）。
            **kwargs: 透传给 identify_images。

        Returns:
            与 identify_images 相同的结果结构。
        """
        if isinstance(image, (str, Path)):
            image = self.encoder.load_image(image)
        return self.identify_images([image], **kwargs)

    def identify_video(self, video_path: str | Path, *, num_frames: int = 16,
                       kind: str | None = "frame", top_k: int | None = None,
                       with_actors: bool = False) -> dict[str, Any]:
        """视频识别：均匀抽帧 → 多帧聚合检索 → 分类器融合 → 元数据补全。

        与 identify_images 的差异：kind 默认 "frame"（只比帧库）、
        演员识别默认关闭（抽帧多为远景，人脸太小不可靠）。

        Args:
            video_path: 视频文件路径（相对路径回退项目根解析）。
            num_frames: 抽帧数量。
            kind: 图库过滤类型，默认 "frame"。
            top_k: 返回候选电影数。
            with_actors: 是否对首帧做演员识别。

        Returns:
            {input_type, source, num_frames, matches, actor, elapsed_ms}。

        Raises:
            cv2 相关异常由 sample_frames_from_video 在文件不可读时抛出。
        """
        t0 = time.time()
        p = Path(video_path)
        if not p.is_absolute() and not p.exists():
            p = ROOT / p
        frames = sample_frames_from_video(p, num_frames)
        k = top_k or self.top_k
        # 与 identify_images 完全一致的策略：先按图库相似度过硬门槛，
        # 分类器只记录辅助证据，最后按图库相似度排序截断。
        hits = self.gallery.search_images(frames, top_k=k * 3, kind=kind)
        hits = [h for h in hits if h["similarity"] >= self.min_similarity]
        cls_scores: dict[str, float] = {}
        if kind in (None, "frame") and self.classifier:
            cls_scores = self._classifier_scores(frames)
            for hit in hits:
                c = cls_scores.get(hit["movie_id"])
                if c is not None:
                    hit["classifier_confidence"] = round(c, 6)
                    hit["fused_score"] = round(0.6 * hit["similarity"] + 0.4 * c, 6)
        hits.sort(key=lambda d: -d["similarity"])
        hits = hits[:k]
        matches = [self._enrich(h) for h in hits]
        for m in matches:
            m["evidence"]["classifier_confidence"] = cls_scores.get(m["movie_id"])
        actor_info = None
        if with_actors and self.actors and frames:
            actor_info = self.actors.recognize_with_movies(frames[0])
        return {
            "input_type": "video",
            "source": str(p),
            "num_frames": len(frames),
            "matches": matches,
            "actor": actor_info,
            "elapsed_ms": int((time.time() - t0) * 1000),
        }

    # ---------------------------------------------------------------- 元数据检索
    def search_movies(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """按片名/别名（含多语言）搜索电影。

        Args:
            query: 查询片名（归一化 + 模糊匹配在 MetaStore.search 内完成）。
            top_k: 最多返回数。

        Returns:
            携带完整档案摘要的候选列表（含 matched_alias 命中依据）。
        """
        hits = self.store.search(query, top_k=top_k)
        out = []
        for h in hits:
            movie = self.store.get_movie(h["movie_id"]) or {}
            out.append({
                "movie_id": h["movie_id"],
                "score": h["score"],
                "matched_alias": h["matched_alias"],
                "title": movie.get("primary_title"),
                "original_title": movie.get("original_title"),
                "aliases": movie.get("aliases") or [],
                "year": movie.get("year"),
                "cast": movie.get("cast") or [],
            })
        return out

    def get_movie(self, movie_id: str) -> dict[str, Any] | None:
        """按 id 取完整电影档案（透传 MetaStore）。"""
        return self.store.get_movie(movie_id)

    def list_movies(self, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        """分页列出电影档案（透传 MetaStore）。"""
        return self.store.list_movies(limit=limit, offset=offset)

    def get_actor(self, person_id: str) -> dict[str, Any] | None:
        """按 person_id 取演员档案及参演电影（透传 MetaStore）。"""
        return self.store.get_actor(person_id)

    def status(self) -> dict[str, Any]:
        """健康检查/巡检信息：元数据、图库、演员模块、编码器与设备状态。

        Returns:
            供 serve.py /status 接口与运维脚本使用的状态 dict。
        """

    def status(self) -> dict[str, Any]:
        return {
            "meta": self.store.stats(),
            "gallery": self.gallery.stats(),
            "actors_enabled": self.actors is not None,
            "encoder": f"{self.encoder.backbone_name}/{self.encoder.feature_dim}d",
            "device": str(self.device),
        }
