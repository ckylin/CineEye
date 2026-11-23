"""演员识别：人脸检测 → 分类头打分 + 参考库检索 → 融合 → 聚合到演员/电影。

识别链路（两条证据互补，缺一可用）：
    1. **分类头**（训练得来）：脸部特征 → 线性分类器 → 每位演员一个概率。
       它"见过"训练集里的演员，对熟脸判断更果断，但天生无法说"不认识"。
    2. **参考库检索**（无需训练）：脸部特征 → 与参考向量算余弦相似度。
       它天然支持开放集（新演员加几张图即可生效），但容易被相近长相干扰。

    融合分 = w_cls × 分类概率 + w_gal × max(0, 检索相似度)，权重可配
    （actors.fusion.classifier / actors.fusion.gallery）。低于
    actors.unknown_threshold 时直接返回空（"宁可少认，不要乱认"）。

被谁使用：identify.py（identify_images/identify_video 的 actor 字段）、
    scripts/13_actor_reference.py（建库/查库）、scripts/14_identify.py（--actor-image）。
依赖：OpenCV（Haar 检测）、actors/encoder.py（演员特征）、actors/train.py
    （分类头结构 ActorHead）、metadata.MetaStore（演员↔电影反查）。
运维要点（改这里之前必须知道的事）：
    1. 分类头与参考库都绑定编码器指纹；换 encoder/backbone/分辨率后旧产物
       会被判定为不兼容并自动忽略——必须重跑 16_train_actors.py；
    2. 参考库目录约定 outputs/actors/<演员名>/*.jpg 仍可用（不训练也能识别），
       但训练后以 train 划分的向量为准（写在同一位置，会被覆盖）；
    3. 检测开启但图里没有人脸时默认"不猜"；生活照/全身照可打开
       actors.fallback_whole_image，用中心裁剪兜底；
    4. 目录名/CSV 里的演员名要与 persons.name 一致，否则只能给出名字、
       查不到参演电影（不是错误，但会缺上下文）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from ..config import ROOT, get_value, path as cfg_path
from ..metadata import MetaStore
from ..utils import get_logger
from .dataset import IMG_EXTS, prepare_face
from .encoder import ActorEncoder
from .face import FaceDetector
from .train import ActorHead

log = get_logger("cineeye.actors")

__all__ = ["ActorRecognizer", "IMG_EXTS"]


class ActorRecognizer:
    """演员识别器：分类头 + 参考库检索 + 多脸支持。

    Args:
        cfg: 全局配置，读取 actors.* 各项。
        encoder: **已废弃**，仅为兼容旧调用保留。演员特征必须与建库/训练时
            的编码器一致，因此这里固定使用 ActorEncoder(actors.encoder)，
            不再复用电影图库的 ImageEncoder（两者向量空间不同）。
        store: 元数据库实例；None 时自建。

    Raises:
        RuntimeError: 检测开启但 Haar 级联加载失败。
    """

    def __init__(self, cfg: dict | None = None, encoder: Any | None = None,
                 store: MetaStore | None = None):
        cfg = cfg or {}
        self.cfg = cfg
        self.enabled = bool(get_value(cfg, "actors.enabled", True))
        self.detect_enabled = bool(get_value(cfg, "actors.face_detection", True))
        self.top_k = int(get_value(cfg, "actors.top_k", 3))
        self.min_sim = float(get_value(cfg, "actors.min_similarity", 0.0))
        self.fallback_whole = bool(get_value(cfg, "actors.fallback_whole_image", True))
        self.pad = float(get_value(cfg, "actors.face_pad", 0.2))
        self.crop_mode = str(get_value(cfg, "actors.crop", "detect"))
        self.w_cls = float(get_value(cfg, "actors.fusion.classifier", 0.7))
        self.w_gal = float(get_value(cfg, "actors.fusion.gallery", 0.3))
        self.ref_dir = cfg_path(cfg, "actors.reference_dir", mkdir=True)
        self.store = store or MetaStore(cfg=cfg)
        self.detector = FaceDetector(cfg) if self.detect_enabled else None
        if encoder is not None:
            # 不报错、不中断：旧调用方（identify.py）仍会传电影编码器进来
            log.debug("忽略外部传入的 encoder：演员识别固定使用 ActorEncoder 以保证向量空间一致")

        self.encoder = ActorEncoder(cfg=cfg)
        # 分类头（可选）：不存在时退化为纯参考库检索
        self.head_path = Path(get_value(cfg, "actors.head_path",
                                        "outputs/checkpoints/actor_head.pt"))
        if not self.head_path.is_absolute():
            self.head_path = ROOT / self.head_path
        self.model: ActorHead | None = None
        self.label_map: dict[int, str] = {}
        self.threshold = float(get_value(cfg, "actors.unknown_threshold", 0.0) or 0.0)
        self._load_head()

        # 参考库（可选）：embeddings.npy + names.json + encoder.json
        self.emb_path = self.ref_dir / "embeddings.npy"
        self.names_path = self.ref_dir / "names.json"
        self.enc_meta_path = self.ref_dir / "encoder.json"
        self.X: np.ndarray | None = None
        self.names: list[str] = []
        self._load_gallery()

    # ------------------------------------------------------------------ 加载
    def _load_head(self) -> None:
        """加载训练好的演员分类头；不存在则保持 None（纯检索模式）。"""
        if not self.head_path.exists():
            log.info("未找到演员分类头（%s），退化为参考库检索模式", self.head_path)
            return
        try:
            ckpt = torch.load(self.head_path, map_location="cpu", weights_only=False)
        except Exception as exc:  # noqa: BLE001 - 损坏的 checkpoint 不应阻塞服务启动
            log.warning("演员分类头加载失败，忽略：%s", exc)
            return
        enc_meta = ckpt.get("encoder", {})
        if enc_meta.get("fingerprint") and enc_meta["fingerprint"] != self.encoder.fingerprint():
            # 指纹不一致 = 用 A 编码器训的头拿 B 编码器提的特征，结果没有意义
            log.warning("演员分类头与当前编码器不匹配（%s != %s），已忽略；"
                        "请重跑 scripts/16_train_actors.py",
                        enc_meta.get("fingerprint"), self.encoder.fingerprint())
            return
        label_map = {int(k): v for k, v in ckpt.get("label_map", {}).items()}
        if not label_map:
            log.warning("演员分类头缺少 label_map，忽略")
            return
        model = ActorHead(int(ckpt.get("feature_dim", self.encoder.feature_dim)), len(label_map))
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        self.model = model
        self.label_map = label_map
        # 配置显式给了阈值就优先用配置；否则用训练时估计的值
        if get_value(self.cfg, "actors.unknown_threshold") is None:
            self.threshold = float(ckpt.get("threshold", 0.0))
        log.info("演员分类头已加载：%d 位演员 / 阈值 %.2f", len(label_map), self.threshold)

    def _load_gallery(self) -> None:
        """加载参考向量库；编码器指纹不一致时拒绝使用，避免静默错配。"""
        if not (self.emb_path.exists() and self.names_path.exists()):
            return
        if not self.enc_meta_path.exists():
            # 旧版检索库没有 encoder.json，无法确认它是哪个编码器产出的。
            # 直接用可能"拿 A 的向量查 B 的库"，宁可忽略并提示重建。
            log.warning("演员参考库缺少 encoder.json（旧版产物），无法校验编码器一致性，已忽略；"
                        "请重跑 scripts/13_actor_reference.py --build 或 scripts/16_train_actors.py")
            return
        try:
            meta = json.loads(self.enc_meta_path.read_text(encoding="utf-8"))
            if meta.get("fingerprint") and meta["fingerprint"] != self.encoder.fingerprint():
                log.warning("演员参考库由另一个编码器构建（%s），已忽略；"
                            "请重跑 13_actor_reference.py --build 或 16_train_actors.py",
                            meta.get("fingerprint"))
                return
        except (OSError, ValueError):
            log.warning("演员参考库的 encoder.json 无法解析，已忽略；请重跑建库脚本")
            return
        X = np.load(self.emb_path).astype(np.float32)
        names = json.loads(self.names_path.read_text(encoding="utf-8"))
        if len(names) != X.shape[0]:
            log.warning("演员参考库损坏：向量 %d 行 / 名字 %d 个，已忽略", X.shape[0], len(names))
            return
        self.X, self.names = X, [str(n) for n in names]
        log.info("演员参考库已加载：%d 位演员 / %d 张参考图",
                 len(set(self.names)), len(self.names))

    # ------------------------------------------------------------------ 建库
    @staticmethod
    def iter_reference_images(ref_dir: Path) -> list[tuple[str, Path]]:
        """返回 (演员名, 图片路径)；目录名即演员名，按名称排序保证稳定。"""
        out: list[tuple[str, Path]] = []
        if not ref_dir.exists():
            return out
        for actor_dir in sorted(p for p in ref_dir.iterdir() if p.is_dir() and not p.name.startswith(".")):
            for img in sorted(p for p in actor_dir.rglob("*") if p.suffix.lower() in IMG_EXTS):
                out.append((actor_dir.name, img))
        return out

    def build_reference(self) -> int:
        """从 outputs/actors/<演员名>/ 构建参考向量库（不训练也能用）。

        Returns:
            成功编码的参考图数量；目录为空返回 0。
        """
        pairs = self.iter_reference_images(self.ref_dir)
        if not pairs:
            log.warning("演员参考图库为空：%s/<演员名>/*.jpg", self.ref_dir)
            return 0
        faces: list[np.ndarray] = []
        names: list[str] = []
        for actor, path in pairs:
            img = cv2.imread(str(path))
            if img is None:
                log.warning("跳过无法读取的参考图：%s", path)
                continue
            crop, _ = prepare_face(img, self.detector, mode=self.crop_mode,
                                   pad=self.pad, fallback_whole=self.fallback_whole)
            if crop is None or crop.size == 0:
                continue
            faces.append(crop)
            names.append(actor)
        if not faces:
            return 0
        X = self.encoder.encode(faces)
        self.X, self.names = X, names
        np.save(self.emb_path, X.astype(np.float32))
        self.names_path.write_text(json.dumps(names, ensure_ascii=False, indent=2), encoding="utf-8")
        self.enc_meta_path.write_text(json.dumps({
            "fingerprint": self.encoder.fingerprint(), "mode": self.encoder.mode,
            "backbone": self.encoder.backbone_name, "image_size": self.encoder.image_size,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("演员参考库已构建：%d 位演员 / %d 张图", len(set(names)), len(names))
        return len(names)

    # ------------------------------------------------------------------ 打分
    def _gallery_scores(self, q: np.ndarray) -> dict[str, float]:
        """查询向量 q 与参考库逐行点积，按演员取最高分（最像的一张）。"""
        if self.X is None or self.X.shape[0] == 0:
            return {}
        sims = self.X @ q  # 向量均已 L2 归一化，点积即余弦相似度
        best: dict[str, float] = {}
        for name, sim in zip(self.names, sims):
            s = float(sim)
            if s < self.min_sim:
                continue
            best[name] = max(best.get(name, -1.0), s)
        return best

    def _classifier_scores(self, q: np.ndarray) -> dict[str, float]:
        """分类头输出 softmax 概率（演员名 → 概率）。"""
        if self.model is None:
            return {}
        with torch.no_grad():
            logits = self.model(torch.tensor(q[None, :], dtype=torch.float32))
            probs = torch.softmax(logits, dim=1)[0].numpy()
        return {self.label_map[int(i)]: float(p) for i, p in enumerate(probs)}

    def _rank(self, crop: np.ndarray) -> list[dict[str, Any]]:
        """对单张脸打分并排序，返回候选（含概率/检索分/融合分，便于排障）。"""
        q = self.encoder.encode([crop])[0]
        cls = self._classifier_scores(q)
        gal = self._gallery_scores(q)
        names = set(cls) | set(gal)
        if not names:
            return []
        rows = []
        for name in names:
            p = cls.get(name, 0.0)
            g = gal.get(name, 0.0)
            if cls and gal:
                fused = self.w_cls * p + self.w_gal * max(0.0, g)
            elif cls:
                fused = p
            else:
                fused = g  # 纯检索模式沿用余弦相似度（可能为负，threshold 默认 0 已能过滤）
            rows.append({
                "name": name,
                "score": round(float(fused), 6),
                "probability": round(float(p), 6) if cls else None,
                "similarity": round(float(g), 6) if gal else None,
            })
        rows.sort(key=lambda d: -d["score"])
        return rows

    def _accept(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """按阈值过滤：最高分低于阈值 → 视为"不认识"，返回空列表。"""
        if not rows:
            return []
        if self.threshold > 0 and rows[0]["score"] < self.threshold:
            log.debug("演员最高分 %.3f < 阈值 %.3f，判定为未知", rows[0]["score"], self.threshold)
            return []
        return rows

    # ------------------------------------------------------------------ 识别
    def recognize(self, image: np.ndarray) -> list[dict[str, Any]]:
        """识别图中**最大人脸**属于哪位演员（单脸场景的便捷接口）。

        Args:
            image: BGR ndarray。

        Returns:
            [{name, score, probability, similarity}, ...] 按融合分降序；
            未启用/无模型且无参考库/无人脸/低于阈值时返回空列表。
        """
        if not self.enabled:
            return []
        crop, _ = prepare_face(image, self.detector, mode=self.crop_mode,
                               pad=self.pad, fallback_whole=self.fallback_whole)
        if crop is None or crop.size == 0:
            return []
        return self._accept(self._rank(crop))[:self.top_k]

    def recognize_all(self, image: np.ndarray) -> dict[str, Any]:
        """识别图中**所有人脸**（合照、双人剧照、群像海报）。

        检测到多张脸时按面积从大到小逐个识别；检测不到脸时，若配置允许
        （fallback_whole_image），把整图当作一张脸处理，否则返回 0 张。

        Args:
            image: BGR ndarray。

        Returns:
            {num_faces, faces: [{bbox, actor, confidence, candidates}], unknown}。
            bbox 为 [x, y, w, h] 像素坐标；actor 为 None 表示该脸低于阈值。
        """
        out: dict[str, Any] = {"num_faces": 0, "faces": []}
        if not self.enabled or image is None or image.size == 0:
            return out
        boxes = self.detector.detect(image) if self.detector else []
        crops: list[tuple[list[int] | None, np.ndarray]] = []
        if boxes:
            for b in boxes:
                crop = b.crop(image, pad=self.pad)
                if crop is not None and crop.size:
                    crops.append(([b.x, b.y, b.w, b.h], crop))
        else:
            crop, _ = prepare_face(image, self.detector, mode=self.crop_mode,
                                   pad=self.pad, fallback_whole=self.fallback_whole)
            if crop is not None and crop.size:
                crops.append((None, crop))
        for bbox, crop in crops:
            rows = self._accept(self._rank(crop))[:self.top_k]
            top = rows[0] if rows else None
            out["faces"].append({
                "bbox": bbox,
                "actor": top["name"] if top else None,
                "confidence": top["score"] if top else 0.0,
                "candidates": rows,
            })
        out["num_faces"] = len(out["faces"])
        out["unknown"] = all(f["actor"] is None for f in out["faces"]) if out["faces"] else True
        return out

    def recognize_with_movies(self, image: np.ndarray) -> dict[str, Any] | None:
        """识别演员并附带参演电影列表（优先取面积最大的人脸）。

        Args:
            image: BGR ndarray。

        Returns:
            {name, confidence, candidates, person_id, movies, faces, num_faces}；
            没有任何可接受的人脸时返回 None。person_id 为 None 表示该名字不在
            persons 表（通常是照片目录名与元数据演员名不一致）。
        """
        detail = self.recognize_all(image)
        faces = detail.get("faces", [])
        known = [f for f in faces if f["actor"]]
        if not known:
            return None
        top = known[0]
        person = self.store.find_person(top["actor"])
        return {
            "name": top["actor"],
            "confidence": top["confidence"],
            "candidates": top["candidates"],
            "person_id": person["id"] if person else None,
            "movies": person["movies"] if person else [],
            "faces": faces,
            "num_faces": detail.get("num_faces", len(faces)),
        }

    # ------------------------------------------------------------------ 巡检
    def status(self) -> dict[str, Any]:
        """返回识别器状态，供 /health、--status 与排障使用。"""
        return {
            "enabled": self.enabled,
            "face_detection": self.detect_enabled,
            "mode": "classifier+gallery" if (self.model and self.X is not None)
                    else ("classifier" if self.model else ("gallery" if self.X is not None else "none")),
            "actors": sorted({*self.label_map.values(), *self.names}),
            "num_actors": len({*self.label_map.values(), *self.names}),
            "gallery_images": 0 if self.X is None else int(self.X.shape[0]),
            "threshold": self.threshold,
            "encoder": {"mode": self.encoder.mode, "backbone": self.encoder.backbone_name,
                        "feature_dim": self.encoder.feature_dim,
                        "fingerprint": self.encoder.fingerprint(), "device": str(self.encoder.device)},
            "head_path": str(self.head_path),
        }
