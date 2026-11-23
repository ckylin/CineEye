"""统一识别的图库相似度门槛测试。

验证「identify.min_similarity」这条硬规则：低于门槛的候选不返回，
且展示的 similarity / confidence 不被分类器融合分覆盖。
用假图库和假分类器隔离模型，纯逻辑测试、秒级完成。
"""

import numpy as np
import pytest

from cineeye.config import load_config
from cineeye.identify import Identifier


class FakeGallery:
    """假图库：search_images 直接返回预置命中，不编码真实图片。"""

    def __init__(self, hits):
        self.hits = hits

    def search_images(self, images, **kwargs):
        # 返回深拷贝，避免测试间互相污染（融合会往 hit 里写字段）
        return [dict(h) for h in self.hits]


class FakeClassifier:
    """假分类器：给每个 movie_id 一个固定的高概率，验证它不会覆盖相似度。"""

    idx2label = {0: "movie_01", 1: "movie_02"}

    def _transform(self, img):
        import torch
        return torch.zeros(3, 4, 4)  # 真实 _transform 返回 tensor，供 torch.stack 使用

    def predict_proba(self, batch):
        # movie_01=0.9 / movie_02=0.1：若被融合，movie_02 的分数会被拉高
        return np.array([[0.9, 0.1]] * len(batch))


def make_identifier(cfg, hits, with_classifier=True):
    """绕过 __init__ 的模型加载，手工装配 Identifier 依赖。"""
    ident = Identifier.__new__(Identifier)
    ident.cfg = cfg
    ident.top_k = 5
    ident.min_similarity = float(cfg.get("identify", {}).get("min_similarity", 0.8))
    ident.gallery = FakeGallery(hits)
    ident.classifier = FakeClassifier() if with_classifier else None
    ident.actors = None
    # _enrich 需要一个能返回档案的 store
    ident.store = type("FakeStore", (), {"get_movie": lambda self, mid: {}})()
    return ident


@pytest.fixture
def cfg():
    return load_config()


def test_min_similarity_filters_low_hits(cfg):
    """低于 0.8 的候选必须被丢弃，只有够像的留下来。"""
    hits = [
        {"movie_id": "movie_01", "similarity": 0.95, "kind": "frame", "matched_path": "a.jpg", "n_evidence": 1},
        {"movie_id": "movie_02", "similarity": 0.55, "kind": "frame", "matched_path": "b.jpg", "n_evidence": 1},
        {"movie_id": "movie_03", "similarity": 0.79, "kind": "frame", "matched_path": "c.jpg", "n_evidence": 1},
    ]
    ident = make_identifier(cfg, hits)
    out = ident.identify_images([np.zeros((4, 4, 3), dtype=np.uint8)], with_actors=False)
    assert [m["movie_id"] for m in out["matches"]] == ["movie_01"]


def test_similarity_not_overridden_by_classifier(cfg):
    """图库相似度 1.0 的原图不能被分类器拉低，融合分只作辅助字段。"""
    hits = [
        # movie_02 相似度只有 0.85，但分类器给它 0.1；movie_01 相似度 1.0
        {"movie_id": "movie_02", "similarity": 0.85, "kind": "frame", "matched_path": "b.jpg", "n_evidence": 1},
        {"movie_id": "movie_01", "similarity": 1.0, "kind": "frame", "matched_path": "a.jpg", "n_evidence": 1},
    ]
    ident = make_identifier(cfg, hits)
    out = ident.identify_images([np.zeros((4, 4, 3), dtype=np.uint8)], with_actors=False)
    top = out["matches"][0]
    # 排序按图库相似度：1.0 的 movie_01 排第一，不再被分类器挤下去
    assert top["movie_id"] == "movie_01"
    assert top["confidence"] == pytest.approx(1.0)
    assert top["evidence"]["similarity"] == pytest.approx(1.0)
    # 融合分作为辅助证据保留，但绝不覆盖 similarity
    assert top["evidence"]["fused_score"] == pytest.approx(0.6 * 1.0 + 0.4 * 0.9)


def test_all_below_threshold_returns_empty(cfg):
    """全部低于门槛时 matches 为空——宁可不认，也不给低分乱猜。"""
    hits = [
        {"movie_id": "movie_01", "similarity": 0.5, "kind": "frame", "matched_path": "a.jpg", "n_evidence": 1},
        {"movie_id": "movie_02", "similarity": 0.6, "kind": "frame", "matched_path": "b.jpg", "n_evidence": 1},
    ]
    ident = make_identifier(cfg, hits)
    out = ident.identify_images([np.zeros((4, 4, 3), dtype=np.uint8)], with_actors=False)
    assert out["matches"] == []


def test_threshold_zero_keeps_all(cfg):
    """门槛设为 0 可关闭过滤（向后兼容旧行为）。"""
    cfg["identify"]["min_similarity"] = 0.0
    hits = [
        {"movie_id": "movie_01", "similarity": 0.5, "kind": "frame", "matched_path": "a.jpg", "n_evidence": 1},
    ]
    ident = make_identifier(cfg, hits)
    out = ident.identify_images([np.zeros((4, 4, 3), dtype=np.uint8)], with_actors=False)
    assert len(out["matches"]) == 1
