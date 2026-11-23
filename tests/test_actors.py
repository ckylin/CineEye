"""演员识别全链路测试：打标签导入、去重、划分、训练、识别、拒识。

测试用合成图片（纯色 + 噪声 + 圆形），只为验证**流程正确性**，
不代表真实人脸识别效果——真实效果取决于照片数量与质量。
因此这里统一关闭人脸检测（actors.face_detection=false），
避免 Haar 在合成图上漏检导致测试不稳定。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from cineeye.actors import ActorRecognizer
from cineeye.actors.dataset import add_photo_bytes, import_photos, read_manifest
from cineeye.actors.train import run_training
from cineeye.config import load_config, set_value

COLORS = {"演员甲": (30, 60, 200), "演员乙": (60, 200, 60), "演员丙": (200, 120, 30)}


def _make_photos(root, per_actor: int = 8, duplicate: bool = True):
    """生成 3 位"演员"的照片：每位一个主色 + 随机噪声，确保特征可分。"""
    rng = np.random.default_rng(7)
    for name, color in COLORS.items():
        d = root / name
        d.mkdir(parents=True)
        for i in range(per_actor):
            img = np.full((160, 160, 3), color, np.uint8)
            img = cv2.add(img, rng.integers(0, 40, (160, 160, 3), dtype=np.uint8))
            cv2.circle(img, (60 + i * 3, 70), 30 + i, (255, 255, 255), -1)
            cv2.imwrite(str(d / f"{i:03d}.jpg"), img)
    if duplicate:
        # 同一张图换个名字再放一份：验证按内容去重
        (root / "演员甲" / "copy.jpg").write_bytes((root / "演员甲" / "000.jpg").read_bytes())


@pytest.fixture
def cfg(tmp_path):
    """隔离的测试配置：所有产物都落在 tmp_path，不污染项目 outputs。"""
    c = load_config()
    set_value(c, "metadata.db", str(tmp_path / "test.db"))
    set_value(c, "actors.data_dir", str(tmp_path / "data"))
    set_value(c, "actors.manifest", str(tmp_path / "data" / "manifest.csv"))
    set_value(c, "actors.head_path", str(tmp_path / "actor_head.pt"))
    set_value(c, "actors.cache", str(tmp_path / "features.npz"))
    set_value(c, "actors.reference_dir", str(tmp_path / "ref"))
    set_value(c, "actors.face_detection", False)  # 合成图不走 Haar
    set_value(c, "actors.crop", "whole")
    set_value(c, "actors.image_size", 128)        # 降分辨率加快测试
    set_value(c, "actors.unknown_threshold", None)
    return c


def test_import_labels_dedup_and_split(tmp_path, cfg):
    """目录即标签：导入后去重、按演员分层划分、manifest 可回读。"""
    src = tmp_path / "photos"
    _make_photos(src, per_actor=8)

    summary = import_photos(
        src=src, dest_dir=tmp_path / "data", copy=True,
        store=None, cfg=cfg, val_ratio=0.25, test_ratio=0.25, seed=42,
    )
    # 3×8=24 张，另加 1 张重复文件 → 去重后仍是 24 张
    assert summary["images"] == 24
    assert summary["actors"] == 3
    assert summary["duplicates"] == 1

    # 每位演员都被切到 train/val/test，且总数守恒
    for name, dist in summary["per_actor"].items():
        assert dist["total"] == 8
        assert dist["train"] >= 1
        assert dist["val"] >= 1
        assert dist["test"] >= 1
        assert dist["train"] + dist["val"] + dist["test"] == 8

    records = read_manifest(tmp_path / "data" / "manifest.csv")
    assert len(records) == 24
    assert {r.split for r in records} == {"train", "val", "test"}
    assert all(r.person_id.startswith("person_") for r in records)


def test_csv_labels_import(tmp_path):
    """CSV 打标签：路径相对 CSV 所在目录解析，英文名一并登记。"""
    photos = tmp_path / "raw"
    photos.mkdir()
    for i in range(4):
        img = np.full((120, 120, 3), (20 * i, 100, 150), np.uint8)
        cv2.imwrite(str(photos / f"a{i}.jpg"), img)
    csv_path = tmp_path / "labels.csv"
    csv_path.write_text(
        "path,actor,name_en\n" + "".join(f"raw/a{i}.jpg,演员甲,Actress A\n" for i in range(4)),
        encoding="utf-8",
    )

    summary = import_photos(csv_path=csv_path, dest_dir=tmp_path / "data", copy=False,
                            store=None, cfg=cfg, val_ratio=0.0, test_ratio=0.0)
    assert summary["images"] == 4
    assert summary["actors"] == 1
    records = read_manifest(tmp_path / "data" / "manifest.csv")
    assert all(r.actor == "演员甲" for r in records)
    assert all(r.split == "train" for r in records)  # 比例设为 0，全部进 train


def test_add_photo_bytes_dedup(tmp_path, cfg):
    """上传单张照片：能入库、能打标签、同内容重复上传不重复计数。"""
    img = np.full((120, 120, 3), COLORS["演员甲"], np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    blob = buf.tobytes()

    r1 = add_photo_bytes("演员甲", blob, dest_dir=tmp_path / "data", cfg=cfg)
    assert r1["added"] is True
    assert r1["duplicate"] is False
    assert Path(r1["path"]).exists()

    # 同一张图再传一次 → 命中去重，manifest 不应增长
    r2 = add_photo_bytes("演员甲", blob, dest_dir=tmp_path / "data", cfg=cfg)
    assert r2["duplicate"] is True
    records = read_manifest(tmp_path / "data" / "manifest.csv")
    assert len(records) == 1
    assert records[0].actor == "演员甲"
    assert records[0].source == "upload"
    assert records[0].person_id.startswith("person_")


def test_train_and_recognize(tmp_path, cfg):
    """训练分类头 → 保存 checkpoint → 新图能被正确识别并带上 person_id。"""
    src = tmp_path / "photos"
    _make_photos(src, per_actor=8)
    import_photos(src=src, dest_dir=tmp_path / "data", copy=True,
                  store=None, cfg=cfg, val_ratio=0.25, test_ratio=0.25, seed=42)

    report = run_training(cfg, epochs=150, lr=0.02)
    assert report["num_classes"] == 3
    assert report["metrics"]["val"]["top1"] >= 0.9
    assert (tmp_path / "actor_head.pt").exists()
    # 检索库应只包含 train 划分的样本（每类 4 张，共 12）
    assert report["split_counts"]["train"] == 12

    rec = ActorRecognizer(cfg=cfg)
    assert "classifier" in rec.status()["mode"]
    assert rec.status()["num_actors"] == 3

    # 生成一张"演员乙"的新照片（训练时没出现过）
    rng = np.random.default_rng(99)
    img = np.full((160, 160, 3), COLORS["演员乙"], np.uint8)
    img = cv2.add(img, rng.integers(0, 30, (160, 160, 3), dtype=np.uint8))
    result = rec.recognize_all(img)
    assert result["num_faces"] == 1
    assert result["faces"][0]["actor"] == "演员乙"
    assert result["faces"][0]["candidates"][0]["probability"] is not None

    # 带电影反查的完整结果：person_id 已登记，但没有参演电影（测试库为空）
    full = rec.recognize_with_movies(img)
    assert full["name"] == "演员乙"
    assert full["person_id"]
    assert full["movies"] == []


def test_unknown_threshold_rejects(tmp_path, cfg):
    """阈值调高后，同一张脸应被判为"不认识"（actor=None，而不是硬猜一个）。"""
    src = tmp_path / "photos"
    _make_photos(src, per_actor=8)
    import_photos(src=src, dest_dir=tmp_path / "data", copy=True,
                  store=None, cfg=cfg, val_ratio=0.25, test_ratio=0.25, seed=42)
    run_training(cfg, epochs=150, lr=0.02)

    # 把阈值顶到 0.999：任何 softmax 概率都不可能通过，等同于"完全不认"
    set_value(cfg, "actors.unknown_threshold", 0.999)
    rec = ActorRecognizer(cfg=cfg)
    img = np.full((160, 160, 3), COLORS["演员甲"], np.uint8)
    result = rec.recognize_all(img)
    assert result["num_faces"] == 1
    assert result["faces"][0]["actor"] is None
    assert result["faces"][0]["confidence"] == 0.0
    assert rec.recognize_with_movies(img) is None
