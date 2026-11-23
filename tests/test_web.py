"""Web 资产层与标签契约测试。

覆盖：标签契约可解析、validate_tags 校验与规范化、AssetStore 入库/去重/
补标/派生写回/老数据导入、JobManager 异步任务与进度。所有测试用临时库，
不碰真实 outputs/cineeye.db。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from cineeye.assets import (
    AssetStore, KIND_DEFAULT_PURPOSE, load_tag_schema, validate_tags,
)
from cineeye.config import ROOT


@pytest.fixture
def store(tmp_path):
    return AssetStore(db_path=tmp_path / "assets.db")


def make_image(path: Path, color=(40, 80, 200)) -> Path:
    img = np.full((120, 160, 3), color, np.uint8)
    cv2.imwrite(str(path), img)
    return path


# ---------------------------------------------------------------- 契约
def test_tag_schema_valid():
    """契约含五层、关键枚举与默认映射齐全。"""
    s = load_tag_schema()
    assert s["version"]
    assert len(s["layers"]) == 5
    assert set(s["enums"]["media_kind"]) >= {"frame", "poster", "actor_portrait", "screenshot"}
    assert s["media_kind_defaults"]["poster"] == "gallery"


def test_validate_tags():
    """非法枚举被拒；bool/number/tags 规范化正确。"""
    clean, errors = validate_tags({"purpose": "train", "face_ok": "true",
                                   "free_tags": "a, b"})
    assert not errors
    assert clean["face_ok"] == 1
    assert json.loads(clean["free_tags"]) == ["a", "b"]

    _, errors = validate_tags({"media_kind": "不存在的类型"})
    assert errors


# ---------------------------------------------------------------- 资产层
def test_add_dedup_and_purpose(tmp_path, store):
    """同内容去重；purpose 按 media_kind 自动推导。"""
    p = make_image(tmp_path / "a.jpg")
    a1 = store.add(p, tags={"media_kind": "screenshot"})
    a2 = store.add(p, tags={"media_kind": "poster"})  # 同内容、不同标签
    assert a1["asset_id"] == a2["asset_id"]
    assert a1["purpose"] == KIND_DEFAULT_PURPOSE["screenshot"]
    assert a1["width"] == 160 and a1["height"] == 120


def test_update_relations_and_derived(tmp_path, store):
    """补多电影/多演员关联，派生写回 movie_images。"""
    # 先建两部电影与一个演员（外键约束要求存在）
    with store._conn() as c:
        c.execute("INSERT INTO movies (id,primary_title) VALUES ('m1','A')")
        c.execute("INSERT INTO movies (id,primary_title) VALUES ('m2','B')")
        c.execute("INSERT INTO persons (id,name) VALUES ('p1','演员甲')")
    p = make_image(tmp_path / "poster.jpg")
    a = store.add(p, tags={"media_kind": "poster"})
    store.update(a["asset_id"], {"movie_ids": ["m1", "m2"], "person_ids": ["p1"],
                                 "notes": "联调"})
    got = store.get(a["asset_id"])
    assert set(got["movie_ids"]) == {"m1", "m2"}
    assert got["person_ids"] == ["p1"]
    with store._conn() as c:
        n = c.execute("SELECT COUNT(*) FROM movie_images WHERE path=?", (got["file_path"],)).fetchone()[0]
    assert n == 2  # 派生到两部电影


def test_list_filter_and_stats(tmp_path, store):
    make_image(tmp_path / "p.jpg", (1, 2, 3))
    make_image(tmp_path / "f.jpg", (9, 8, 7))
    store.add(tmp_path / "p.jpg", tags={"media_kind": "poster"})
    store.add(tmp_path / "f.jpg", tags={"media_kind": "frame"})
    assert len(store.list(kind="poster")) == 1
    assert len(store.list(kind="frame")) == 1
    assert store.stats()["total"] == 2


# ---------------------------------------------------------------- 任务
def test_job_submit_and_progress(tmp_path):
    """异步任务提交后立即返回，最终完成（refresh_derived 无需子进程）。"""
    from cineeye.jobs import JobManager
    jm = JobManager(db_path=tmp_path / "jobs.db")
    tid = jm.submit("refresh_derived", {})
    assert tid.startswith("job_")
    for _ in range(100):
        st = jm.get(tid)
        if st["status"] in ("done", "failed"):
            break
        time.sleep(0.02)
    assert st["status"] == "done"
    assert st["progress"] == 100
    with pytest.raises(ValueError):
        jm.submit("unknown_kind")
