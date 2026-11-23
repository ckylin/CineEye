"""扩展模块单元测试：元数据库、别名归一化、图库索引、演员检测、统一识别。"""

import numpy as np
import pytest

from cineeye.config import load_config
from cineeye.metadata import MetaStore
from cineeye.metadata.normalize import normalize, similarity
from cineeye.metadata.providers.local_table import LocalTableProvider
from cineeye.actors.face import FaceDetector


@pytest.fixture
def cfg():
    return load_config()


def test_normalize_aliases():
    assert normalize("愛的色放") == normalize("爱的色放")
    assert normalize("Summer.Time.2001", punct=True, case=True) == "summertime2001"
    assert similarity("Summertime", "Summer Time") > 0.8


def test_meta_store_alias_and_actor(tmp_path):
    store = MetaStore(db_path=tmp_path / "test.db")
    store.upsert_movie(
        "m1", "主片名", original_title="Original Name", year=2000,
        aliases=[
            {"alias": "异形别名", "lang": "zh", "source": "manual"},
            {"alias": "Alias English", "lang": "en", "source": "tmdb"},
        ],
        cast=[{"name": "演员甲", "role": "actor"}],
        provider="test",
    )
    assert store.count_movies() == 1
    assert any(h["movie_id"] == "m1" for h in store.search("异形别名"))
    assert any(h["movie_id"] == "m1" for h in store.search("Alias English"))
    m = store.get_movie("m1")
    assert len(m["aliases"]) >= 3  # 主片名 + 原名 + 2 别名中至少 3 条
    assert len(m["cast"]) == 1
    assert len(store.list_actors()) == 1


def test_meta_store_empty_search(tmp_path):
    store = MetaStore(db_path=tmp_path / "test.db")
    assert store.search("") == []
    assert store.stats()["movies"] == 0


def test_local_provider_csv(tmp_path):
    import csv
    import json

    path = tmp_path / "movies.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["movie_id", "primary_title", "year", "aliases", "cast"])
        w.writerow([
            "m2", "测试电影", 2020,
            json.dumps([{"alias": "别名X", "lang": "zh"}], ensure_ascii=False),
            json.dumps([{"name": "演员乙"}], ensure_ascii=False),
        ])
    recs = LocalTableProvider(path=path).fetch_all()
    assert len(recs) == 1
    assert recs[0].movie_id == "m2"
    assert recs[0].year == 2020
    assert recs[0].aliases[0]["alias"] == "别名X"
    assert recs[0].cast[0]["name"] == "演员乙"


def test_face_detector_init(cfg):
    d = FaceDetector(cfg)
    assert not d.cascade.empty()
    blank = np.full((100, 100, 3), 128, dtype=np.uint8)
    assert d.detect(blank) == []


def test_gallery_index_roundtrip(tmp_path, cfg):
    from cineeye.gallery import GalleryIndex

    idx = GalleryIndex(cfg=cfg, gallery_dir=tmp_path / "gallery")
    assert idx.search(np.ones(512, dtype=np.float32), top_k=3) == []
    assert idx.stats()["images"] == 0
