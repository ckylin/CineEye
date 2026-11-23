"""轻量单元测试（不需要真实数据，也不下载预训练权重）。"""

import numpy as np
import pandas as pd
import torch

from cineeye.config import ROOT, get_value, load_config, path
from cineeye.data.dataset import FrameDataset, build_transforms
from cineeye.data.manifest import group_split
from cineeye.models.classical import image_to_features
from cineeye.models.cnn import CineEyeNet, SUPPORTED_BACKBONES


def test_config_override():
    cfg = load_config(overrides=["train.epochs=7", "model.pretrained=false", "inference.top_k=3"])
    assert get_value(cfg, "train.epochs") == 7
    assert get_value(cfg, "model.pretrained") is False
    assert get_value(cfg, "inference.top_k") == 3
    assert path(cfg, "data.manifest").is_absolute()
    assert path(cfg, "data.manifest").parent == ROOT / "data"


def test_group_split_has_no_leakage():
    rows = []
    for label in ["A", "B"]:
        for v in range(6):
            for f in range(5):
                rows.append({"frame_path": f"{label}/v{v}/{f}.jpg", "label": label,
                             "video_id": f"{label}/v{v}", "frame_index": str(f)})
    df = pd.DataFrame(rows)
    df["split"] = group_split(df, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=42)
    assert set(df["split"]) == {"train", "val", "test"}
    assert (df.groupby("video_id")["split"].nunique() == 1).all()
    assert df[df["split"] == "test"]["label"].nunique() == 2


def test_transforms_shape():
    cfg = load_config()
    from PIL import Image

    img = Image.fromarray(np.full((180, 320, 3), 128, dtype=np.uint8))
    for train in (True, False):
        t = build_transforms(cfg, train=train)(img)
        assert t.shape == (3, 224, 224)
        # 常量灰度图经过 ImageNet 逐通道 Normalize 后，三通道均值必然不同。
        # 不能断言 min() < 0：128/255 恰好比三个通道的均值都大，归一化后全为正。
        means = [float(t[c].mean()) for c in range(3)]
        assert max(means) - min(means) > 0.1


def test_model_forward_random_weights():
    model = CineEyeNet("resnet18", num_classes=3, pretrained=False)
    model.freeze_backbone(True)
    model.eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        logits = model(x)
        feats = model(x, return_feature=True)
    assert logits.shape == (2, 3)
    assert feats.shape == (2, 512)
    assert not any(p.requires_grad for p in model.backbone.parameters())
    assert all(p.requires_grad for p in model.head.parameters())
    assert "resnet18" in SUPPORTED_BACKBONES


def test_handcrafted_features():
    img = np.random.default_rng(0).integers(0, 255, (128, 200, 3), dtype=np.uint8)
    f = image_to_features(img, use_hog=True)
    assert f.dtype == np.float32
    assert f.ndim == 1 and len(f) > 2000
    assert np.isfinite(f).all()


def test_frame_dataset_fallback_on_broken_image():
    cfg = load_config()
    df = pd.DataFrame([{"frame_path": "__not_exist__.jpg", "label_idx": 0, "video_id": "x/y"}])
    ds = FrameDataset(df, transform=build_transforms(cfg, train=False), root=ROOT)
    tensor, label, vid = ds[0]
    assert tensor.shape == (3, 224, 224)
    assert label == 0 and vid == "x/y"
