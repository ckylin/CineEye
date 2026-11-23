"""演员识别：照片打标签 → 训练分类头 → 检测/编码/融合识别。

用途：对外导出演员识别全链路组件：
    - FaceDetector / FaceBox：OpenCV Haar 人脸检测与裁剪；
    - ActorEncoder：演员特征编码（ImageNet 或电影主干）；
    - ActorRecognizer：分类头 + 参考库检索融合，支持多脸与 unknown 拒识；
    - ActorHead / run_training：训练侧入口（供 16_train_actors.py 使用）。
被谁使用：identify.py（actor 字段）、scripts/13_actor_reference.py、
    scripts/15_import_actors.py、scripts/16_train_actors.py。
依赖：OpenCV、torch；metadata.MetaStore。
运维要点：
    1. 只 import 轻量类时不会触发训练模块（train 在函数内才用到 torch 的
       训练相关逻辑），但 actors.train 会 import torch，属正常依赖；
    2. 演员特征与电影图库特征是两条独立通道（向量空间不同），不要混用。
"""

from .dataset import PhotoRecord, add_photo_bytes, import_photos, read_manifest, split_records
from .encoder import ActorEncoder
from .face import FaceBox, FaceDetector
from .recognizer import ActorRecognizer
from .train import ActorHead, run_training

__all__ = [
    "FaceDetector", "FaceBox", "ActorRecognizer", "ActorEncoder", "ActorHead",
    "PhotoRecord", "import_photos", "add_photo_bytes", "read_manifest",
    "split_records", "run_training",
]
