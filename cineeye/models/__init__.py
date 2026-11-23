"""模型层：PyTorch 主干 / 分类头，以及 scikit-learn 基线。

用途：统一模型层的导入入口。注意这里**只导出深度模型相关符号**，
``cineeye.models.classical``（HOG + sklearn 基线）需要显式导入子模块，
因为 sklearn/joblib 属于可选依赖，不希望每次 ``import cineeye.models``
都把 sklearn 拉进来（会显著拖慢 CLI 启动）。

被谁使用：
    - ``cineeye/train.py``、``features.py``、``predict.py``、``evaluate.py``；
    - ``cineeye/gallery/encoder.py``（复用主干做向量编码）；
    - ``scripts/00_check_env.py``（环境自检）。

依赖：torch、torchvision（经由 :mod:`.cnn`）。

运维要点：新增主干或模型时，除了改 ``cnn.SUPPORTED_BACKBONES``，
还要确认 ``configs/default.yaml`` 的注释与文档同步更新，避免运维照配置改却报错。
"""

from .cnn import CineEyeNet, SUPPORTED_BACKBONES, build_backbone, build_model

__all__ = ["CineEyeNet", "SUPPORTED_BACKBONES", "build_backbone", "build_model"]
