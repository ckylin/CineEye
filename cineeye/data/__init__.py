"""数据层：视频抽帧、标注清单、数据集与 DataLoader。

用途：对外提供数据链路的统一入口，上层代码只 ``from cineeye.data import ...``，
不必关心功能具体落在 extract_frames / manifest / dataset 哪个文件。

被谁使用：
    - ``cineeye/train.py``、``cineeye/evaluate.py``、``cineeye/features.py``
      通过本包导入数据集与清单构造；
    - ``scripts/01_extract_frames.py``、``scripts/02_build_manifest.py`` 等脚本。

依赖：本包不引入额外依赖，全部转嫁给子模块（torch / pandas / cv2）。

运维要点：这里只做「再导出」，请不要在 ``__init__`` 里写任何有副作用的代码
（建目录、读配置、探测设备等）。导入 ``cineeye.data`` 必须是零成本的，
否则 CLI 启动、单测收集阶段会被拖慢，还可能在没有 GPU 的环境里报错。
"""

from .dataset import FrameDataset, build_dataloader, build_transforms, load_manifest
from .manifest import build_manifest, group_split, scan_frames

__all__ = [
    "FrameDataset", "build_dataloader", "build_transforms", "load_manifest",
    "build_manifest", "group_split", "scan_frames",
]
