"""启动影瞳推理 HTTP 服务。

用途：薄启动器。真正的服务实现（路由、模型单例、CORS、请求体限制）在
cineeye/serve.py，这里只负责把项目根加入 sys.path 并转调 serve.main()。
之所以单独留一个脚本：运维习惯用 python scripts/xx.py 统一调用，
且避免运维必须记住 "python -m cineeye.serve" 这种模块式入口。

典型调用：
    python scripts/06_serve.py --port 8080
    python scripts/06_serve.py --port 8080 --checkpoint outputs/checkpoints/best.pt --host 127.0.0.1
    python scripts/06_serve.py --port 8080 --with-identifier   # 同时启用档案检索/演员接口

输入/输出：
    读取：权重（默认 outputs/checkpoints/best.pt）、configs/default.yaml、
          outputs/cineeye.db、outputs/gallery/*（--with-identifier 时才加载）
    输出：仅服务日志（stdout）；常驻进程，不写业务文件

运维要点：
    1. 默认监听 0.0.0.0:8080，内网可直接访问；对外暴露务必改 --host 127.0.0.1
       并前置反向代理做鉴权（服务本身没有认证）。
    2. 所有参数原样交给 cineeye.serve.main() 解析，接口文档见 docs/API_GUIDE.md。
    3. 长驻进程建议用 systemd/nohup 托管，并把 TMPDIR 指到项目 .tmp
       （本机 /tmp 只有 10MB）。
"""

import sys
from pathlib import Path

# 直接执行脚本时 sys.path[0] 是 scripts/，先补项目根，后面的 import 才能成功
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cineeye.serve import main  # noqa: E402

if __name__ == "__main__":
    # main() 内部解析 --host/--port/--checkpoint/--config/--device/--with-identifier
    main()
