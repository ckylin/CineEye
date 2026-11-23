"""影瞳 · 推理 HTTP 服务（零新增依赖，基于标准库 http.server）。

用途：把训练好的封闭集分类模型（Predictor）与开放集档案检索（Identifier）
暴露为 JSON over HTTP 的 REST 接口，供前端页面、外部程序或其他服务调用。
之所以选标准库 http.server + ThreadingHTTPServer 而不用 FastAPI/Flask：
生产环境是内网、单机、低并发场景，标准库零额外依赖（pip 装不上/离线机器也能跑），
升级 Python 不需要跟着升级 Web 框架；代价是没有自动 OpenAPI、没有异步与限流，
若将来要做公网高并发服务，再换成 FastAPI 也只是替换这一层的 Handler。

典型调用：
    python -m cineeye.serve --port 8080
    python scripts/06_serve.py --port 8080 --checkpoint outputs/checkpoints/best.pt
    python scripts/06_serve.py --port 8080 --with-identifier   # 同时启用扩展接口

输入/输出：
    读取：--checkpoint 权重（默认 outputs/checkpoints/best.pt）、configs/default.yaml、
          outputs/cineeye.db（元数据库）、outputs/gallery/*（图库向量索引）、
          outputs/actors/*（演员参考库）
    输出：仅标准输出日志，不落盘任何文件；模型全部常驻内存。

运维要点：
    1. 模型在进程内是模块级单例（_PREDICTOR/_IDENTIFIER），只加载一次；
       多进程部署（gunicorn -w N）会各加载一份，注意显存/内存翻倍。
    2. _DEVICE 必须透传给 Identifier，否则 GPU 机器上会出现
       「分类器在 GPU、图库编码在 CPU」的设备割裂，速度慢一半。
    3. 请求体上限由环境变量 CINEYE_MAX_BODY_MB 控制（默认 64MB）；
       base64 会把图片体积放大 ~33%，大图/多图请求要相应调大。
    4. 本机 /tmp 与 /dev/shm 只有 10MB，视频解码等临时写入需依赖
       TMPDIR 指向项目 .tmp（由上层脚本设置），不要在此处解压大文件。

基础接口（封闭集分类）：
    GET  /health    健康检查
    GET  /labels    类别列表（编号 → 名称）
    POST /predict   识别一张或多张图片 / 一段视频
    POST /predict/image  仅图片
    POST /predict/video  仅视频

扩展接口（开放集电影档案检索）：
    POST /identify        统一识别（图片/视频/base64）→ 完整电影档案（多片名/演员）
    GET  /search?q=...    按片名/别名搜索（支持多语言、模糊）
    GET  /movies          电影列表（分页）
    GET  /movies/{id}     单部电影完整档案
    GET  /actors/{id}     演员及其参演电影
    GET  /gallery/stats   图库状态

启动方式见上方「典型调用」；也可直接 python -m cineeye.serve（main 在文件末尾）。

请求体（JSON）示例：
    {"image": "/abs/path/to/frame.jpg"}
    {"images": ["frame1.jpg", "frame2.jpg"]}
    {"image_base64": "<base64 字符串>"}
    {"video": "/abs/path/to/clip.mp4", "num_frames": 16, "aggregation": "mean_prob"}
    {"image": "...", "kind": "poster", "with_actors": false}   # /identify
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import cv2
import numpy as np

from .assets import AssetStore, load_tag_schema
from .config import ROOT, get_value, load_config
from .predict import Predictor
from .utils import ensure_dir, get_logger

log = get_logger("cineeye.serve")

# 模块级单例，启动时加载一次模型（加载权重较慢，避免每个请求重复加载）
# 说明：http.server 每个请求都在同一进程的不同线程里，全局变量天然可共享；
# Predictor 内部是 eval + no_grad 推理，只读权重，多线程并发读是安全的。
_PREDICTOR: Predictor | None = None
_IDENTIFIER = None  # cineeye.identify.Identifier，惰性初始化
_ACTOR_REC = None   # cineeye.actors.ActorRecognizer，演员识别独立单例（不依赖 best.pt）
_DEVICE: str | None = None  # 启动参数 --device，供惰性初始化的 Identifier 复用
# Web 管理后台的三个单例：资产库、任务管理器、前端静态目录
_ASSET_STORE: AssetStore | None = None
_JOB_MANAGER = None
_STATIC_DIR = ROOT / "static"  # Vite 构建产物输出目录（见 web/vite.config.js）
# 请求体硬上限：先于 rfile.read() 校验，防止超大 POST 把内存吃光（简单 DoS 防护）。
# 默认 64MB，可按部署需要调 CINEYE_MAX_BODY_MB；视频上传走 base64 时会放大 ~33%，
# 大视频建议用 /admin/upload 的 multipart 分片走流式接口（见 _handle_admin_upload）。
_MAX_BODY = int(os.environ.get("CINEYE_MAX_BODY_MB", "256")) * 1024 * 1024
# 上传文件大小上限（默认 2GiB，环境变量 CINEYE_MAX_UPLOAD_MB）
_MAX_UPLOAD = int(os.environ.get("CINEYE_MAX_UPLOAD_MB", "2048")) * 1024 * 1024


def get_predictor() -> Predictor:
    """取已加载的分类器；未初始化时直接抛错（服务未正确启动时早失败、不吞异常）。"""
    global _PREDICTOR
    if _PREDICTOR is None:
        raise RuntimeError("模型尚未初始化")
    return _PREDICTOR


def init_predictor(checkpoint: str | Path | None = None, cfg: dict | None = None,
                   device: str | None = None) -> Predictor:
    """进程启动时调用一次：建全局 Predictor，并记下 --device 供后续扩展服务复用。

    device 留 None 时由 Predictor 自己按 cfg.train.device 解析（auto 会自动选 cuda/cpu）。
    """
    global _PREDICTOR, _DEVICE
    _DEVICE = device
    _PREDICTOR = Predictor(checkpoint=checkpoint, cfg=cfg, device=device)
    return _PREDICTOR


def get_actor_recognizer():
    """演员识别器单例（惰性）：只依赖 actors.* 配置与演员数据，不需要电影模型。

    与 get_identifier() 分开的原因：只部署"认演员"能力时，机器上可能根本
    没有 best.pt（电影分类模型），此时 Identifier 构造会失败，而演员识别
    仍应可用。两者各持一个编码器实例，代价是几百 MB 内存，换来部署解耦。
    """
    global _ACTOR_REC
    if _ACTOR_REC is None:
        from .actors import ActorRecognizer
        cfg = load_config()
        if _DEVICE:
            from .config import set_value
            set_value(cfg, "train.device", _DEVICE)
        _ACTOR_REC = ActorRecognizer(cfg=cfg)
    return _ACTOR_REC


def get_identifier():
    """扩展服务：电影档案识别器，首次调用时加载（复用 --device 参数）。

    惰性初始化的原因：Identifier 会依次加载 元数据库 + 图库向量 + 演员库 + 分类器，
    不做扩展识别的部署没必要付这份启动开销；只有第一次访问扩展接口时才构建。
    注意必须把 _DEVICE 传进去：Identifier 内部用同一 device 建图库编码器与分类器，
    否则 GPU 机器上会出现图库走 CPU、分类器走 GPU 的割裂。
    """
    global _IDENTIFIER
    if _IDENTIFIER is None:
        from .identify import Identifier
        _IDENTIFIER = Identifier(cfg=load_config(), device=_DEVICE)
    return _IDENTIFIER


def init_identifier(identifier) -> None:
    """外部注入一个已构建好的 Identifier（--with-identifier 预热时用，省去首个请求的等待）。"""
    global _IDENTIFIER
    _IDENTIFIER = identifier


def get_asset_store() -> AssetStore:
    """Web 后台资产库单例（惰性）。独立于电影模型，后台管理不依赖 best.pt。"""
    global _ASSET_STORE
    if _ASSET_STORE is None:
        _ASSET_STORE = AssetStore(cfg=load_config())
    return _ASSET_STORE


def get_job_manager():
    """Web 后台异步任务管理器单例（惰性）。"""
    global _JOB_MANAGER
    if _JOB_MANAGER is None:
        from .jobs import JobManager
        _JOB_MANAGER = JobManager()
    return _JOB_MANAGER


def _decode_image_base64(data: str) -> np.ndarray:
    """支持纯 base64 或 data URL（data:image/jpeg;base64,xxx）。

    返回 BGR ndarray（cv2.imdecode 的输出），与 cv2.imread 的通道顺序一致，
    这样下游 Predictor/Identifier 不必区分「来自路径」还是「来自内存」。
    """
    # 只在前 64 字符里找逗号：避免把超长 base64 整体 startswith 判断拖慢
    if "," in data[:64] and data.strip().startswith("data:"):
        data = data.split(",", 1)[1]
    buf = base64.b64decode(data)
    arr = np.frombuffer(buf, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("image_base64 不是可解码的图片")
    return img


def _resolve_path(p: str) -> Path:
    """把请求里的相对路径按项目根目录展开；支持 ~。

    服务端只接受本机路径（内网工具性质），不做 URL 抓取，避免 SSRF。
    """
    path = Path(p).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path


class Handler(BaseHTTPRequestHandler):
    """单次请求处理器：负责收包、路由分发、JSON 响应与错误码映射。

    ThreadingHTTPServer 会为每个连接起一个线程，因此这里不要存任何实例级
    可变状态——模型与索引都放在模块级全局变量里共享。
    """

    server_version = "CineEye/1.0"

    # ------------------------------------------------------------ 基础设施
    def _send(self, status: int, payload: Any) -> None:
        """统一的 JSON 出口：写状态行 + 头 + 体，并附带 CORS 头。

        CORS 全放开（Access-Control-Allow-Origin: *）是因为本项目定位为
        内网/本机演示服务，前端页面（含 file:// 打开的静态页）要能直接调；
        若对外暴露，务必改成白名单域名，否则任意站点都能借用户浏览器调用。
        """
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        # Content-Length 必须准确：Keep-Alive 下浏览器按它判断响应结束
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PATCH, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def _ok(self, data: Any) -> None:
        """成功响应统一外壳 {"ok": true, "data": ...}，前端只需判 ok。"""
        self._send(200, {"ok": True, "data": data})

    def _error(self, status: int, msg: str) -> None:
        """失败响应统一外壳 {"ok": false, "error": "..."}；msg 直接来自异常文本。"""
        self._send(status, {"ok": False, "error": msg})

    def _read_json(self) -> dict:
        """读取并校验 JSON 请求体。

        这里先看 Content-Length 再读，是为了在真正分配内存之前就拦住超大请求；
        没有 Content-Length（如 chunked）时按 0 处理，直接判为非法，服务端不做流式解析。
        """
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            raise ValueError("请求体为空，需要 JSON")
        if length > _MAX_BODY:
            raise ValueError(f"请求体超过 {_MAX_BODY // 1024 // 1024}MB 限制")
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("请求体不是合法 JSON") from exc
        if not isinstance(data, dict):
            raise ValueError("JSON 顶层必须是对象")
        return data

    def _serve_static(self, rel: str) -> bool:
        """托管 Vite 构建产物（纯静态文件）。返回是否成功处理。

        rel 为 URL 路径（如 /assets/index-abc.js 或 /admin）。规则：
        - 先按文件实路径尝试；不存在且无扩展名时回退 index.html（支持前端路由 /admin）；
        - 严格路径校验：拼接后必须仍在 _STATIC_DIR 内，防目录穿越；
        - 未构建前端（static/ 不存在）时返回 False，让路由层回退到 JSON 服务发现。
        """
        if not _STATIC_DIR.exists():
            return False
        rel = unquote(rel).lstrip("/")
        if rel in ("", "admin", "index.html"):
            target = _STATIC_DIR / "index.html"
        else:
            target = (_STATIC_DIR / rel).resolve()
            # 防目录穿越：解析后的真实路径必须在静态目录内
            if not str(target).startswith(str(_STATIC_DIR.resolve())):
                self._error(403, "禁止访问")
                return True
        if not target.is_file() and "." not in Path(rel).name:
            target = _STATIC_DIR / "index.html"  # 前端路由回退
        if not target.is_file():
            return False
        ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)
        return True

    def log_message(self, fmt: str, *args) -> None:
        """接管标准库默认的 stderr 裸打印，改走项目 logger（统一格式与级别）。"""
        log.info("%s - %s", self.address_string(), fmt % args)

    # ------------------------------------------------------------ 路由
    def do_OPTIONS(self) -> None:  # noqa: N802
        """浏览器跨域预检（PUT/POST + JSON 会先发 OPTIONS），返回空 200 即可。"""
        self._send(200, {})

    def do_GET(self) -> None:  # noqa: N802
        """GET 路由表：健康检查、类别、元数据检索、图库状态。

        用 if/elif 顺序匹配而不是正则路由，是因为路径只有十来个、
        且 GET 的路径都很短；复杂匹配交给 startswith 前后缀即可。
        ValueError → 400（参数问题），其他异常 → 500（服务端问题）。
        """
        parsed = urlparse(self.path)
        # 统一去掉结尾斜杠，让 /movies/ 与 /movies 走同一分支（根路径单独兜底）
        raw_path = parsed.path
        path = raw_path.rstrip("/") or "/"
        try:
            # 前端页面优先：已构建静态产物时，/ 和 /admin 及 /assets/* 走静态托管
            if path in ("/", "/admin") or path.startswith("/assets/"):
                if self._serve_static(raw_path):
                    return
            if path == "/api":
                # JSON 版服务发现（前端页面占用 / 后，机器自检改打 /api）
                self._ok({"service": "影瞳 CineEye", "version": "3.0",
                          "ui": {"user": "/", "admin": "/admin"},
                          "endpoints": ["/health", "/labels", "/predict", "/predict/video",
                                        "/identify", "/search", "/movies", "/movies/{id}",
                                        "/actors", "/actors/recognize", "/actors/photos",
                                        "/actors/{id}", "/gallery/stats", "/tag-schema",
                                        "/admin/assets", "/admin/train", "/admin/tasks"]})
            elif path == "/tag-schema":
                # 标签契约：React 表单据此动态生成，与后端校验同源
                self._ok(load_tag_schema())
            elif path == "/admin/assets":
                qs = parse_qs(parsed.query)
                self._ok(get_asset_store().list(
                    kind=(qs.get("kind") or [None])[0],
                    purpose=(qs.get("purpose") or [None])[0],
                    q=(qs.get("q") or [None])[0],
                    status=(qs.get("status") or ["active"])[0],
                    limit=int((qs.get("limit") or ["100"])[0]),
                    offset=int((qs.get("offset") or ["0"])[0]),
                ))
            elif path.startswith("/admin/assets/"):
                aid = path[len("/admin/assets/"):]
                asset = get_asset_store().get(aid)
                if not asset:
                    self._error(404, f"素材不存在：{aid}")
                else:
                    self._ok(asset)
            elif path == "/admin/tasks":
                qs = parse_qs(parsed.query)
                self._ok({"tasks": get_job_manager().list(
                    limit=int((qs.get("limit") or ["50"])[0]),
                    offset=int((qs.get("offset") or ["0"])[0]))})
            elif path.startswith("/admin/tasks/"):
                tid = path[len("/admin/tasks/"):]
                task = get_job_manager().get(tid)
                if not task:
                    self._error(404, f"任务不存在：{tid}")
                else:
                    self._ok(task)
            elif path == "/admin/status":
                # 状态中心：模型/图库/元数据/演员/素材/任务 汇总
                self._ok(self._admin_status())
            elif path == "/health":
                # 健康检查顺带报告模型元信息，便于确认「跑的是哪个权重/哪块设备」
                p = get_predictor()
                data = {"status": "ok", "model_loaded": True,
                        "num_classes": p.num_classes, "backbone": p.mode,
                        "device": str(p.device)}
                try:
                    # 扩展服务可能没启用/没建索引，这里失败不能让健康检查挂掉
                    data["identifier"] = get_identifier().status()
                except Exception:  # noqa: BLE001 - 扩展服务未启用不影响健康
                    data["identifier"] = None
                self._ok(data)
            elif path == "/labels":
                # 同时给出 name→idx 与 idx→name，两种遍历方向前端都用得上
                p = get_predictor()
                self._ok({"label_map": p.label_map,
                          "labels": [p.idx2label[i] for i in range(p.num_classes)]})
            elif path == "/search":
                # 开放集：按片名/别名模糊搜索（多语言，见 metadata.normalize）
                q = (parse_qs(parsed.query).get("q") or [""])[0]
                if not q:
                    raise ValueError("需要提供查询参数 q")
                limit = int((parse_qs(parsed.query).get("top_k") or ["10"])[0])
                self._ok({"query": q, "results": get_identifier().search_movies(q, top_k=limit)})
            elif path == "/movies":
                # 分页列举；total 单独查一次 count，前端才能算页码
                qs = parse_qs(parsed.query)
                limit = int((qs.get("limit") or ["100"])[0])
                offset = int((qs.get("offset") or ["0"])[0])
                ident = get_identifier()
                self._ok({"total": ident.store.count_movies(),
                          "movies": ident.list_movies(limit=limit, offset=offset)})
            elif path.startswith("/movies/"):
                # 单部档案：多片名 / 演员 / 年份 / 海报等
                mid = path[len("/movies/"):]
                movie = get_identifier().get_movie(mid)
                if not movie:
                    self._error(404, f"电影不存在：{mid}")
                else:
                    self._ok(movie)
            elif path == "/actors":
                # 演员名册：persons 表 + 当前识别器状态（模型是否加载、支持哪些人）
                rec = get_actor_recognizer()
                limit = int((parse_qs(parsed.query).get("limit") or ["100"])[0])
                actors = rec.store.list_actors(limit=limit)
                self._ok({"total": len(actors), "actors": actors,
                          "recognizer": rec.status()})
            elif path.startswith("/actors/"):
                # 演员档案：个人信息 + 参演电影（person_id 由元数据库生成）
                aid = path[len("/actors/"):]
                actor = get_actor_recognizer().store.get_actor(aid)
                if not actor:
                    self._error(404, f"演员不存在：{aid}")
                else:
                    self._ok(actor)
            elif path == "/gallery/stats":
                # 图库向量索引状态：图片数、维度、编码器指纹、更新时间
                self._ok(get_identifier().gallery.stats())
            elif path in ("/", "/admin"):
                # 到这里说明前端静态产物未构建，回退到 JSON 提示（引导先 npm run build）
                self._ok({"service": "影瞳 CineEye", "version": "3.0",
                          "message": "Web 前端尚未构建，请先在 web/ 目录执行 npm install && npm run build",
                          "api": "/api", "tag_schema": "/tag-schema"})
            else:
                self._error(404, "未知接口")
        except ValueError as exc:
            self._error(400, str(exc))
        except Exception as exc:  # noqa: BLE001
            # 兜底 500：不打印堆栈到响应里，但日志里保留完整 traceback 便于排查
            self._error(500, str(exc))

    def do_POST(self) -> None:  # noqa: N802
        """POST 路由表：分类（/predict*）与开放集档案识别（/identify）。

        异常到状态码的映射是约定：ValueError→400（参数缺失/格式错）、
        FileNotFoundError→404（路径写了但文件不存在）、其余→500。
        """
        path = urlparse(self.path).path.rstrip("/")
        try:
            data = self._read_json()
            if path in ("/predict", "/predict/image"):
                # /predict 兼容老客户端：与 /predict/image 行为一致（也是彩蛋式兼容，
                # 若将来 /predict 要支持视频，需同步改这里与文档）
                self._handle_image(data)
            elif path == "/predict/video":
                self._handle_video(data)
            elif path == "/identify":
                self._handle_identify(data)
            elif path == "/actors/recognize":
                # 只做演员识别（不需要电影模型）：合照/生活照/剧照都能传
                self._handle_actors(data)
            elif path == "/actors/photos":
                # 上传一张演员照片并打标签入库（增量，不会重排已有划分）
                self._handle_actor_upload(data)
            elif path == "/admin/upload":
                # Web 后台上传：图片/视频（base64 或本地路径）→ assets 入库
                self._handle_admin_upload(data)
            elif path.startswith("/admin/assets/"):
                # 补标：路径末段是素材 ID（PATCH 语义用 POST + 路径表达，兼容不支持 PATCH 的客户端）
                if path.endswith("/archive"):
                    aid = path[len("/admin/assets/"):-len("/archive")]
                    self._ok({"archived": get_asset_store().archive(aid)})
                else:
                    aid = path[len("/admin/assets/"):]
                    self._ok(get_asset_store().update(aid, data))
            elif path == "/admin/train":
                # 触发训练：异步，立即返回 task_id
                kind = data.get("kind")
                if not kind:
                    raise ValueError("需要 kind 字段")
                payload = {k: v for k, v in data.items() if k != "kind"}
                tid = get_job_manager().submit(kind, payload)
                self._ok({"task_id": tid, "status": "queued"})
            elif path.startswith("/admin/tasks/") and path.endswith("/cancel"):
                tid = path[len("/admin/tasks/"):-len("/cancel")]
                self._ok({"canceled": get_job_manager().cancel(tid)})
            else:
                self._error(404, "未知接口")
        except ValueError as exc:
            self._error(400, str(exc))
        except FileNotFoundError as exc:
            self._error(404, str(exc))
        except Exception as exc:  # noqa: BLE001
            log.exception("推理请求失败")
            self._error(500, str(exc))

    def do_PATCH(self) -> None:  # noqa: N802
        """PATCH /admin/assets/{id}：补标/改标，走与 POST 相同的 update 逻辑。"""
        path = urlparse(self.path).path.rstrip("/")
        try:
            if path.startswith("/admin/assets/"):
                aid = path[len("/admin/assets/"):]
                data = self._read_json()
                self._ok(get_asset_store().update(aid, data))
            else:
                self._error(404, "未知接口")
        except ValueError as exc:
            self._error(400, str(exc))
        except Exception as exc:  # noqa: BLE001
            self._error(500, str(exc))

    # ------------------------------------------------------------ 业务
    def _handle_image(self, data: dict) -> None:
        """/predict 与 /predict/image：封闭集分类，支持路径、多路径、base64 混传。

        三种入参可同时出现（base64 + 单张 + 批量），按出现顺序依次推理；
        每条结果都带 source 字段回溯来源（base64 记为 "base64"）。
        """
        p = get_predictor()
        images: list[tuple[str, Any]] = []  # (source, ndarray 或路径)

        if "image_base64" in data:
            images.append(("base64", _decode_image_base64(str(data["image_base64"]))))

        single = data.get("image")
        if single:
            images.append((str(single), str(single)))

        many = data.get("images") or []
        if isinstance(many, list):
            for item in many:
                images.append((str(item), str(item)))

        if not images:
            raise ValueError("需要提供 image / images / image_base64 之一")

        # 区分路径与内存图像：路径走 predict_images，base64 走 predict_frames
        # 之所以分两条路：predict_image 走 cv2.imread 有磁盘 IO 与大图解码；
        # 单张 base64 已经解好 ndarray，直接用 predict_frames（内部多帧聚合，单帧等价）
        results = []
        for source, payload in images:
            if isinstance(payload, str):
                path = _resolve_path(payload)
                if not path.exists():
                    raise FileNotFoundError(f"图片不存在：{path}")
                r = p.predict_image(path)
            else:
                r = p.predict_frames([payload], source=source)
            r["source"] = source
            results.append(r)

        self._ok({"count": len(results), "results": results})

    def _handle_video(self, data: dict) -> None:
        """/predict/video：整段视频多帧聚合分类。

        num_frames / aggregation 留空时用 cfg.inference 的默认值
        （默认 16 帧、mean_prob），客户端一般不用显式传。
        """
        raw = data.get("video")
        if not raw:
            raise ValueError("需要提供 video 路径")
        path = _resolve_path(str(raw))
        if not path.exists():
            raise FileNotFoundError(f"视频不存在：{path}")
        p = get_predictor()
        num_frames = data.get("num_frames")
        agg = data.get("aggregation")
        result = p.predict_video(
            path,
            num_frames=int(num_frames) if num_frames else None,
            aggregation=str(agg) if agg else None,
        )
        self._ok({"result": result})

    def _handle_identify(self, data: dict) -> None:
        """扩展接口：统一识别 → 完整电影档案。

        与 /predict 的区别：/predict 只能在训练过的类别里选一个（封闭集），
        这里走图库向量检索 + 元数据补全（开放集），返回多片名、演员、年份等；
        视频输入会先均匀抽帧再做多图检索聚合。
        """
        ident = get_identifier()
        kind = data.get("kind")  # frame/poster/actor/still/roadshow
        top_k = int(data["top_k"]) if data.get("top_k") else None
        with_actors = data.get("with_actors", True)

        # 视频输入
        if data.get("video"):
            path = _resolve_path(str(data["video"]))
            if not path.exists():
                raise FileNotFoundError(f"视频不存在：{path}")
            result = ident.identify_video(
                path, num_frames=int(data.get("num_frames", 16)),
                kind=kind or "frame", top_k=top_k,
                with_actors=bool(with_actors),
            )
            self._ok(result)
            return

        # 图片输入：路径 / 批量 / base64
        # 注意这里用 cv2.imread 而不是 ident.encoder.load_image，是为了统一走
        # 「路径必须存在」的校验（不存在就直接 404，不要让空图混进编码器）
        images: list[np.ndarray] = []
        sources: list[str] = []
        if data.get("image_base64"):
            images.append(_decode_image_base64(str(data["image_base64"])))
            sources.append("base64")
        for key in ("image", "images"):
            raw = data.get(key)
            # 允许 "image": "a.jpg" 与 "image": ["a.jpg", "b.jpg"] 两种写法
            items = raw if isinstance(raw, list) else ([raw] if raw else [])
            for item in items:
                path = _resolve_path(str(item))
                if not path.exists():
                    raise FileNotFoundError(f"图片不存在：{path}")
                images.append(cv2.imread(str(path)))
                sources.append(str(item))
        if not images:
            raise ValueError("需要提供 image / images / image_base64 / video 之一")

        result = ident.identify_images(
            images, kind=kind, top_k=top_k, with_actors=bool(with_actors))
        result["sources"] = sources  # 回填每个输入对应的来源，便于前端逐图展示
        self._ok(result)

    def _handle_actors(self, data: dict) -> None:
        """/actors/recognize：纯演员识别，返回每张图的**每一张脸**及候选。

        入参三种写法（可同时出现）：
            {"image": "a.jpg"} / {"images": ["a.jpg","b.jpg"]} / {"image_base64": "..."}
        输出：
            {num_images, results: [{source, num_faces, faces: [
                {bbox, actor, confidence, candidates, person_id, movies}]}]}
        与 /identify 的区别：这里不依赖电影分类模型，只回答"画面里是谁"，
        适合演员照片入库校验、合影标注、粉丝站素材批量打标等场景。
        """
        rec = get_actor_recognizer()
        images: list[tuple[str, np.ndarray]] = []
        if data.get("image_base64"):
            images.append(("base64", _decode_image_base64(str(data["image_base64"]))))
        for key in ("image", "images"):
            raw = data.get(key)
            items = raw if isinstance(raw, list) else ([raw] if raw else [])
            for item in items:
                path = _resolve_path(str(item))
                if not path.exists():
                    raise FileNotFoundError(f"图片不存在：{path}")
                img = cv2.imread(str(path))
                if img is None:
                    # 文件存在但解码失败（损坏/特殊格式）要明确报错，不能把 None 传进编码器
                    raise ValueError(f"图片无法解码：{path}")
                images.append((str(item), img))
        if not images:
            raise ValueError("需要提供 image / images / image_base64 之一")

        results = []
        for source, img in images:
            detail = rec.recognize_all(img)
            for face in detail.get("faces", []):
                # 补上"参演电影"，让接口一次返回完整上下文；查不到人就留 None
                person = rec.store.find_person(face["actor"]) if face.get("actor") else None
                face["person_id"] = person["id"] if person else None
                face["movies"] = person["movies"] if person else []
            results.append({"source": source, **detail})
        self._ok({"num_images": len(results), "results": results,
                  "recognizer": rec.status()})

    def _handle_actor_upload(self, data: dict) -> None:
        """/actors/photos：上传一张演员照片并打标签。

        请求：
            {"actor": "刘德华", "image_base64": "...", "name_en": "Andy Lau"}
        - actor 必填，作为标签；会同步登记到 persons 表
        - image_base64 支持带 data URL 前缀
        - name_en 可选

        行为：
            按内容 MD5 去重 → 转存为 data/actors/<演员名>/<hash>.jpg
            → 追加到 manifest.csv（split=train）→ 返回入库结果。
        注意：上传只做增量入库，不重新分层；攒够一批照片后请重跑
        scripts/15_import_actors.py（重新划分）与 16_train_actors.py（重训），
        新演员才会被分类头认识。
        """
        actor = str(data.get("actor") or "").strip()
        if not actor:
            raise ValueError("需要提供 actor（演员名）")
        b64 = data.get("image_base64")
        if not b64:
            raise ValueError("需要提供 image_base64")
        raw = str(b64)
        # 允许 "data:image/jpeg;base64,xxxx" 形式，只取逗号后的正文
        if raw.startswith("data:") and "," in raw:
            raw = raw.split(",", 1)[1]
        try:
            blob = base64.b64decode(raw, validate=False)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"image_base64 解码失败：{exc}") from exc

        from .actors.dataset import add_photo_bytes
        cfg = load_config()
        result = add_photo_bytes(
            actor, blob,
            dest_dir=get_value(cfg, "actors.data_dir", "data/actors"),
            cfg=cfg,
            name_en=data.get("name_en"),
        )
        result["next_step"] = ("照片已入库；重跑 scripts/15_import_actors.py 与 "
                              "scripts/16_train_actors.py 后新演员才会被识别")
        self._ok(result)

    def _handle_admin_upload(self, data: dict) -> None:
        """/admin/upload：Web 后台上传素材到资产库（幂等，按内容 sha256 去重）。

        请求（二选一）：
            {"image_base64": "...", "file_name": "poster.jpg", "tags": {...}}
            {"path": "data/uploads/xxx.jpg", "tags": {...}}   # 服务端已有文件
        - file_name 可选，仅用于展示；真正的文件名由 asset_id 派生，忽略客户端路径
        - tags 可选，按 /tag-schema 的字段传（media_kind/purpose/movie_ids/…）
        - 图片支持 data URL 前缀；视频建议用 path（先由运维放到 data/uploads/）
        行为：解码写盘 → AssetStore.add() → 返回素材完整标签。重复内容返回同一素材。
        """
        tags = data.get("tags") or {}
        if not isinstance(tags, dict):
            raise ValueError("tags 必须是对象")
        store = get_asset_store()
        upload_dir = ROOT / "data" / "uploads"

        if data.get("path"):
            # 服务端已有文件：直接登记（路径仅相对项目根，拒绝越界）
            src = _resolve_path(str(data["path"]))
            if not src.exists():
                raise FileNotFoundError(f"文件不存在：{src}")
            asset = store.add(src, tags=tags)
        elif data.get("image_base64") or data.get("file_base64"):
            b64 = data.get("image_base64") or data.get("file_base64")
            raw = str(b64)
            if raw.startswith("data:") and "," in raw:
                # 记录声明的扩展名（data URL 里带），缺省 jpg
                header = raw.split(",", 1)[0]
                ext_hint = header.split("/")[1].split(";")[0] if "/" in header else "jpg"
                raw = raw.split(",", 1)[1]
            else:
                ext_hint = ""
            try:
                blob = base64.b64decode(raw, validate=False)
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"base64 解码失败：{exc}") from exc
            if len(blob) > _MAX_UPLOAD:
                raise ValueError(f"文件超过 {_MAX_UPLOAD // 1024 // 1024}MB 上传上限")
            # 用内容 sha256 前 16 位命名，天然去重；扩展名取文件名/ data URL 提示
            digest = __import__("hashlib").sha256(blob).hexdigest()[:16]
            name = str(data.get("file_name") or "upload")
            ext = Path(name).suffix.lower() or (f".{ext_hint}" if ext_hint else ".bin")
            ensure_dir(upload_dir)
            dest = upload_dir / f"{digest}{ext}"
            if not dest.exists():
                dest.write_bytes(blob)  # 大文件写盘，不做图片解码（视频在此阶段不解析）
            asset = store.add(dest, tags=tags)
            asset["original_name"] = name
        else:
            raise ValueError("需要提供 image_base64 / file_base64 / path 之一")
        self._ok(asset)

    def _admin_status(self) -> dict[str, Any]:
        """状态中心汇总：把各处 stats 聚合成一屏可读的概览。"""
        status: dict[str, Any] = {"assets": get_asset_store().stats()}
        # 模型信息（best.pt 可能未加载，用 try 包住）
        try:
            p = get_predictor()
            status["model"] = {"num_classes": p.num_classes, "backbone": p.mode,
                               "device": str(p.device)}
        except Exception:  # noqa: BLE001
            status["model"] = None
        # 图库 / 元数据 / 演员 / 任务
        for key, fn in [
            ("gallery", lambda: get_identifier().gallery.stats()),
            ("metadata", lambda: get_identifier().store.stats()),
            ("actors", lambda: get_actor_recognizer().store.stats()),
            ("jobs", lambda: {"recent": get_job_manager().list(limit=5)}),
        ]:
            try:
                status[key] = fn()
            except Exception as exc:  # noqa: BLE001
                status[key] = {"error": str(exc)}
        return status


def main() -> None:
    """命令行入口：解析参数 → 加载模型 → 可选预热扩展服务 → 阻塞提供服务。

    启动顺序有意把 init_predictor 放在建 server 之前：
    权重缺失/路径写错时当场退出（非 0 退出码），而不是先占端口再返回 500。
    """
    ap = argparse.ArgumentParser(description="影瞳 · 推理 HTTP 服务")
    ap.add_argument("--host", default="0.0.0.0", help="监听地址（默认 0.0.0.0）")
    ap.add_argument("--port", type=int, default=8080, help="监听端口（默认 8080）")
    ap.add_argument("--checkpoint", default=None, help="模型权重路径")
    ap.add_argument("--config", default=None, help="配置文件路径")
    ap.add_argument("--device", default=None, help="推理设备：auto/cpu/cuda")
    ap.add_argument("--with-identifier", action="store_true",
                    help="启动时预热扩展服务（电影档案检索 / 别名搜索 / 演员识别）")
    args = ap.parse_args()

    cfg = load_config(args.config)
    init_predictor(args.checkpoint, cfg=cfg, device=args.device)
    # Web 后台不依赖电影模型，但同样在启动时把两个单例建好（建表/恢复中断任务，
    # 都是毫秒级），这样第一次点后台接口无需等待。
    get_asset_store()
    get_job_manager()
    if args.with_identifier:
        # 预热：把 Identifier 的首次构建（图库向量 + 元数据库 + 演员库）搬到启动阶段，
        # 避免第一个 /identify 请求等十几秒甚至超时。
        from .identify import Identifier
        init_identifier(Identifier(cfg=cfg, device=args.device))
        log.info("扩展服务已预热：图库检索 / 元数据库 / 演员识别")

    # ThreadingHTTPServer：每个请求一个线程。推理本身是 CPU/GPU 密集但调用的是
    # torch 内部并行，GIL 影响有限；小并发下比单线程 HTTPServer 体验好很多。
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    log.info("影瞳推理服务已启动：http://%s:%d", args.host, args.port)
    # 静态产物存在才提示 Web 入口，避免误导（未构建前端时 / 返回 JSON 指引）
    if _STATIC_DIR.exists():
        log.info("Web 界面：用户前台 /  管理后台 /admin  （同一端口）")
    else:
        log.info("Web 前端未构建：在 web/ 目录执行 npm install && npm run build 后刷新即可")
    log.info("基础接口：GET /health  GET /labels  POST /predict  POST /predict/video")
    log.info("扩展接口：POST /identify  GET /search  GET /movies  GET /actors/{id}  GET /gallery/stats")
    log.info("管理接口：GET /tag-schema  POST /admin/upload  POST /admin/train  GET /admin/tasks/{id}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        # Ctrl+C 属于正常停机路径，不打堆栈
        log.info("服务已停止")
    finally:
        # 释放监听端口；已进入的处理线程由守护线程机制随进程退出
        server.server_close()


if __name__ == "__main__":
    main()
