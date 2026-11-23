# 影瞳 CineEye · 推理接口文档

> 影瞳把训练好的电影识别模型暴露为 HTTP REST 接口，供外部系统直接调用。
> 服务基于 Python 标准库实现，**无需额外安装 Web 框架**，训练完即可启动。
> 训练与验证流程见 [TRAINING_GUIDE.md](TRAINING_GUIDE.md)。

---

## 目录

- [1. 启动服务](#1-启动服务)
- [2. 接口总览](#2-接口总览)
- [3. 接口详情](#3-接口详情)
- [4. 调用示例](#4-调用示例)
- [5. 通用响应结构](#5-通用响应结构)
- [6. 部署到生产环境的注意事项](#6-部署到生产环境的注意事项)
- [7. 常见问题](#7-常见问题)
- [8. 扩展接口详情（电影档案）](#8-扩展接口详情电影档案)
- [9. Web 管理接口（上传/标签/异步训练）](#9-web-管理接口上传标签异步训练)

> 带界面的操作方式（用户前台 `/` + 管理后台 `/admin`）见 [Web 界面指南](WEB_GUIDE.md)，
> 完整设计见 [WEB_UI_DESIGN.md](WEB_UI_DESIGN.md)。

---

## 1. 启动服务

```bash
cd /home/kylin/Workarea/CineEye

# 最简启动（监听 0.0.0.0:8080，任意网卡可访问）
./.venv/bin/python scripts/06_serve.py

# 指定地址/端口/权重
./.venv/bin/python scripts/06_serve.py \
    --host 0.0.0.0 --port 8080 \
    --checkpoint outputs/checkpoints/best.pt

# 仅本机监听（更安全，调试用）
./.venv/bin/python scripts/06_serve.py --host 127.0.0.1 --port 8080

# 预热扩展服务（/identify、/search、/movies 等电影档案接口，避免首次请求等待加载）
./.venv/bin/python scripts/06_serve.py --port 8080 --with-identifier

# 有 NVIDIA GPU 时显式指定（默认 auto 也会自动用 GPU）
./.venv/bin/python scripts/06_serve.py --port 8080 --device cuda --with-identifier
```

> `--device` 同时作用于基础 `/predict` 接口与扩展识别服务（图库/演员编码），默认 `auto` 会自动探测 CUDA；详见 [GPU 使用指南](GPU_GUIDE.md)。

启动成功的标志（日志）：

```text
[INFO] 模型已加载：best.pt | 主干=resnet18 | 类别=6 | 模式=feature | 设备=cpu
[INFO] 影瞳推理服务已启动：http://0.0.0.0:8080
```

模型只在启动时加载一次，之后所有请求复用，无需每次重新读权重。启动约需数秒（加载 43MB 权重）。

---

## 2. 接口总览

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 服务信息（名称、版本、接口列表） |
| GET | `/health` | 健康检查 |
| GET | `/labels` | 类别列表与编号映射 |
| POST | `/predict` | 识别图片（路径 / base64 / 批量） |
| POST | `/predict/image` | 同 `/predict`，语义更明确 |
| POST | `/predict/video` | 识别视频（多帧聚合） |

**扩展接口**（电影档案检索，需 `--with-identifier` 预热或首次调用惰性加载；详见 `docs/EXPANSION_GUIDE.md`）：

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/identify` | 统一识别（图片/视频/base64）→ 完整电影档案（多片名/演员） |
| GET | `/search?q=...` | 按片名/别名搜索（多语言、模糊） |
| GET | `/movies` | 电影列表（`limit`/`offset` 分页） |
| GET | `/movies/{id}` | 单部电影完整档案 |
| GET | `/actors` | 演员名册 + 识别器状态（模型/演员/阈值） |
| GET | `/actors/{id}` | 演员及其参演电影 |
| POST | `/actors/recognize` | 纯演员识别：多脸检测 → 演员名 + 候选 + 参演电影 |
| POST | `/actors/photos` | 上传演员照片并打标签入库（增量、按内容去重） |
| GET | `/gallery/stats` | 图库规模与状态 |

- 请求/响应均为 `application/json; charset=utf-8`。
- 已开启 CORS（`Access-Control-Allow-Origin: *`），浏览器可直连。
- 支持 `OPTIONS` 预检请求。

---

## 3. 接口详情

### 3.1 GET `/health` — 健康检查

响应：

```json
{
  "ok": true,
  "data": {
    "status": "ok",
    "model_loaded": true,
    "num_classes": 6,
    "backbone": "feature",
    "device": "cpu",
    "identifier": {"meta": {"movies": 6, "aliases": 6, "actors": 0, "cast_links": 0, "images": 240},
                   "gallery": {"images": 240, "dim": 512, "movies": 6},
                   "actors_enabled": true, "encoder": "resnet18/512d"}
  }
}
```

| 字段 | 说明 |
|---|---|
| `status` | 服务状态，`ok` 为正常 |
| `model_loaded` | 模型是否已加载 |
| `num_classes` | 类别数（本次 6） |
| `backbone` | 模型模式（feature / finetune） |
| `device` | 推理设备（cpu / cuda） |
| `identifier` | 扩展服务状态（元数据库 / 图库 / 演员 / 编码器）；未启用扩展时为 `null` |

### 3.2 GET `/labels` — 类别列表

响应：

```json
{
  "ok": true,
  "data": {
    "label_map": {"movie_01": 0, "movie_02": 1, "movie_03": 2, "movie_04": 3, "movie_05": 4, "movie_06": 5},
    "labels": ["movie_01", "movie_02", "movie_03", "movie_04", "movie_05", "movie_06"]
  }
}
```

对接方用 `labels` 展示可选类别，用 `label_map` 做编号与名称互转。

> 本次类别是中性编号 `movie_01~06`，对应的原始片名见 `outputs/reports/movie_id_map.json`。若需要接口直接返回片名，可在该映射表中翻译后对外暴露。

### 3.3 POST `/predict` — 图片识别

请求体支持三种写法（可单独用，也可组合）：

| 字段 | 类型 | 说明 |
|---|---|---|
| `image` | string | 图片路径（绝对路径，或相对项目根目录） |
| `images` | string[] | 多张图片路径，批量识别 |
| `image_base64` | string | 图片 base64，支持 `data:image/...;base64,` 前缀 |

单个请求示例：

```json
{"image": "/abs/path/to/frame.jpg"}
```

批量示例：

```json
{"images": ["frame1.jpg", "frame2.jpg", "frame3.jpg"]}
```

base64 示例：

```json
{"image_base64": "iVBORw0KGgoAAAANSUhEUgAA..."}
```

响应（真实返回）：

```json
{
  "ok": true,
  "data": {
    "count": 1,
    "results": [
      {
        "label": "movie_01",
        "label_idx": 0,
        "confidence": 0.912992,
        "unknown": false,
        "top_k": [
          {"label": "movie_01", "confidence": 0.912992},
          {"label": "movie_04", "confidence": 0.066356},
          {"label": "movie_06", "confidence": 0.018229},
          {"label": "movie_02", "confidence": 0.001851},
          {"label": "movie_05", "confidence": 0.000379}
        ],
        "source": "data/frames/movie_01/seg_01/000001.jpg"
      }
    ]
  }
}
```

| 字段 | 说明 |
|---|---|
| `count` | 本次返回的识别结果数 |
| `results[].label` | 识别出的类别 |
| `results[].label_idx` | 类别编号 |
| `results[].confidence` | 置信度（0–1） |
| `results[].unknown` | 是否为「未知影片」（启用拒识且置信度低于阈值时为 true） |
| `results[].top_k` | 前 K 个候选类别及置信度（K 由 `inference.top_k` 配置，默认 5） |
| `results[].source` | 输入来源（路径或 `base64`） |

### 3.4 POST `/predict/video` — 视频识别

请求体：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `video` | string | 是 | 视频路径（绝对路径，或相对项目根目录） |
| `num_frames` | int | 否 | 均匀采样帧数，默认 16 |
| `aggregation` | string | 否 | 聚合方式：`mean_prob` / `log_prob` / `majority`，默认 `mean_prob` |

示例：

```json
{"video": "/abs/path/to/clip.mp4", "num_frames": 16, "aggregation": "mean_prob"}
```

响应（真实返回）：

```json
{
  "ok": true,
  "data": {
    "result": {
      "label": "movie_04",
      "label_idx": 3,
      "confidence": 0.624793,
      "unknown": false,
      "top_k": [
        {"label": "movie_04", "confidence": 0.624793},
        {"label": "movie_03", "confidence": 0.135523},
        {"label": "movie_05", "confidence": 0.098419},
        {"label": "movie_06", "confidence": 0.058859},
        {"label": "movie_02", "confidence": 0.050431}
      ],
      "source": "/home/kylin/movie_train/某片.mp4",
      "aggregation": "mean_prob",
      "num_frames": 16,
      "frame_agreement": 0.75,
      "frame_votes": {"movie_02": 1, "movie_03": 2, "movie_04": 12, "movie_05": 1}
    }
  }
}
```

视频接口额外字段：

| 字段 | 说明 |
|---|---|
| `aggregation` | 实际使用的聚合方式 |
| `num_frames` | 实际采样帧数 |
| `frame_agreement` | 帧一致率：最多类别的票数 / 总帧数，越高说明整段画面越一致 |
| `frame_votes` | 各单帧投票分布，便于判断结果是否稳定 |

**为何用视频接口**：单帧可能是黑屏、转场或通用镜头；多帧聚合后准确率显著更高（本次视频级 100% vs 帧级 86%）。业务上优先传视频。

---

## 4. 调用示例

### curl

```bash
# 健康检查
curl --noproxy '*' http://127.0.0.1:8080/health

# 识别图片（路径）
curl --noproxy '*' -X POST http://127.0.0.1:8080/predict \
  -H 'Content-Type: application/json' \
  -d '{"image":"/abs/path/to/frame.jpg"}'

# 识别图片（base64）
curl --noproxy '*' -X POST http://127.0.0.1:8080/predict \
  -H 'Content-Type: application/json' \
  -d "{\"image_base64\":\"$(base64 -w0 frame.jpg)\"}"

# 识别视频
curl --noproxy '*' -X POST http://127.0.0.1:8080/predict/video \
  -H 'Content-Type: application/json' \
  -d '{"video":"/abs/path/to/clip.mp4","num_frames":16,"aggregation":"mean_prob"}'
```

> `--noproxy '*'` 是因为本机 shell 配了 HTTP 代理，直连本机地址需绕过。你自己的环境通常不需要。

### Python

```python
import json, urllib.request

BASE = "http://127.0.0.1:8080"

def post(path, payload):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))

# 图片
res = post("/predict", {"image": "/abs/path/to/frame.jpg"})
print(res["data"]["results"][0]["label"], res["data"]["results"][0]["confidence"])

# 视频
res = post("/predict/video", {"video": "/abs/path/to/clip.mp4", "num_frames": 16})
print(res["data"]["result"]["label"])
```

### JavaScript / fetch

```javascript
async function predictImage(path) {
  const r = await fetch("http://127.0.0.1:8080/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: path }),
  });
  const j = await r.json();
  return j.data.results[0];
}
```

---

## 5. 通用响应结构

所有接口统一结构：

```json
{"ok": true,  "data": { ... }}
{"ok": false, "error": "错误信息"}
```

HTTP 状态码：

| 状态码 | 含义 |
|---|---|
| 200 | 成功 |
| 400 | 请求参数错误（缺字段、非法 JSON、base64 无法解码、请求体超限） |
| 404 | 文件不存在或接口路径错误 |
| 500 | 服务内部错误（推理异常） |

请求体上限默认 64MB，可用环境变量 `CINEYE_MAX_BODY_MB` 调整。base64 图片建议单张不超过 10MB。

---

## 6. 部署到生产环境的注意事项

1. **绑定地址**：默认 `0.0.0.0:8080` 对外可访问。若只在内网用，绑定内网 IP 或加防火墙规则；不要在公网无保护地暴露。
2. **进程保活**：直接用 `python scripts/06_serve.py` 在前台运行，关闭终端即停止。生产环境用 `systemd`、`supervisor` 或 `nohup`/`screen` 保活：
   ```bash
   nohup ./.venv/bin/python scripts/06_serve.py --port 8080 > serve.log 2>&1 &
   ```
3. **并发**：服务基于多线程 HTTP Server，CPU 推理是计算密集型，并发请求会排队。单核 CPU 建议控制并发 ≤ 2–4；需要更高吞吐时多起几个进程 + 负载均衡。
4. **路径安全**：`image` / `video` 参数接受服务器本地路径。若暴露给不可信调用方，务必做路径白名单/沙箱，避免任意文件读取风险。推荐对接方用 `image_base64` 传图，避免暴露服务器路径。
5. **类别名称**：接口返回中性编号。若业务需要真实片名，在服务端用 `outputs/reports/movie_id_map.json` 做映射后再返回，不要直接把内部编号暴露给终端用户。
6. **未知影片拒识**：若业务要求「不属于这 6 部的图片不要强行归类」，在 `configs/default.yaml` 设置 `inference.unknown_threshold`（如 0.5），返回体的 `unknown` 字段会标记低置信度结果。
7. **HTTPS**：标准库服务本身不提供 TLS。公网部署时在前面挂 Nginx 做反向代理 + HTTPS。
8. **换模型**：重新训练后 `best.pt` 会被覆盖，重启服务即可加载新模型（模型在启动时加载一次）。

---

## 7. 常见问题

| 现象 | 原因 | 解决 |
|---|---|---|
| 启动报 `地址已占用` | 端口被占用 | `--port` 换一个，或 `lsof -i:8080` 查占用进程 |
| curl 返回 `Connection refused` | 服务没起来 / 走了代理 | 看 `serve.log`；本机直连加 `--noproxy '*'` |
| 请求返回 400「图片不存在」 | 路径写错或服务端无此文件 | 用绝对路径，或确认文件在服务器上 |
| base64 返回 400「不是可解码的图片」 | base64 格式错 | 用 `base64 -w0 file.jpg` 生成，不要带换行 |
| 视频识别很慢 | 长片解码 + 抽帧 | 减小 `num_frames`（如 8），或改用视频片段文件 |
| 返回都是同一类 | 输入与训练分布差异大 | 确认是同类内容；看 `top_k` 与 `frame_agreement` 判断是否真的低置信 |
| 想换类别体系 | 类别由训练数据决定 | 重新组织数据并重训，接口 `/labels` 会自动反映新类别 |

---

## 8. 扩展接口详情（电影档案）

启动服务时建议加 `--with-identifier` 预热图库与元数据库，避免首次请求等待加载。

### 8.1 POST `/identify`

统一识别：任意视觉线索 → 完整电影档案。

请求字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `image` | string | 图片路径（单张） |
| `images` | string[] | 图片路径（批量，多图聚合） |
| `image_base64` | string | base64 图片（可含 data URL 前缀） |
| `video` | string | 视频路径（多帧聚合） |
| `kind` | string | 图片类型过滤：`frame`/`poster`/`actor`/`still`/`roadshow`/`other` |
| `num_frames` | int | 视频采样帧数，默认 16 |
| `top_k` | int | 返回候选数 |
| `with_actors` | bool | 是否做演员识别，默认 true |

响应 `data` 示例（实测）：

```json
{
  "input_type": "images",
  "matches": [{
    "movie_id": "movie_04",
    "confidence": 0.870476,
    "title": "movie_04",
    "original_title": null,
    "aliases": [{"alias": "movie_04", "lang": "id", "region": null, "source": "labelmap"}],
    "year": null, "director": null, "cast": [], "poster": null,
    "evidence": {"kind": "frame", "similarity": 0.870476,
                 "matched_path": "data/frames/movie_04/seg_03/000005.jpg",
                 "classifier_confidence": 0.9736}
  }],
  "actor": null,
  "elapsed_ms": 69
}
```

> **返回门槛（重要）**：默认只返回图库相似度 ≥ 80% 的候选（配置项 `identify.min_similarity`，设 0 可关闭）。`confidence`、`evidence.similarity` 与排序都以图库相似度为准，所见即所得——画面够像才认，低于门槛的乱猜一律不返回（`matches: []`）。帧类图片的「60% 图库 + 40% 分类器」融合分不再覆盖主分数，仅作辅助证据保留在 `evidence.fused_score`；`evidence.classifier_confidence` 仍是分类器原始概率，便于排障。

### 8.2 GET `/search?q=...`

按片名/别名搜索，支持多语言译名与模糊匹配（繁简、大小写、标点归一）。返回 `results[]`，每项含 `movie_id`、`score`、`matched_alias`、`aliases`、`cast`。

### 8.3 GET `/movies`、`/movies/{id}`、`/actors/{id}`

- `/movies?limit=100&offset=0`：返回 `total` 与 `movies[]`；
- `/movies/{id}`：单部电影完整档案（含别名、演员、图片）；
- `/actors/{id}`：演员信息及其参演电影列表。

### 8.4 POST `/actors/recognize`

只做演员识别，不依赖电影分类模型（机器上没有 `best.pt` 也能用）。
请求字段与 `/identify` 的图片部分一致：

| 字段 | 类型 | 说明 |
|---|---|---|
| `image` | string | 单张图片路径 |
| `images` | string[] | 多张图片路径 |
| `image_base64` | string | base64 图片（可含 data URL 前缀） |

响应示例：

```json
{
  "ok": true,
  "data": {
    "num_images": 1,
    "results": [{
      "source": "合影.jpg",
      "num_faces": 2,
      "unknown": false,
      "faces": [{
        "bbox": [120, 80, 96, 96],
        "actor": "刘德华",
        "confidence": 0.87,
        "candidates": [
          {"name": "刘德华", "score": 0.87, "probability": 0.91, "similarity": 0.78}
        ],
        "person_id": "person_xxxxxxxxxxxx",
        "movies": [{"movie_id": "movie_01", "primary_title": "...", "year": 2002}]
      }]
    }],
    "recognizer": {"mode": "classifier+gallery", "num_actors": 8, "threshold": 0.42}
  }
}
```

说明：
- `bbox` 为 `[x, y, w, h]` 像素坐标，按人脸面积从大到小排列；
- `actor` 为 `null` 表示该脸低于拒识阈值（`unknown`）；
- `candidates` 同时给出融合分 `score`、分类概率 `probability`、检索相似度 `similarity`，便于排障；
- 识别器的训练与调优见 [演员识别指南](ACTOR_GUIDE.md)。

### 8.5 POST `/actors/photos`

上传一张演员照片并打上标签，用于"边用边攒数据"。

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `actor` | string | 是 | 演员名（标签），同步登记到 persons 表 |
| `image_base64` | string | 是 | 图片 base64，可带 `data:image/jpeg;base64,` 前缀 |
| `name_en` | string | 否 | 英文名 |

响应示例：

```json
{
  "ok": true,
  "data": {
    "added": true,
    "duplicate": false,
    "path": "data/actors/刘德华/1cfccb4c4228.jpg",
    "actor": "刘德华",
    "sha1": "1cfccb4c42281604c7017713c874a343",
    "total": 12,
    "next_step": "照片已入库；重跑 15/16 后新演员才会被识别"
  }
}
```

说明：
- 按图片内容 MD5 去重，重复上传返回 `duplicate: true`；
- 只做增量入库（默认 `split=train`），不重排已有划分；
- 攒够一批后重跑 `scripts/15_import_actors.py` 与 `scripts/16_train_actors.py` 才生效。

### 8.6 GET `/gallery/stats`

图库状态：`images`、`dim`、`movies`、`by_kind`、`encoder_fingerprint`、`updated_at`。

---

基础接口（health、labels、图片路径、base64、视频聚合、400/404）与扩展接口（identify、search、movies、gallery/stats）均已在本机端到端验证通过。

---

## 9. Web 管理接口（上传/标签/异步训练）

> 这些接口供管理后台页面与脚本调用，统一 `{ok, data, error}` 包络。
> 完整字段定义见 [WEB_UI_DESIGN.md](WEB_UI_DESIGN.md)，界面用法见 [WEB_GUIDE.md](WEB_GUIDE.md)。

### 9.1 接口清单

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api` | JSON 版服务发现（前端页面占用 `/` 后用它自检） |
| GET | `/tag-schema` | 五层标签契约，前端表单据此生成 |
| POST | `/admin/upload` | 上传图片/视频入库（base64 或服务端路径），按 sha256 去重 |
| GET | `/admin/assets` | 素材列表，支持 `?kind=&purpose=&q=&status=&limit=&offset=` |
| GET | `/admin/assets/{id}` | 单个素材完整标签（含多电影/演员关联） |
| PATCH | `/admin/assets/{id}` | 补标/改标，按契约校验并派生写回现有流程 |
| POST | `/admin/assets/{id}/archive` | 归档（软删） |
| POST | `/admin/train` | 触发训练，立即返回 task_id（异步） |
| GET | `/admin/tasks` | 任务列表 |
| GET | `/admin/tasks/{id}` | 任务状态 + 进度 + 日志尾部 |
| POST | `/admin/tasks/{id}/cancel` | 取消任务 |
| GET | `/admin/status` | 状态中心汇总（模型/图库/元数据/演员/素材/任务） |

### 9.2 上传素材

```bash
# base64（图片，可带 data URL 前缀）
curl -X POST http://127.0.0.1:8080/admin/upload \
  -H 'Content-Type: application/json' \
  -d '{"file_base64":"data:image/jpeg;base64,<...>","file_name":"poster.jpg","tags":{"media_kind":"poster"}}'

# 服务端已有文件（大视频推荐：先放到 data/uploads/ 再登记）
curl -X POST http://127.0.0.1:8080/admin/upload \
  -H 'Content-Type: application/json' \
  -d '{"path":"data/uploads/clip.mp4","tags":{"media_kind":"clip"}}'
```

响应 `data` 为素材完整标签（含 `asset_id`、`purpose` 按 `media_kind` 自动推导）。同内容重复上传返回同一 `asset_id`。

### 9.3 补标签

```bash
curl -X PATCH http://127.0.0.1:8080/admin/assets/asset_xxxxxxxx \
  -H 'Content-Type: application/json' \
  -d '{"media_kind":"still","purpose":"gallery","movie_ids":["movie_01"],"person_ids":["person_xxx"],"free_tags":["场景:雨夜"],"notes":"路演现场"}'
```

`movie_ids` / `person_ids` 可多值；保存后自动派生写回 `movie_images` 表与演员派生清单。枚举非法值返回 400。

### 9.4 触发训练与轮询

```bash
# 触发（立即返回，不阻塞）
curl -X POST http://127.0.0.1:8080/admin/train \
  -H 'Content-Type: application/json' \
  -d '{"kind":"train_movie","epochs":40}'
# => {"ok":true,"data":{"task_id":"job_20261119...","status":"queued"}}

# 轮询进度
curl http://127.0.0.1:8080/admin/tasks/job_20261119...
# => data: {status, progress, log_tail, result, log_path, ...}
```

`kind` 可选：`train_movie`（抽帧→清单→训练）、`train_actor`（导入→训练）、`build_gallery`（重建图库）、`refresh_derived`（仅同步标签，秒级）。
状态流转：`queued` → `running` → `done` / `failed` / `canceled`。线程池并发上限 2，超出的任务排队。
