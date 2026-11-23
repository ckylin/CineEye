# 影瞳 · Web 双界面 + 统一标签体系 完整设计

> 本文是「用户前台 + 管理后台 + 统一标签 schema + 异步训练」这一期工程的**唯一事实源**。
> 后续所有代码、接口、数据库变更都以此为准。实施顺序见第 9 节。
>
> 状态：设计已定稿（2026-11-19），待实施。

---

## 1. 目标与已确认决策

### 1.1 一句话目标

给「影瞳」增加两个 Web 页面（**用户前台** + **管理后台**），后台能上传视频/图片并打一套统一完整的五层标签、一键异步触发训练并查看进度；前台上传即识别，返回电影名、别名、演员。所有素材用同一份标签契约和资产表管理，且**完全兼容已经跑通的训练和图库流程**。

### 1.2 已确认决策（用户拍板，不要再改）

| 决策项 | 结论 | 影响 |
|---|---|---|
| 前端技术栈 | **React + Vite**，构建为纯静态文件由 `serve.py` 同端口托管 | 不引入额外服务进程，部署只多一步 `npm run build` |
| 鉴权 | **仅本机/局域网，不登录** | 不做会话；代码注释标明"对外暴露需补鉴权+限流" |
| 训练反馈 | **异步任务 + 进度轮询** | 点一下立即返回 task_id，页面轮询状态/日志，浏览器不卡死 |
| 标签范围 | **完整五层，合规层可选** | 基础/分类/关联/训练四层可用，合规字段预留不强制 |
| 关联基数 | **多电影 + 多演员** | 用多对多关联表，合影/群像可标多人 |
| 打标时机 | **先上传，后补标** | 上传只需选文件即可入库，标签在素材库页慢慢补 |

### 1.3 设计原则

1. **不推倒重来**：现有训练脚本（`01~05`、`15/16`）、图库、元数据库全部保留。新增的资产层是"旁路"，通过派生写回兼容老流程。
2. **契约单一**：标签只在 `configs/tag_schema.json` 定义一次，React 表单和后端校验都读它，杜绝两端漂移。
3. **离线可用**：React 依赖一次性 `npm install`，构建产物随项目走；运行时不依赖 CDN。
4. **环境适配**：沿用项目既有的 `/tmp`→`.tmp`、`/dev/shm` 检测、设备自适应约定。

---

## 2. 总体架构

```
浏览器                          同一端口（默认 8080）
┌─────────────────────┐
│ 用户前台  /          │──┐
│ 管理后台  /admin      │──┤
└─────────────────────┘  │
                         ▼
              ┌────────────────────────────┐
              │  cineeye/serve.py          │
              │  · 静态托管 / /admin /assets│
              │  · 现有识别接口             │
              │  · 新增 /admin/* 管理接口   │
              └───────────┬────────────────┘
                          │
        ┌─────────────────┼──────────────────┐
        ▼                 ▼                  ▼
 ┌─────────────┐   ┌──────────────┐   ┌──────────────┐
 │ assets.py   │   │ jobs.py      │   │ 现有能力      │
 │ 资产+标签层  │   │ 异步任务调度  │   │ 识别/检索/训练│
 └──────┬──────┘   └──────┬───────┘   └──────────────┘
        │                 │ 启动子进程
        ▼                 ▼
 ┌──────────────────────────────────────────┐
 │ 存储：outputs/cineeye.db（SQLite）        │
 │  · assets / asset_movies / asset_persons  │
 │  · asset_tags / jobs（新增）              │
 │  · movies / persons / movie_images（现有）│
 │ 文件：data/uploads/…  outputs/reports/…   │
 └──────────────────────────────────────────┘
```

---

## 3. 统一标签 Schema（五层）

### 3.1 定义位置

`configs/tag_schema.json`，结构：

```json
{
  "version": "1.0",
  "layers": [
    {
      "key": "basic",
      "label": "基础属性",
      "fields": [
        {"key": "asset_type", "label": "素材类型", "type": "enum",
         "options": ["video", "image"], "required": true, "default": "image"},
        {"key": "source", "label": "来源", "type": "enum",
         "options": ["upload", "local", "tmdb", "douban", "manual"], "default": "upload"}
      ]
    }
  ]
}
```

字段 `type` 取值：`string` / `enum` / `number` / `bool` / `date` / `multi_enum` / `tags`。
React 前端按 `type` 自动选择控件（下拉/多选/输入框/日期）；后端按 `type` + `options` + `required` 校验。

### 3.2 五层字段清单

**第 1 层 基础属性（basic）** — 上传时自动填充，用户一般不改

| 字段 | 类型 | 说明 |
|---|---|---|
| `asset_id` | string | 全局唯一 ID，`asset_` + sha256 前 16 位 |
| `asset_type` | enum | `video` / `image` |
| `file_path` | string | 相对项目根的 POSIX 路径 |
| `file_name` | string | 原始文件名 |
| `sha256` | string | 内容指纹，去重依据 |
| `size` | number | 字节数 |
| `width` / `height` | number | 像素（视频取首帧） |
| `duration` | number | 秒（仅视频） |
| `ext` | string | 扩展名 |
| `source` | enum | `upload/local/tmdb/douban/manual` |
| `source_url` | string | 原始链接（合规层，可选） |
| `uploaded_at` | date | 入库时间 |
| `status` | enum | `active` / `archived` |

**第 2 层 内容分类（content）** — 补标时选择

| 字段 | 类型 | 说明 |
|---|---|---|
| `media_kind` | enum | 见 3.3 受控枚举；决定素材进哪个图库/训练流程 |
| `lang` | string | 语言，如 `zh-CN` |
| `region` | string | 地区，如 `CN` |

**第 3 层 语义关联（semantic）** — 多对多

| 字段 | 类型 | 说明 |
|---|---|---|
| `movie_ids` | multi_enum | 关联电影（可多），来自 `movies` 表 |
| `person_ids` | multi_enum | 出镜演员（可多），来自 `persons` 表 |
| `free_tags` | tags | 自由标签，如 `场景:雨夜`、`年代:2000s` |

**第 4 层 训练用途（training）** — 决定素材如何被模型使用

| 字段 | 类型 | 说明 |
|---|---|---|
| `purpose` | enum | `train` / `query` / `gallery` / `reference` |
| `split` | enum | `train` / `val` / `test`（留空由划分脚本自动定） |
| `label` | string | 分类目标（电影识别=标签名，演员识别=演员名；留空从关联推导） |
| `quality` | enum | `high` / `medium` / `low`（可由清晰度启发式预填） |
| `face_ok` | bool | 是否检出人脸 |
| `face_boxes` | string | JSON 字符串，人脸框 `[x,y,w,h]` 列表 |
| `used_in` | tags | 已用于哪几次训练（task_id 列表） |

**第 5 层 合规采集（compliance）** — 全部可选，预留

| 字段 | 类型 | 说明 |
|---|---|---|
| `license` | string | 版权/授权说明 |
| `consent` | bool | 肖像授权是否取得 |
| `notes` | string | 备注 |

### 3.3 `media_kind` 受控枚举（可扩展）

| 值 | 中文 | 默认 purpose | 说明 |
|---|---|---|---|
| `frame` | 视频帧 | train | 视频抽帧，电影识别主训练数据 |
| `poster` | 海报 | gallery | 海报检索 |
| `still` | 剧照 | gallery | 官方剧照 |
| `behind_scenes` | 花絮 | gallery | 幕后花絮 |
| `roadshow` | 路演照 | gallery | 路演/活动现场 |
| `actor_portrait` | 演员写真 | reference | 演员识别参考/训练 |
| `actor_life` | 演员生活照 | reference | 演员识别训练 |
| `screenshot` | 用户截图 | query | 用户随手截图，走识别查询 |
| `trailer` | 预告片片段 | train | 视频类 |
| `clip` | 正片片段 | train | 视频类 |
| `other` | 其他 | query | 兜底 |

新增枚举值：改 `tag_schema.json` 的 `options` 即可，前后端自动生效（无需改代码）。

---

## 4. 数据模型（`outputs/cineeye.db` 新增表）

沿用现有 `MetaStore` 的 SQLite 连接，新增 5 张表（均 `CREATE TABLE IF NOT EXISTS`，不动旧表）：

```sql
CREATE TABLE IF NOT EXISTS assets (
    asset_id    TEXT PRIMARY KEY,
    asset_type  TEXT NOT NULL,
    file_path   TEXT NOT NULL,
    file_name   TEXT,
    sha256      TEXT NOT NULL UNIQUE,
    size        INTEGER,
    width       INTEGER,
    height      INTEGER,
    duration    REAL,
    ext         TEXT,
    source      TEXT DEFAULT 'upload',
    source_url  TEXT,
    media_kind  TEXT DEFAULT 'other',
    lang        TEXT,
    region      TEXT,
    purpose     TEXT DEFAULT 'query',
    split       TEXT,
    label       TEXT,
    quality     TEXT,
    face_ok     INTEGER DEFAULT 0,
    face_boxes  TEXT,
    used_in     TEXT,
    license     TEXT,
    consent     INTEGER,
    notes       TEXT,
    status      TEXT DEFAULT 'active',
    uploaded_at TEXT DEFAULT (datetime('now')),
    updated_at  TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS asset_movies (
    asset_id TEXT NOT NULL REFERENCES assets(asset_id) ON DELETE CASCADE,
    movie_id TEXT NOT NULL REFERENCES movies(id) ON DELETE CASCADE,
    PRIMARY KEY (asset_id, movie_id)
);

CREATE TABLE IF NOT EXISTS asset_persons (
    asset_id  TEXT NOT NULL REFERENCES assets(asset_id) ON DELETE CASCADE,
    person_id TEXT NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    PRIMARY KEY (asset_id, person_id)
);

CREATE TABLE IF NOT EXISTS asset_tags (
    asset_id TEXT NOT NULL REFERENCES assets(asset_id) ON DELETE CASCADE,
    tag      TEXT NOT NULL,
    PRIMARY KEY (asset_id, tag)
);

CREATE TABLE IF NOT EXISTS jobs (
    id         TEXT PRIMARY KEY,
    kind       TEXT NOT NULL,          -- train_movie / train_actor / build_gallery / refresh_derived
    status     TEXT NOT NULL,          -- queued / running / done / failed / canceled
    payload    TEXT,                   -- JSON：触发参数
    progress   INTEGER DEFAULT 0,      -- 0-100 粗粒度
    log_path   TEXT,                   -- 日志文件相对路径
    created_at TEXT DEFAULT (datetime('now')),
    started_at TEXT,
    finished_at TEXT,
    result     TEXT                    -- JSON：退出码/指标摘要
);
CREATE INDEX IF NOT EXISTS idx_assets_kind ON assets(media_kind);
CREATE INDEX IF NOT EXISTS idx_assets_sha  ON assets(sha256);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
```

### 4.1 与现有流程的兼容（关键）

| 现有产物 | 派生规则 | 触发时机 |
|---|---|---|
| `movie_images` 表 | `assets` 中 `media_kind ∈ {frame,poster,still,behind_scenes,roadshow}` 且已关联 `movie_id` → 写 `movie_images(movie_id, kind, path, source)` | 补标保存时 / `refresh_derived` 任务 |
| `data/actors/manifest.csv` | `person_ids` 非空 且 `media_kind ∈ {actor_portrait,actor_life}` → 追加演员照片记录 | 补标保存时 |
| `data/frames/<label>/...` | `asset_type=video` 且 `purpose=train` → 训练任务抽帧到此目录 | `train_movie` 任务 |
| 图库 `gallery/index` | `purpose=gallery/frame` 的 image 素材 → 建索引 | `build_gallery` 任务 |

**老数据反向导入**：提供一个 `assets.import_legacy()`，扫描现有 `movie_images`、两份 manifest，按 `file_path` 生成 sha256 回填 `assets`（`source='local'`），保证"老数据也能在素材库页面看到并补标"。

---

## 5. 异步任务机制（`cineeye/jobs.py`）

### 5.1 设计

- 一个进程内 `ThreadPoolExecutor(max_workers=2)`（避免同时跑两个训练抢资源）。
- 任务提交：写 `jobs` 表（`status=queued`）→ 丢进线程池 → 线程内起 `subprocess.Popen` 跑现成脚本。
- 日志：子进程 stdout/stderr 实时追加到 `outputs/reports/jobs/<task_id>.log`；`progress` 按日志里的 epoch 关键字粗略估算（或脚本末尾统一写 100）。
- 状态轮询：前端每 1~2s `GET /admin/tasks/{id}`，返回 `status/progress/log_tail/result`。
- 取消：`POST /admin/tasks/{id}/cancel` 终止子进程（尽力而为，标记 canceled）。

### 5.2 任务类型与对应命令

| kind | 触发命令（复用现成脚本） |
|---|---|
| `train_movie` | `01_extract_frames.py` → `02_build_manifest.py` → `03_train.py`（可选 `--set` 覆盖） |
| `train_actor` | `15_import_actors.py` → `16_train_actors.py` |
| `build_gallery` | `11_import_metadata.py`（若需）→ `12_build_gallery.py` |
| `refresh_derived` | 纯 Python 派生写回，无子进程 |

> 电影训练沿用 `video_id` 分组防泄漏逻辑；上传的视频先落到 `data/videos/<label>/`，`label` 取 `assets.label` 或关联电影的 `primary_title`。

---

## 6. HTTP 接口清单

### 6.1 现有接口（不改，前端复用）

`/health`、`/labels`、`/predict`、`/predict/image`、`/predict/video`、`/identify`、`/search`、`/movies`、`/movies/{id}`、`/actors`、`/actors/{id}`、`/actors/recognize`、`/actors/photos`、`/gallery/stats`

### 6.2 新增接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/tag-schema` | 返回 `tag_schema.json`，前端据此生成表单 |
| GET | `/` | 用户前台 `index.html` |
| GET | `/admin` | 管理后台（同一 SPA，前端路由切换） |
| GET | `/assets/*` | Vite 构建产物静态资源 |
| POST | `/admin/upload` | 上传视频/图片：multipart 或 base64 JSON；入库 `assets`，返回 `asset_id` |
| GET | `/admin/assets` | 素材库列表，支持 `?kind=&purpose=&q=` 过滤与分页 |
| GET | `/admin/assets/{id}` | 单个素材完整标签 |
| PATCH | `/admin/assets/{id}` | 补标/改标，按 `tag_schema.json` 校验，触发派生写回 |
| DELETE | `/admin/assets/{id}` | 归档（软删，status=archived） |
| POST | `/admin/train` | `{kind, options}` → 返回 `{task_id}` |
| GET | `/admin/tasks` | 任务列表 |
| GET | `/admin/tasks/{id}` | 任务状态 + 进度 + 日志尾部 |
| POST | `/admin/tasks/{id}/cancel` | 取消任务 |
| GET | `/admin/status` | 汇总：模型/图库/元数据/演员/任务统计 |

所有新接口沿用现有 `{ok, data, error}` 响应包络与 CORS 处理。

---

## 7. 前端结构（`web/`）

```
web/
├── package.json            # react + react-dom + vite
├── vite.config.js          # base:'/' , build.outDir:'../static'
├── index.html
└── src/
    ├── main.jsx            # 入口，按 pathname 路由 / 与 /admin
    ├── api.js              # 统一 fetch 封装
    ├── UserApp.jsx         # 前台：拖拽上传 → 识别 → 结果卡片
    ├── AdminApp.jsx        # 后台壳：侧边导航
    ├── pages/
    │   ├── Assets.jsx      # 素材库：表格 + 筛选 + 补标抽屉
    │   ├── Upload.jsx      # 上传（拖拽、多文件、进度）
    │   ├── Train.jsx       # 触发训练 + 任务列表 + 进度/日志
    │   └── Status.jsx      # 状态中心
    └── components/
        ├── TagForm.jsx     # 按 tag_schema.json 动态渲染表单
        └── ResultCard.jsx  # 识别结果卡片
```

构建：`cd web && npm run build` → 产物写入 `static/`，由 `serve.py` 托管。
开发：`npm run dev`（Vite 独立端口，需配 proxy 指向 8080 后端）。

**前台交互**：拖拽/选择图片或视频 → 调 `/identify`（图片）或 `/predict/video`（视频）→ 结果卡片展示片名、别名（多语言）、演员、置信度；演员照可切 `/actors/recognize`。

**后台交互**：
- 素材库：表格列出 `assets`，按 kind/purpose/搜索过滤，点行打开补标抽屉（`TagForm` 动态生成五层表单）。
- 上传：拖拽多文件，上传后提示"去素材库补标"。
- 训练：选任务类型 + 参数，点开始 → 跳到任务详情轮询进度与日志。
- 状态中心：展示模型信息、图库统计、元数据统计、演员数。

---

## 8. 安全与运维边界

- **不做登录**（已确认仅内网）；`serve.py` 相关注释统一标注：对外暴露前必须补鉴权、限流、上传路径白名单与文件大小上限。
- 上传目录固定在 `data/uploads/`，文件名由 `asset_id` 生成，忽略客户端路径（防目录穿越）。
- `file_path` 一律相对项目根 POSIX 字符串，跨机器可用。
- 大视频上传设置可配置上限（默认 2 GiB，`CINEYE_MAX_UPLOAD_MB`）。
- 合规字段（license/consent/source_url）仅记录，不做法律判断。

---

## 9. 落地顺序（实施清单）

1. **设计文档**（本文）✅
2. **标签契约 + 资产层**：`configs/tag_schema.json`、`cineeye/assets.py`、新增 5 表 + `import_legacy()`
3. **后端接口 + 任务**：`cineeye/jobs.py`、`serve.py` 扩展 12 个接口、静态托管
4. **React 前后台**：`web/` 全套，构建产物入 `static/`
5. **联调 + 测试 + 文档**：`tests/test_web.py`、README/API_GUIDE 入口、项目日志

每步完成后跑 `pytest`（当前基线 17 passed）确保无回归；最后端到端验证：
上传一张图 → 补标 → 触发一次轻量训练（或 `refresh_derived`）→ 看进度 → 前台识别。

---

## 10. 验收标准

- [ ] `GET /tag-schema` 返回五层契约，前端表单据此生成
- [ ] 上传图片/视频入库 `assets`，重复内容被 sha256 去重
- [ ] 补标多电影/多演员写入关联表，并派生到现有流程
- [ ] `/admin/train` 立即返回 task_id，任务异步执行，进度/日志可轮询
- [ ] 前台上传图片/视频能拿到识别结果卡片
- [ ] `/`、`/admin` 页面由 8080 同端口打开，无需单独起前端服务
- [ ] `pytest` 全绿（含新增 web 测试），现有 17 项无回归
- [ ] 文档入口更新，负责人能照着用
