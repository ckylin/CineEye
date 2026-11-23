# 影瞳 CineEye · 扩展架构设计

> 目标：从「封闭集分类器」升级为「以电影为中心的开放检索系统」。
>
> 输入：电影帧、海报、演员照、花絮照、路演照、视频等任意视觉线索。
> 输出：电影档案（多语言片名 / 别名 / 演员 / 年份 / 导演 / 海报）+ 置信度。
> 规模目标：几百到几千部电影；新增电影**无需重训模型**。

---

## 1. 为什么要改架构

现有系统是**封闭集分类**：训练时见过 6 类，推理时只能从这 6 类里选。它有三个硬限制：

1. 新增一部电影必须重新抽帧、重训模型；
2. 演员照片、海报、花絮照不在训练分布里，识别准确率没有保证；
3. 输出只有一个 `movie_0X` 编号，没有片名、演员等信息。

扩展后改为**开放集检索**：

- 用一个统一的图片编码器（复用已训练的 ResNet18 主干）把任意图片变成 512 维向量；
- 每部电影建立一个**图库**，存放它的各类参考图（帧 / 海报 / 演员照 / 花絮 / 路演）的向量；
- 输入图片时，用向量相似度检索出最相关的电影，**不需要重新训练**；
- 检索结果再去**元数据库**查出片名（含多别名）、演员等档案信息，一起返回。

---

## 2. 总体架构

```
输入 (图片 / 视频 / base64)
        │
        ▼
┌───────────────────────────────────────────────┐
│  统一特征编码器 FeatureEncoder                 │
│  复用 best.pt 的 ResNet18 主干（512 维）        │
└───────────────────────────────────────────────┘
        │
        ├──────────────► 人脸检测 (Haar, 可选)
        │                    │
        │                    ▼
        │            演员参考图库比对 → 演员身份
        │
        ▼
┌───────────────────────────────────────────────┐
│  图库检索引擎 GalleryIndex                     │
│  NumPy 余弦相似度 / sklearn 近邻（预留 faiss）  │
│  按图片类型过滤：frame/poster/actor/still/...  │
└───────────────────────────────────────────────┘
        │  电影候选 + 置信度
        ▼
┌───────────────────────────────────────────────┐
│  元数据库 MetaStore (SQLite)                   │
│  电影 / 别名(多语言) / 演员 / 演员-电影关系      │
│  数据提供者：本地表格 | TMDB API | 豆瓣抓取      │
└───────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────┐
│  统一识别服务 Identifier + REST API            │
│  返回：电影档案(多片名/演员) + 演员识别 + 证据   │
└───────────────────────────────────────────────┘
```

---

## 3. 元数据库设计（SQLite）

文件：`outputs/cineeye.db`（可用 `--set metadata.db=...` 改）。

### 3.1 表结构

| 表 | 字段 | 说明 |
|---|---|---|
| `movies` | `id`, `primary_title`, `original_title`, `year`, `director`, `synopsis`, `poster_path`, `release_date`, `created_at`, `updated_at` | 电影主体，`id` 用中性编号（`movie_01`）|
| `movie_aliases` | `id`, `movie_id`, `alias`, `lang`, `region`, `source` | 多片名/译名，一部电影可有多条；`source` 标记来自 TMDB / 豆瓣 / 手工 |
| `persons` | `id`, `name`, `name_en`, `role`(actor/director), `tmdb_id` | 演员与导演实体 |
| `movie_cast` | `movie_id`, `person_id`, `character`, `billing_order` | 电影 ↔ 演员多对多关系 |
| `movie_images` | `id`, `movie_id`, `kind`, `path`, `source`, `created_at` | 图库图片索引，`kind` 见下 |
| `sources` | `movie_id`, `provider`, `external_id`, `fetched_at` | 外部数据源追溯 |

`movie_images.kind` 支持：`frame`（影片帧）、`poster`（海报/封面）、`actor`（演员照）、`still`（剧照/花絮）、`roadshow`（路演照）、`other`。

### 3.2 多片名处理

- 一部电影的中文译名、原名、英文名、别名全部进 `movie_aliases`；
- 搜索时对用户查询做**归一化**：去除标点/空格、繁转简（基础映射）、大小写统一，再做模糊匹配；
- API 返回电影档案时，`aliases` 是一个数组，按语言分组返回。

### 3.3 三个数据提供者（可插拔）

| Provider | 输入 | 输出 | 依赖 |
|---|---|---|---|
| `local_table` | CSV / JSON 文件 | 电影、别名、演员、图片路径 | 无 |
| `tmdb` | 电影名 / TMDB ID | 多语言片名、演员、海报、简介 | 需要 API key，用标准库 `urllib` |
| `douban` | 电影名 | 中文片名、别名、演员、海报 | 无 key，但受反爬影响，字段稳定性一般 |

统一接口：`fetch_by_title(title) -> dict` / `fetch_by_id(id) -> dict`，返回标准化字典，由 `MetaStore.upsert_movie()` 统一入库。三者都只写本地库，运行时查询**完全离线**。

---

## 4. 图库检索引擎

### 4.1 特征编码器

复用 `best.pt` 的 `model_state`，用 `CineEyeNet.extract_features()` 输出 512 维向量（ImageNet 归一化）。与现有 `Predictor` 共用权重，不新增模型、不联网。

### 4.2 索引结构

```
outputs/gallery/
├─ embeddings.npy      # (N, 512) float32，L2 归一化后
├─ meta.json           # 每行对应 movie_id / kind / path / source
└─ index.meta.json     # 维度、样本数、编码器指纹、更新时间
```

- 支持**增量添加**：`GalleryIndex.add()` 会加载已有向量，追加新向量后整体重写（`rebuild()` 为全量重建）；
- 查询：`embedding @ index.T` 得到余弦相似度，取 top-k；按 `kind` 过滤；
- 多图聚合：输入多张图（或视频抽帧）时，对每张图的候选分数做融合（`mean` / `max` / `log_sum`，由 `gallery.aggregation` 配置）；
- 规模：几千部 × 每部几十张 = 十万级向量，NumPy 矩阵乘法毫秒级；配置里预留 `use_faiss: true` 开关，未来向量再多也无需改接口。

### 4.3 为什么保留原分类器

`Predictor`（封闭集分类）继续保留，作为**电影帧的精确识别通道**；图库检索负责**开放集与多类型图片**。统一服务层会同时调用两者，帧类图片以分类器结果为准，其他类型以检索结果为准。

---

## 5. 演员识别（两阶段）

### 阶段一：演员照 → 相关电影（立即可用）

演员照直接进图库检索：如果该演员的参考照已入库并标注了 `person_id`，检索时不仅返回电影，还返回演员身份。

### 阶段二：演员照 → 认出是谁（本次实现骨架）

1. 用 OpenCV 自带的 Haar 级联（`haarcascade_frontalface_default.xml`，随 opencv 安装，零下载）检测并裁剪人脸区域；
2. 对人脸区域提特征，与**演员参考图库**比对，返回最相似的演员；
3. 再由 `movie_cast` 查出该演员参演的所有电影。

> **精度说明**：ResNet18 的通用图像特征不是专用人脸特征，跨年龄/妆容/角度的识别精度有限。代码里把编码器抽象成 `FaceEncoder` 接口，未来可一键替换为 InsightFace/ArcFace 等专用模型而不改上层逻辑。当前实现定位为**可用的骨架**，不是生产级人脸识别。

---

## 6. 统一识别服务

`cineeye/identify.py` 的 `Identifier.identify_image/identify_images/identify_video` 编排：

1. 判断输入类型（单图 / 多图 / 视频）；
2. 视频 → 均匀抽帧；
3. 若启用人脸检测 → 检测人脸并尝试识别演员；
4. 图库检索得到电影候选（可指定 `kind` 过滤，如只看海报）；
5. 电影帧额外走一次原分类器，作为高置信证据融合；
6. 查元数据库，补全每部候选电影的：主片名、原名、全部别名（按语言）、演员表、年份、导演、海报路径；
7. 返回统一结构：

```json
{
  "input_type": "image",
  "matches": [
    {
      "movie_id": "movie_04",
      "confidence": 0.83,
      "title": "主片名",
      "original_title": "原文片名",
      "aliases": [{"alias": "...", "lang": "zh", "region": "CN"}],
      "year": 1992,
      "director": "...",
      "cast": [{"name": "...", "character": "..."}],
      "poster": "数据路径或URL",
      "evidence": {"kind": "frame", "similarity": 0.83, "matched_path": "..."}
    }
  ],
  "actor": {"name": "演员名", "confidence": 0.71, "movies": ["movie_04", ...]},
  "elapsed_ms": 123
}
```

---

## 7. REST API 扩展

保留原有 `/health`、`/labels`、`/predict`、`/predict/video` 全部兼容，新增：

| 接口 | 方法 | 用途 |
|---|---|---|
| `/identify` | POST | 统一识别（图片/视频/base64），返回完整电影档案 |
| `/search?q=...` | GET | 按片名/别名搜索电影（模糊匹配） |
| `/movies/{id}` | GET | 查单部电影完整档案 |
| `/movies` | GET | 列出所有电影（分页） |
| `/actors/{id}` | GET | 查演员及其参演电影 |
| `/gallery/stats` | GET | 查看图库规模与索引状态 |

---

## 8. 实施顺序与文件清单

| 阶段 | 文件 | 状态 |
|---|---|---|
| 设计 | `docs/EXPANSION_DESIGN.md` | 已完成（本文） |
| 元数据库 | `cineeye/metadata/{store,providers,normalize}.py` | 已完成 |
| 图库检索 | `cineeye/gallery/{encoder,index}.py` | 已完成 |
| 演员识别 | `cineeye/actors/{face,recognizer}.py` | 已完成（骨架） |
| 统一服务 | `cineeye/identify.py` | 已完成 |
| API | 扩展 `cineeye/serve.py` | 已完成 |
| 脚本 | `scripts/11_import_metadata.py` ~ `scripts/14_identify.py` | 已完成 |
| 配置 | `configs/default.yaml` 增加 `metadata/gallery/actors` 段 | 已完成 |
| 文档 | 更新 `README.md` / `docs/API_GUIDE.md` / 新增 `docs/EXPANSION_GUIDE.md` | 已完成 |

> 当前图库已登记 240 张帧（6 部 × 40 帧，均为 `frame` 类型）；元数据库含 6 部占位档案、6 条别名，演员数据为空（需显式导入）。新增海报/演员照后重建图库即可，无需重训模型。

---

## 9. 设计原则

1. **零新增重依赖**：复用 torch / torchvision / sklearn / cv2 / 标准库；TMDB 用 `urllib`；不装 faiss / transformers / 人脸大模型。
2. **离线优先**：外部数据源只在导入时联网，运行时全部查本地 SQLite 与图库。
3. **开放集**：加新电影 = 加图片 + 入库，不重训。
4. **可替换**：人脸编码器、向量索引、元数据 provider 都是接口，未来可平滑升级。
5. **向后兼容**：原有分类接口与模型保持不变，扩展是叠加层。
