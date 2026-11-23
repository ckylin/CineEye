# 影瞳 CineEye · 扩展功能使用指南

> 从「给一张电影帧问是哪部片」升级为「任何视觉线索 → 电影完整档案」。
>
> 新增能力：海报/演员照/花絮/路演图片检索、多语言片名、演员表、开放集检索（加新片不用重训）。
> 架构设计见 `docs/EXPANSION_DESIGN.md`，本文只讲**怎么用**。

---

## 1. 核心概念

### 1.1 封闭集 vs 开放集

| | 原有 `/predict` | 新增 `/identify` |
|---|---|---|
| 输入 | 电影帧 | 帧 / 海报 / 演员照 / 花絮 / 路演 / 视频 |
| 输出 | `movie_04` + 置信度 | 完整档案：多片名、演员、年份、导演、海报 |
| 新增电影 | 必须重训模型 | **只需加图片 + 入库，不重训** |
| 实现 | 分类头 | 图库向量检索 + 原分类器加权融合 |

### 1.2 三个新增组件

1. **元数据库** `outputs/cineeye.db`（SQLite）：电影、多别名、演员、图片索引；
2. **图库索引** `outputs/gallery/`：240 张已登记图片的 512 维向量（当前 6 部 × 40 帧）；
3. **演员参考库** `outputs/actors/<演员名>/*.jpg`：人脸检测 + 比对（需你放入参考照）。

---

## 2. 完整使用流程

### 步骤 0：环境自检（可选）

```bash
cd /home/kylin/Workarea/CineEye
./.venv/bin/python scripts/00_check_env.py
```

### 步骤 1：导入电影元数据

**方式 A：从已有训练产物生成占位档案**（最快，当前 6 部已导入）

```bash
./.venv/bin/python scripts/11_import_metadata.py --source labelmap
```

> 占位档案只含中性编号 `movie_01~06`，**不含片名**。真实片名/演员请用下面三种方式显式导入，便于控制对外暴露的信息。

**方式 B：本地表格（推荐，离线可控）**

```bash
cp outputs/metadata/movies.csv.example outputs/metadata/movies.csv
# 编辑 movies.csv，按模板填写片名、别名、演员
./.venv/bin/python scripts/11_import_metadata.py --source local --file outputs/metadata/movies.csv
```

CSV 字段：

| 列 | 说明 |
|---|---|
| `movie_id` | 与图库一致的中性编号，如 `movie_01` |
| `primary_title` | 主片名 |
| `original_title` | 原文片名 |
| `year` / `director` / `synopsis` | 年份 / 导演 / 简介 |
| `poster_path` | 海报路径或 URL |
| `aliases` | JSON 数组：`[{"alias":"译名","lang":"zh","region":"CN"}]` |
| `cast` | JSON 数组：`[{"name":"演员","character":"角色"}]` |

**方式 C：TMDB 自动抓取（多语言片名 + 演员 + 海报）**

```bash
export TMDB_API_KEY=你的key
./.venv/bin/python scripts/11_import_metadata.py --source tmdb --titles "电影A" "电影B"
```

**方式 D：豆瓣抓取（无 key，中文片名/别名较全，受反爬影响）**

```bash
./.venv/bin/python scripts/11_import_metadata.py --source douban --titles "电影A" "电影B"
```

> 三个 provider 可混用，按 `movie_id` 覆盖合并。TMDB 返回 `tmdb_<数字ID>` 作为 movie_id，若要对照片库编号，请用本地表格手动指定相同的 `movie_id`。

### 步骤 2：构建图库索引

```bash
# 从元数据库已登记的图片构建（增量）
./.venv/bin/python scripts/12_build_gallery.py

# 全量重建（换模型/编码器时）
./.venv/bin/python scripts/12_build_gallery.py --rebuild

# 追加一批海报，指定类型和电影
./.venv/bin/python scripts/12_build_gallery.py \
  --add-dir data/posters/movie_01 --kind poster --movie-id movie_01

# 只登记图片路径到元数据库（不编码）——先登记再 build
./.venv/bin/python -c "
from cineeye.config import load_config
from cineeye.metadata import MetaStore
s=MetaStore(cfg=load_config())
s.add_image('movie_01','data/posters/movie_01/a.jpg',kind='poster',source='manual')
"
```

图片类型 `kind`：`frame`（影片帧）、`poster`（海报封面）、`actor`（演员照）、`still`（剧照/花絮）、`roadshow`（路演）、`other`。

### 步骤 3：准备演员识别（两种方式，任选）

**方式 A：参考库检索（不训练，5 分钟见效）**

```bash
mkdir -p outputs/actors/演员甲
cp 演员甲的照片.jpg outputs/actors/演员甲/
./.venv/bin/python scripts/13_actor_reference.py --build
```

**方式 B：照片打标签 + 训练（推荐，准确率更高）**

```bash
# 1) 照片按 <演员名>/ 放好（剧照/生活照/写真都行），导入并打标签
./.venv/bin/python scripts/15_import_actors.py --src data/actors
# 2) 训练分类头（CPU 上秒级；会同时生成检索库）
./.venv/bin/python scripts/16_train_actors.py
```

> 方式 B 的完整说明（照片要求、CSV 打标签、阈值调优、多脸识别、HTTP 接口）见
> **[演员识别指南](ACTOR_GUIDE.md)**。两种方式可以共存：训练后会以训练得到的
> 分类头 + train 划分检索库为准。

### 步骤 4：开始识别

```bash
# 一张海报/剧照/演员照（返回完整档案）
./.venv/bin/python scripts/14_identify.py --image 照片.jpg --top-k 3

# 指定类型：只在海报图库里检索
./.venv/bin/python scripts/14_identify.py --image 海报.jpg --kind poster

# 视频（多帧聚合）
./.venv/bin/python scripts/14_identify.py --video 片段.mp4 --num-frames 16

# 只做演员识别（合影/生活照，不需要电影模型）
./.venv/bin/python scripts/14_identify.py --actor-image 合影.jpg

# 按片名/别名搜索（多语言、模糊）
./.venv/bin/python scripts/14_identify.py --search "Example Movie"

# 查看电影档案 / 系统状态
./.venv/bin/python scripts/14_identify.py --movie movie_01
./.venv/bin/python scripts/14_identify.py --status
```

实测示例（识别 movie_04 的一帧）：

```json
{
  "input_type": "images",
  "num_images": 1,
  "matches": [
    {
      "movie_id": "movie_04",
      "confidence": 0.870476,
      "title": "movie_04",
      "original_title": null,
      "aliases": [{"alias": "movie_04", "lang": "id", "region": null, "source": "labelmap"}],
      "year": null, "director": null, "cast": [], "poster": null,
      "evidence": {
        "kind": "frame", "similarity": 0.870476,
        "matched_path": "data/frames/movie_04/seg_03/000005.jpg",
        "classifier_confidence": 0.9736
      }
    }
  ],
  "actor": null,
  "elapsed_ms": 69
}
```

> 帧类图片会把原分类器概率以 40% 权重融合进相似度（60% 图库 + 40% 分类器），避免色调相近的不同影片被图库误匹配。

---

## 3. HTTP 接口（对外提供服务）

启动（加 `--with-identifier` 预热扩展服务）：

```bash
TMPDIR=$PWD/.tmp ./.venv/bin/python scripts/06_serve.py \
  --host 0.0.0.0 --port 8080 --with-identifier
```

新增接口（原有 `/predict` 系列保持兼容）：

| 接口 | 方法 | 用途 |
|---|---|---|
| `/identify` | POST | 统一识别 → 完整电影档案 |
| `/search?q=...` | GET | 按片名/别名搜索（多语言、模糊） |
| `/movies` | GET | 电影列表（支持 `limit`/`offset`） |
| `/movies/{id}` | GET | 单部电影完整档案 |
| `/actors/{id}` | GET | 演员及其参演电影 |
| `/gallery/stats` | GET | 图库规模与状态 |

调用示例：

```bash
# 图片识别（可指定 kind 与是否做演员识别）
curl -X POST http://127.0.0.1:8080/identify \
  -H 'Content-Type: application/json' \
  -d '{"image":"照片.jpg","kind":"poster","top_k":3,"with_actors":false}'

# 视频识别
curl -X POST http://127.0.0.1:8080/identify \
  -H 'Content-Type: application/json' \
  -d '{"video":"/path/片段.mp4","num_frames":16}'

# 别名搜索
curl 'http://127.0.0.1:8080/search?q=Example%20Movie'
```

> 本机有 HTTP 代理时 curl 直连需加 `--noproxy '*'`。完整字段说明见 `docs/API_GUIDE.md`。

---

## 4. 加一部新电影（不重训）

以加入一部新电影为例：

```bash
# 1) 准备图库图片（帧/海报/演员照任意），登记到元数据库
./.venv/bin/python -c "
from cineeye.config import load_config
from cineeye.metadata import MetaStore
s=MetaStore(cfg=load_config())
s.upsert_movie('movie_07','新电影', year=2024,
  aliases=[{'alias':'New Movie','lang':'en'},{'alias':'新片别名','lang':'zh'}],
  cast=[{'name':'主演甲'}], provider='manual')
s.add_image('movie_07','data/posters/movie_07/a.jpg',kind='poster')
s.add_image('movie_07','data/posters/movie_07/b.jpg',kind='poster')
"

# 2) 增量构建图库（只编码新图）
./.venv/bin/python scripts/12_build_gallery.py

# 3) 立即可用（无需训练）
./.venv/bin/python scripts/14_identify.py --image 新海报.jpg --kind poster
```

---

## 5. Python API

```python
from cineeye.identify import Identifier

ident = Identifier()

# 图片识别
r = ident.identify_image("海报.jpg", kind="poster", top_k=3)
for m in r["matches"]:
    print(m["movie_id"], m["confidence"], m["title"], m["aliases"], m["cast"])

# 视频识别
r = ident.identify_video("片段.mp4", num_frames=16)

# 别名搜索（多语言）
for h in ident.search_movies("Example Movie"):
    print(h["movie_id"], h["score"], h["matched_alias"])

# 系统状态
print(ident.status())
```

---

## 6. 常见问题

**Q：为什么当前演员数是 0？**
A：演员信息需要显式导入（本地表格 / TMDB / 豆瓣）。占位档案只有中性编号，不含演员。

**Q：识别结果 `original_title` 是 null？**
A：占位导入刻意不写源文件名。填入真实片名请用方式 B/C/D。

**Q：图库检索和分类器结果不一致？**
A：帧类图片以融合结果为准（60% 图库 + 40% 分类器）；海报/演员照等只有图库结果（分类器没在这些图片上训练）。

**Q：几十万张图片还快吗？**
A：当前 NumPy 矩阵乘法在几千部规模下毫秒级。上万部时可把 `gallery.use_faiss` 设为 true 并另装 faiss。

**Q：演员识别不准？**
A：当前是通用特征骨架。放入足够参考照（每人 3~5 张不同角度）会改善；生产级请接入 InsightFace（接口已预留）。

**Q：TMDB/豆瓣失败怎么办？**
A：两者失败都安全跳过，不影响主流程。TMDB 检查 key 和网络，豆瓣调大 `metadata.providers.douban.request_interval`。

---

## 7. 相关文档

- `docs/EXPANSION_DESIGN.md` — 架构与数据模型设计
- `docs/MODEL_GUIDE.md` — 模型文件与基础用法
- `docs/TRAINING_GUIDE.md` — 训练流程
- `docs/API_GUIDE.md` — HTTP 接口完整字段说明
