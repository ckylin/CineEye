# 演员识别指南（照片打标签 → 训练 → 识别）

> 这份文档解决一件事：**你上传演员照片、给照片打上"这是谁"的标签，系统训练之后就能在新照片里认出这位演员。**
> 照片不限于剧照——生活照、路透、写真、红毯照、合影都可以，模型只关心"这张脸是谁"。

---

## 1. 它能做什么 / 不能做什么

| 能做 | 说明 |
|---|---|
| 单张照片认人 | 输入一张照片，返回画面里每张脸对应的演员名、置信度和候选 |
| 合影/多脸 | 一张图里多个人，逐脸识别，按人脸面积从大到小输出 |
| 生活照/全身照 | 人脸检测失败时可用中心裁剪兜底（`actors.fallback_whole_image`） |
| 说"不认识" | 置信度低于阈值时返回 `unknown`，不会硬猜一个名字 |
| 关联参演电影 | 演员名在 `persons` 表里时，一并返回 `person_id` 与参演作品 |

| 不能做 / 有局限 | 原因与建议 |
|---|---|
| 极其精准的工业级人脸识别 | 当前特征是 ImageNet 通用主干（ResNet18），不是 ArcFace 专用人脸模型；同卵双胞胎、极端妆造、重度遮挡会混淆 |
| 侧脸、低头、大角度 | Haar 检测器只擅长正脸；建议每位演员多准备正脸样本，或后续接入 DNN/InsightFace 检测 |
| 只给 1~2 张照片就想很准 | 建议每位演员 **≥ 8 张**、至少 3 张；样本越多样（不同发型/年龄/光线）泛化越好 |
| 认没训练过的新演员 | 分类头只认识训练集里的人；新增演员必须重新导入 + 重训（或只用参考库检索模式） |

> 一句话定位：**这是"可用、可运维、可迭代"的演员识别能力，不是安防级人脸系统。** 数据量上来后，换更强的人脸特征（`face.FaceEncoderStub` 预留接口）即可升级，上层流程不用改。

---

## 2. 准备照片

### 2.1 数量与质量

- **每人 8~30 张**最佳；少于 3 张只能进训练集，无法评估效果。
- 尽量覆盖：正脸、轻微侧脸、不同发型、不同年龄段、室内/室外光线、笑/不笑。
- 避免：严重模糊、脸太小（小于画面 5%）、大面积遮挡、黑白与彩色混杂过多。
- 合影也能用：系统会自动裁出**最大人脸**作为该图样本；但一张合影只取一张脸，多人合影建议手动裁成单人图再入库。

### 2.2 目录结构（方式 A：目录即标签）

```
data/actors/
├── 刘德华/
│   ├── 001.jpg
│   ├── 002.jpg
│   └── 2024活动/红毯.jpg      # 允许再分子目录，方便按年份/来源整理
├── 梁朝伟/
│   └── ...
└── 周星驰/
    └── ...
```

**子目录名 = 演员名 = 标签。** 名字要和元数据库里的人名一致，否则识别出来了也查不到参演电影。

### 2.3 CSV 清单（方式 B：照片可以散落各处）

```csv
path,actor,name_en
/data/photos/a1.jpg,刘德华,Andy Lau
/data/photos/b2.jpg,梁朝伟,Tony Leung
```

- `path` 支持相对路径（相对 **CSV 文件所在目录**）或绝对路径；
- `name_en` 可省略；
- 适合照片已经在 NAS/移动硬盘、不想复制进项目的情况（导入时加 `--no-copy`）。

---

## 3. 打标签导入

```bash
# 方式 A：目录即标签（推荐，最省事）
./.venv/bin/python scripts/15_import_actors.py --src data/actors

# 方式 B：CSV 清单
./.venv/bin/python scripts/15_import_actors.py --csv photos.csv

# 先预览，不落盘（标签打错了能及时改）
./.venv/bin/python scripts/15_import_actors.py --src data/actors --dry-run

# 照片位置长期不变，只登记不复制
./.venv/bin/python scripts/15_import_actors.py --csv photos.csv --no-copy

# 已经分好 train/val/test 的话，把比例设 0（全部进 train，稍后手动改 manifest）
./.venv/bin/python scripts/15_import_actors.py --src data/actors --val-ratio 0 --test-ratio 0
```

它做了这些事：

1. **校验**：读不出来的图、小于 48px 的图会被跳过并计入 `invalid`；
2. **去重**：按文件内容 MD5，同一张图换名字重复导入只保留一份（避免刷高某人的样本数）；
3. **归档**：默认复制到 `data/actors/<演员名>/<内容哈希>.jpg`，数据集自包含，迁移只拷一个目录；
4. **登记演员**：写入 `outputs/cineeye.db` 的 `persons` 表，拿到稳定的 `person_id`（`--no-db` 可关闭）；
5. **分层划分**：每位演员按比例切 `train/val/test`，保证验证集里的演员训练时见过；
6. **写清单**：`data/actors/manifest.csv`，训练唯一数据源。

输出摘要示例：

```json
{
  "images": 96,
  "actors": 8,
  "duplicates": 3,
  "invalid": 0,
  "per_actor": {"刘德华": {"train": 8, "val": 2, "test": 2, "total": 12}},
  "face_detected_rate": 0.94
}
```

> `face_detected_rate` 低于 0.8 时要留意：说明大量照片没检测到人脸（侧脸/脸太小/非人像），
> 这些样本会退化为中心裁剪，效果会打折。可以先用 `--no-face-check` 跳过，但更建议换照片。

---

## 4. 训练

```bash
# 默认读 data/actors/manifest.csv，200 轮
./.venv/bin/python scripts/16_train_actors.py

# 常用调参
./.venv/bin/python scripts/16_train_actors.py --epochs 300 --lr 0.02
./.venv/bin/python scripts/16_train_actors.py --rebuild-cache   # 照片改过，强制重算特征
./.venv/bin/python scripts/16_train_actors.py --eval-only       # 只评估已有模型
```

### 4.1 训练在做什么

1. **人脸裁剪**：按 `actors.crop`（默认 `detect`）取最大人脸，外扩 `face_pad`；
2. **冻结主干提特征**：用 ResNet18（ImageNet 权重，离线可用）把每张脸编码成 512 维向量；
3. **缓存特征**：存入 `outputs/features/actor_features.npz`，签名包含「清单内容 + 编码器指纹 + 裁剪参数」，任一变化自动重算；
4. **训练线性分类头**：在特征上做带类别权重的线性分类（小样本友好），CPU 上通常 1~3 秒；
5. **评估**：train/val/test 的 top-1、top-3，以及混淆矩阵；
6. **估阈值**：默认按验证集置信度分位数估计（有未知样本则真实校准）；
7. **落盘**：`outputs/checkpoints/actor_head.pt` + `outputs/reports/actor_train_report.json` + 检索库。

### 4.2 怎么读报告

```json
{
  "actors": ["刘德华", "周星驰", "梁朝伟"],
  "split_counts": {"train": 72, "val": 12, "test": 12},
  "metrics": {
    "val":  {"n": 12, "top1": 0.9167, "top3": 1.0},
    "test": {"n": 12, "top1": 0.8333, "top3": 1.0}
  },
  "threshold": 0.42,
  "face_detected_rate": 0.94
}
```

- **看 test top-1**，不要看 train：train 高只说明"记住了"，test 才代表新照片上的表现。
- `top3` 明显高于 `top1` → 说明候选里通常有正确答案，问题在"排序"，可加大样本量或调融合权重。
- `threshold` 太高 → 真人也会被判 unknown（漏认）；太低 → 陌生人也被硬认（误认）。见第 6 节。
- 混淆矩阵在 `outputs/reports/actor_train_report.json` 的 `confusion` 字段（也可看 checkpoint），能定位"总把 A 认成 B"。

---

## 5. 识别

### 5.1 命令行（不需要电影模型）

```bash
# 单张 / 多张
./.venv/bin/python scripts/14_identify.py --actor-image 合影.jpg

# 输出：每张图、每张脸的演员名 + 置信度 + 候选 + 参演电影
```

### 5.2 Python

```python
import cv2
from cineeye.actors import ActorRecognizer

rec = ActorRecognizer()                     # 自动读 actors.* 配置
img = cv2.imread("合影.jpg")

detail = rec.recognize_all(img)
for face in detail["faces"]:
    print(face["bbox"], face["actor"], face["confidence"], face["candidates"])

full = rec.recognize_with_movies(img)       # 只取最大人脸，带参演电影
print(full["name"], full["person_id"], full["movies"])

print(rec.status())                          # 模式/演员列表/阈值/编码器
```

### 5.3 HTTP

```bash
# 启动（演员接口不依赖 best.pt，可单独部署）
./.venv/bin/python scripts/06_serve.py --port 8080 --with-identifier

# 只做演员识别
curl -X POST http://127.0.0.1:8080/actors/recognize \
  -H 'Content-Type: application/json' \
  -d '{"images": ["/path/a.jpg", "/path/b.jpg"]}'

# base64 也可以
curl -X POST http://127.0.0.1:8080/actors/recognize \
  -H 'Content-Type: application/json' \
  -d '{"image_base64": "<base64>"}'

# 上传一张照片并打标签（前端/脚本可直接调）
curl -X POST http://127.0.0.1:8080/actors/photos \
  -H 'Content-Type: application/json' \
  -d '{"actor": "刘德华", "image_base64": "<base64>", "name_en": "Andy Lau"}'

# 演员名册 + 识别器状态
curl http://127.0.0.1:8080/actors
```

上传接口只做**增量入库**（按内容去重、默认 `split=train`），不会重新分层。
攒够一批后仍需重跑 `15_import_actors.py`（重新划分）与 `16_train_actors.py`（重训），
新演员才会被分类头认识——接口返回里的 `next_step` 也会提示这一点。

响应结构：

```json
{
  "ok": true,
  "data": {
    "num_images": 1,
    "results": [{
      "source": "a.jpg",
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

### 5.4 和电影识别一起用

`/identify` 与 `Identifier.identify_images/identify_video` 的 `actor` 字段会自动带上演员结果（多脸时在 `faces` 里），无需额外调用。

---

## 6. 调优：认不出、认错、乱认怎么办

| 现象 | 优先检查 | 处理 |
|---|---|---|
| 真人被拒（unknown） | `threshold` 是否过高；照片是否侧脸/太暗 | 降低 `actors.unknown_threshold`（如 0.5 → 0.35）；补正脸样本重训 |
| 陌生人被硬认成某演员 | `threshold` 是否过低 | 提高阈值；在 `actors.unknown_dir` 放 10~30 张"非目标演员"照片后重训，阈值会自动校准 |
| A 总被认成 B | 两人样本是否太少/太像；混淆矩阵 | 增加 A 的多角度样本；调低 `actors.fusion.classifier`（更依赖检索）或反之 |
| 检测不到脸 | `face_detected_rate` 低 | 换更清晰的正脸照；或 `actors.crop: center` + `fallback_whole_image: true` |
| 换了 GPU 后结果变了 | 编码器是否一致 | 特征与设备无关，结果应一致；若不一致请检查 `actors.encoder` / `backbone` 是否被改过（指纹会拒绝旧模型） |

### 关键配置（`configs/default.yaml` 的 `actors` 段）

```yaml
encoder: imagenet        # imagenet（离线）/ movie（复用 best.pt 主干）/ auto
backbone: resnet18       # 换主干必须重训
image_size: 224          # 改分辨率也必须重训
crop: detect             # detect / center / whole
fallback_whole_image: true
unknown_threshold: null  # null=训练时自动估；也可写死 0.35
fusion:
  classifier: 0.7        # 分类头权重
  gallery: 0.3           # 检索相似度权重
```

> 融合逻辑：`score = 0.7 × 分类概率 + 0.3 × max(0, 余弦相似度)`。
> 样本多、标签干净 → 提高 `classifier`；新演员多、来不及重训 → 提高 `gallery`。

---

## 7. 常见问题

**Q：一定要训练吗？只放几张参考图行不行？**
A：行。不训练时系统用"参考库检索"模式：把照片放到 `outputs/actors/<演员名>/`，跑
`scripts/13_actor_reference.py --build`，即可识别。训练（15→16）会额外得到一个分类头，
通常更准、更稳，尤其在你已有 8 张以上/人的时候。

**Q：训练要多久？**
A：CPU 上典型 100~200 人以内、每人 10~30 张，编码阶段几十秒到几分钟，分类头训练 1~3 秒。
GPU 上编码更快（`--device cuda`），分类头仍在 CPU（线性层太小，没必要占显存）。

**Q：加了新演员怎么办？**
A：把新演员照片放进 `data/actors/<新演员名>/`，重跑 15（重新生成 manifest）→ 16（重训）。
不需要删旧数据，去重与分层会自动处理。

**Q：删了某位演员的照片，模型还认识他吗？**
A：重跑 15 后 manifest 里没有他了，重跑 16 后分类头也不再包含他；但**旧 checkpoint 仍在**，
识别器加载的是最新 `actor_head.pt`，所以重训后即生效。

**Q：为什么 `person_id` 是 null？**
A：照片目录名/CSV 里的演员名在 `persons` 表里找不到（或导入时用了 `--no-db`）。
名字改成与元数据一致，或用 11_import_metadata.py 的 `cast` 字段补录即可。

**Q：能识别视频里的演员吗？**
A：可以，但视频抽帧多为远景，人脸小、侧脸多，效果不如照片。建议用 `--video` 走电影识别，
再用返回的演员表补全；或对关键帧单独跑 `--actor-image`。

**Q：数据安全吗？照片会被上传吗？**
A：不会。全流程本地运行，不联网（`encoder: imagenet` 用的是本地缓存的权重）。

---

## 8. 产物与运维

| 路径 | 内容 | 何时更新 |
|---|---|---|
| `data/actors/<演员名>/*.jpg` | 照片归档 | 导入时 |
| `data/actors/manifest.csv` | 带标签清单（唯一数据源） | 每次 15 |
| `outputs/features/actor_features.npz` | 特征缓存（带签名） | 自动 |
| `outputs/checkpoints/actor_head.pt` | 分类头 + 标签表 + 阈值 + 指标 + 编码器指纹 | 每次 16 |
| `outputs/reports/actor_train_report.json` | 训练报告（含混淆矩阵、阈值、检出率） | 每次 16 |
| `outputs/actors/embeddings.npy` | 检索库向量（仅 train 划分） | 每次 16 |
| `outputs/actors/names.json` | 与向量逐行对应的演员名 | 每次 16 |
| `outputs/actors/encoder.json` | 检索库的编码器指纹 | 每次 16 / 13 |

**指纹机制**：分类头与检索库都记录 `encoder` 指纹（模式+主干+维度+分辨率）。
如果你换了 `backbone` / `image_size` / `encoder`，旧产物会被识别器判定为不兼容并**忽略**（日志有 warning），
此时必须重训——这是为了避免"用 A 的向量查 B 的库"这种静默错误。

**备份建议**：`data/actors/`（照片+清单）+ `outputs/checkpoints/actor_head.pt` + `outputs/cineeye.db` 三样齐全即可完整恢复。

---

## 9. 升级路线

1. **换更强的人脸特征**：接入 InsightFace/ArcFace（`actors/face.py` 的 `FaceEncoderStub` 已预留
   `encode(faces) -> (N,D)` 约定），替换 `ActorEncoder` 的实现即可，训练与识别流程不变；
2. **换人脸检测器**：DNN/RetinaFace 替代 Haar，侧脸与遮挡场景会明显改善；
3. **度量学习**：样本量上千后，可把线性分类头换成 ArcFace/CosFace 训练，天然支持开放集；
4. **多帧投票**：视频场景对同一演员的多帧结果做加权投票，降低单帧误判。

---

## 10. 相关文档

- [扩展功能使用指南](EXPANSION_GUIDE.md) — 电影档案、多别名、图库检索、HTTP 接口
- [模型文件与使用指南](MODEL_GUIDE.md) — 模型产物位置与加载方式
- [训练与验证实操指南](TRAINING_GUIDE.md) — 电影识别模型的训练流程
- [GPU 使用指南](GPU_GUIDE.md) — RTX 50 系环境与加速
