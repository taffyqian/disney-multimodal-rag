# 迪士尼多模态 RAG 助手

> 一个面向企业客服场景的**多模态检索增强生成（RAG）系统**，支持文本、图片、视频混合检索。
> 检索层采用 **向量 + BM25 双通道召回 → RRF 融合 → Cross-Encoder 精排** 的三阶段工业级架构。

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![FAISS](https://img.shields.io/badge/FAISS-IndexFlatIP-green)
![Reranker](https://img.shields.io/badge/Reranker-bge--reranker--base-orange)
![LLM](https://img.shields.io/badge/LLM-qwen3.8--max-purple)

---

## 目录

- [一、项目背景](#一项目背景)
- [二、系统架构](#二系统架构)
- [三、目录结构](#三目录结构)
- [四、快速开始](#四快速开始)
- [五、关键技术点](#五关键技术点)
- [六、运行效果](#六运行效果)
- [七、技术栈](#七技术栈)
- [八、性能与限制](#八性能与限制)
- [九、后续优化方向](#九后续优化方向)
- [十、License](#十license)

---

## 一、项目背景

在迪士尼客服场景中，用户问题覆盖面广：

- **文本类**：门票规则、退改政策、园区攻略、年卡说明
- **图片类**：活动海报、园区地图、商品图
- **视频类**：服务演示、安全须知、应急处理

**单一检索方式无法覆盖全部场景：**

| 检索方式 | 优势 | 局限 |
|---------|------|------|
| 纯向量检索 | 语义泛化能力强 | 关键词精确匹配弱，易召回"泛相关"噪声 |
| 纯 BM25 | 关键词精确匹配 | 同义词、语义变体无能为力 |
| 单模态检索 | 简单统一 | 无法处理图片/视频 |

本项目实现了一套**多模态、多通道、多阶段**的检索系统，并通过 **Cross-Encoder Reranker** 精排，显著提升 Top-K 精度。

**核心解决的问题：**

1. **召回率**：双通道召回覆盖语义和关键词两类信号
2. **精度**：Cross-Encoder 精排解决 Bi-Encoder 的语义偏差
3. **多模态**：跨模态 embedding 统一向量空间
4. **防幻觉**：Prompt 约束 + 严格的检索质量保证

---

## 二、系统架构

### 2.1 整体流程图

```
                       用户 Query
                           │
              ┌────────────┴────────────┐
              │                         │
       ┌──────▼───────┐         ┌───────▼───────┐
       │  通道1：向量   │         │  通道2：BM25   │
       │  Bi-Encoder  │         │ jieba + rank  │
       │   Top-20     │         │   Top-20      │
       └──────┬───────┘         └───────┬───────┘
              │                         │
              └────────────┬────────────┘
                           │
                    ┌──────▼───────┐
                    │   RRF 融合    │  Reciprocal Rank Fusion
                    │   Top-20     │  score = Σ 1/(k + rank)
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │   Reranker   │  Cross-Encoder（本地）
                    │   Top-3      │  BAAI/bge-reranker-base
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  LLM 生成     │  qwen3.8-max
                    │  + 媒体附加   │  防幻觉 Prompt
                    └──────────────┘
```

### 2.2 关键组件

| 阶段 | 组件 | 作用 | 参数 |
|------|------|------|------|
| 召回 | 向量检索 | 语义相似 | `tongyi-embedding-vision-flash`，768 维，Top-20 |
| 召回 | BM25 | 关键词精确匹配 | `jieba` 分词 + `rank_bm25`，Top-20 |
| 融合 | RRF | 无量纲融合两路结果 | k=60，Top-20 |
| 精排 | Reranker | Cross-Encoder 精排 | `BAAI/bge-reranker-base`，Top-3 |
| 生成 | LLM | 基于资料回答 | `qwen3.8-max`，防幻觉 |
| 多模态 | 独立向量通道 + 意图识别 | 图片/视频匹配 | 阈值 0.10 |

### 2.3 多模态处理流程

```
用户 Query
   │
   ├─► 意图识别（正则 + LLM 兜底）
   │      ├─ want_image: "海报"、"图片"、"长什么样"...
   │      └─ want_video: "视频"、"录像"、"播放"...
   │
   ├─► 文本通道：BM25 + 向量 → RRF → Rerank → Top-3
   │
   └─► 媒体通道：向量检索（阈值 0.10）→ Top-1 图片 + Top-1 视频
               │
               └─► 附加到 LLM 答案末尾
```

---

## 三、目录结构

```
CASE-迪士尼RAG助手/
├── .env                          # API Key（gitignore）
├── .gitignore
├── requirements.txt
├── README.md                     # 本文档
│
├── 4-disney_build_index.py       # 知识库构建（入库）
├── 5-disney_query.py             # 查询与 RAG 问答
├── test_endpoint.py              # endpoint 连通性测试
├── download_reranker.py          # 下载本地 Reranker 模型
├── test_reranker.py              # Reranker 单元测试
│
├── disney_knowledge_base/        # 知识库
│   ├── 1-上海迪士尼门票规则.docx
│   ├── 2-迪士尼老人票价规定.docx
│   ├── 3-迪士尼乐园游玩攻略清单.docx
│   ├── 4-上海迪士尼乐园酒店会员制度.docx
│   └── images/
│       ├── 1-聚在一起说奇妙.jpg
│       └── 2-万圣节.jpeg
│
├── disney_index.faiss            # FAISS 向量索引（构建产物）
├── disney_metadata.json          # 元数据（构建产物）
│
└── models/                       # 本地模型
    └── models/
        └── BAAI--bge-reranker-base/
            └── snapshots/master/ # 权重文件（约 1.1GB）
```

---

## 四、快速开始

### 4.1 环境要求

- **Python** 3.11+
- **操作系统**：macOS / Linux
- **账号**：阿里云百炼账号（新加坡区域，已开通 `qwen3.8-*` 和 `tongyi-embedding-vision-flash`）

### 4.2 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

**`requirements.txt`**：

```txt
dashscope>=1.25.0
faiss-cpu>=1.7.4
numpy>=1.26.0
openai>=1.0.0
python-docx>=1.1.0
python-dotenv>=1.0.0
rank-bm25>=0.2.2
jieba>=0.42.1
sentence-transformers>=2.2.0
modelscope>=1.9.0
```

### 4.3 配置 API Key

创建 `.env` 文件：

```dotenv
DASHSCOPE_API_KEY=sk-你的新加坡区域key
```

**获取 Key**：访问 [阿里云百炼控制台](https://bailian.console.aliyun.com/) → 区域选**新加坡** → API-KEY 管理 → 创建新的 API-KEY。

**⚠️ 注意**：`key` 和 `endpoint` **必须在同一区域**，否则 401。

### 4.4 下载本地 Reranker 模型

```bash
python download_reranker.py
```

**下载量**：`BAAI/bge-reranker-base` 约 1.1GB。
**缓存位置**：`models/models/BAAI--bge-reranker-base/snapshots/master/`

**⚠️ 提示**：国内网络下载可能较慢，可考虑使用 [HF-Mirror](https://hf-mirror.com) 加速。

### 4.5 构建知识库

```bash
python 4-disney_build_index.py
```

**输出**：

- `disney_index.faiss` —— FAISS 向量索引
- `disney_metadata.json` —— 元数据（文本 chunk、图片路径、视频 URL）

**预期输出**：

```
--- 构建多模态知识库 V4 ---
Endpoint: https://ws-75fcsrpt6tqktl06.ap-southeast-1.maas.aliyuncs.com/api/v1
Embedding: tongyi-embedding-vision-flash
分块: chunk=300, overlap=50（按段落+句号切）
  处理文档: 1-上海迪士尼门票规则.docx
    长度: 1342, chunk 数: 5
  ...
  处理图片...
    - 1-聚在一起说奇妙.jpg
    - 2-万圣节.jpeg
  处理视频...
    - 汽车剐蹭视频

向量维度: 768（已 L2 归一化）
✅ 索引已保存: .../disney_index.faiss
✅ 元数据已保存: .../disney_metadata.json

完成! 文本:N, 图片:2, 视频:1
```

### 4.6 运行 RAG 问答

```bash
python 5-disney_query.py
```

**首次运行**会加载 Reranker 模型（约 3~5 秒），然后执行 4 个测试用例。

**输出示例**：

```
加载本地 Reranker: .../BAAI--bge-reranker-base/snapshots/master
Reranker 就绪
已加载索引: 18 条, 维度: 768
BM25 索引: 15 条文本

################### 测试1 ###################
Query: 我想了解一下迪士尼门票的退款流程

[检索流水线]
  Step1 召回: 向量 18 | BM25 15
  Step2 RRF: 18
  Step3 Rerank: 3

[向量 Top-5]
  0.6905 | 开周一和节假日...
[BM25 Top-5]
  3.9634 | 迪士尼乐园游玩攻略清单...
[RRF 融合 Top-5]
  0.031778 | 迪士尼乐园游玩攻略清单...
[Rerank Top-3] ★ 最终入选
  0.8870 | 春节、圣诞节）老人票需提前3天预约...30天内改期
  0.6895 | 籍游客用护照...证件原件
  0.5769 | 开周一和节假日...
```

---

## 五、关键技术点

### 5.1 为什么用三阶段检索？

**单通道召回有偏差：**

- **纯向量**：会被"泛化语义"带偏。例如"安全须知"跟任何迪士尼问题都有 0.5+ 相似度 → **Top-K 混入大量泛相关噪声**
- **纯 BM25**：对同义词无能为力。"退款"和"退改"虽然语义相近，但 BM25 只认字面

**三阶段解决：**

1. **双通道召回**：向量 + BM25 互补，**召回率**大幅提升
2. **RRF 融合**：解决两路分数**量纲不一致**的问题
3. **Cross-Encoder 精排**：解决 Bi-Encoder **语义偏差**问题，**精度**提升一个量级

### 5.2 为什么用 RRF 而不是加权求和？

**核心是量纲问题**：

- 向量相似度：**归一化到 [0,1]**
- BM25 分数：**无上界实数**（实测 chunk 间差异可达 7.8 vs 0.5）

**直接加权求和没有物理意义**。要先各自归一化，但归一化会引入**分布偏差**。

**RRF 只用排名信息**：

```
RRF_score(d) = Σ_{r ∈ channels} 1 / (k + rank_r(d))
```

- **不关心原始分数多大多小，只关心相对排名**
- **天然无量纲**
- **对异常值鲁棒**

工业界标准做法：Elasticsearch RRF、LlamaIndex `QueryFusionRetriever` 均采用。

### 5.3 为什么用 Cross-Encoder 而不是继续用 Bi-Encoder？

**两者结构不同**：

| 维度 | Bi-Encoder（向量） | Cross-Encoder（Reranker） |
|------|-------------------|--------------------------|
| 编码方式 | query 和 doc **分别编码** | `(query, doc)` **拼接后一起编码** |
| 交互 | 无 | **每 token attend 对方** |
| 预计算 | ✅ 可预编码 | ❌ 每次重算 |
| 复杂度 | O(1) 近似 | O(N) |
| 精度 | 中 | **高** |
| 用途 | 召回 | 精排 |

**精度实测**：

- Query: "迪士尼门票退款"
- Doc: "门票退款政策：提前48小时可免费改期"

| 模型 | 打分 |
|------|------|
| Bi-Encoder（余弦） | 0.69 |
| **Cross-Encoder** | **0.89** |

**工业标准**：Bi-Encoder 召回 Top-20 → Cross-Encoder 精排 Top-3。**两级筛选，兼顾效率和精度。**

### 5.4 为什么 Reranker 用本地部署？

**三个考虑**：

1. **数据合规**：企业场景下，候选 chunk 含原文片段，传给第三方 API 有**数据外泄风险**
2. **成本可控**：每个 query 要算 20 对 `(query, doc)`。API 按 token 计费，QPS 高成本爆炸
3. **延迟稳定**：本地推理无网络抖动，P99 延迟可控

**代价**：需自己管理 1.1GB 权重。**长期看更划算。**

### 5.5 多模态怎么处理？

**方案对比：**

| 方案 | 做法 | 优点 | 缺点 |
|------|------|------|------|
| **A（采用）** | 跨模态 embedding，文本/图/视频编码到同一空间 | 统一索引、检索简单 | 跨模态相似度天然低，阈值要分开 |
| B | 用 VLM 把图/视频转文本，再走纯文本检索 | 统一文本空间 | 信息损失、离线成本高 |

**关键工程细节：**

- **跨模态相似度天然低**：
  - 文本↔文本：0.5~0.7
  - 文本↔图片：0.1~0.2
- **阈值必须分开设**：
  - 文本：0.3~0.5
  - 媒体：**0.10**

如果统一用 0.3，**图片永远匹配不到**。

### 5.6 意图识别双层设计

**正则优先，LLM 兜底：**

```python
# 第一层：正则（快）
IMAGE_PATTERNS = [r"图片", r"海报", r"照片", r"长什么样", r"给我.*图"]
VIDEO_PATTERNS = [r"视频", r"录像", r"影片", r"播放"]

# 第二层：LLM（全，兜底）
def detect_intent_by_llm(query):
    # 调 qwen3.8-flash 输出 JSON
    ...
```

**优势**：

- 高频词（"海报"、"视频"）**正则直接命中，0 延迟**
- 边界 case（"我想看那个聚会的图"）**LLM 兜底**
- **整体延迟低、准确率高**

### 5.7 防幻觉

**两层保障**：

1. **Prompt 约束**：

   ```
   你是一个迪士尼客服助手。请严格根据以下资料回答用户问题。
   资料中没有的信息，请回答"抱歉，我暂时没有这方面的资料"，不要编造。
   ```

2. **检索质量保证**：检索不到相关 chunk 时，LLM 只能承认

**这就是 RAG 相比纯 LLM 的核心优势：有据可查，无据不编。**

### 5.8 分块策略

**两阶段分块：**

1. **按空行分段**：`re.split(r'\n\s*\n', text)`
2. **超长段落按句末标点切句**：`re.split(r'(?<=[。！？；])', p)`
3. **贪心合并**成 ~300 字的 chunk
4. **加 50 字 overlap**

**保证每个 chunk 以完整句子开头**，避免语义被切断（例如避免把"未使用"切成"未使"+"用"）。

**参数**：`CHUNK_SIZE=300, CHUNK_OVERLAP=50`

### 5.9 Mac 兼容（防 segfault）

`torch` 在 Mac 上和 OpenMP 多线程库冲突，容易在 `CrossEncoder.predict()` 时段错误。

**解决方案**：

```python
# 文件顶部（import 之前）
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# import torch 后
import torch
torch.set_num_threads(1)

# 加载 Reranker
_RERANKER = CrossEncoder(str(RERANKER_PATH), device="cpu", max_length=512)

# 推理（逐条而非批量）
for c in candidates:
    s = reranker.predict([(query, c["metadata"]["content"])])
```

---

## 六、运行效果

### 6.1 测试用例

| # | Query | 类型 | 预期 |
|---|-------|------|------|
| 1 | "我想了解一下迪士尼门票的退款流程" | 文本 | 返回退改规则 |
| 2 | "最近万圣节的活动海报是什么" | 图片 | 匹配万圣节.jpeg |
| 3 | "我的汽车被剐蹭了，你能看到视频么？" | 视频 | 匹配汽车剐蹭视频 |
| 4 | "聚在一起说奇妙的海报" | 图片 | 匹配聚在一起说奇妙.jpg |

### 6.2 检索结果

| # | Rerank Top-1 | Rerank 分数 | 结果 |
|---|-------------|-------------|------|
| 1 | 退改规则 chunk | 0.8870 | ✅ 给出完整退改规则 |
| 2 | 万圣节.jpeg | 0.0921（图片档 Top-1） | ✅ 匹配 |
| 3 | 汽车剐蹭视频 | **0.9998** | ✅ 精准命中 |
| 4 | 聚在一起说奇妙.jpg | **0.9723** | ✅ 精准命中 |

### 6.3 关键输出示例

**测试 3（视频查询）**：

```
[Rerank Top-3] ★ 最终入选
  0.9998 | [视频] 汽车剐蹭视频...
  0.0002 | [图片] 2-万圣节.jpeg...
  0.0000 | [图片] 1-聚在一起说奇妙.jpg...

意图: 图片=False, 视频=True
  → 视频: https://.../car.mp4 (sim=0.1878)

最终答案:
很抱歉听到您的车辆被剐蹭。我这边资料中显示有一个名为"汽车剐蹭视频"
的视频，但我无法进一步查看或确认视频里的具体内容。

[相关视频]: https://.../car.mp4
```

**测试 1（文本查询）**：

```
[Rerank Top-3] ★ 最终入选
  0.8870 | 春节、圣诞节）老人票需提前3天预约...
  0.6895 | 籍游客用护照...
  0.5769 | 开周一和节假日...

最终答案:
根据现有规则，可以为您提供以下退改信息：

- 老人票退改规则：未使用门票在有效期届满前 7天可免费退改；
  距有效期 3天前办理需收取10%手续费；48小时内不退不改...
- 上海迪士尼门票退改规则：提前 48小时可免费修改一次日期...
- 过期未使用门票：30天内可申请补差价延期，需支付20%手续费...

如果您想了解具体在哪里提交退款申请、需要上传哪些材料、退款到账时间等
详细流程，抱歉，我暂时没有这方面的资料。
```

---

## 七、技术栈

| 层级 | 技术 | 版本 |
|------|------|------|
| 语言 | Python | 3.11+ |
| 向量库 | FAISS (IndexFlatIP) | 1.7.4+ |
| Embedding | 阿里云 `tongyi-embedding-vision-flash` | API |
| BM25 | `rank_bm25` + `jieba` | - |
| Reranker | `BAAI/bge-reranker-base` | 本地 |
| LLM | 阿里云 `qwen3.8-max` | API |
| 框架 | 纯 Python（无 LangChain 依赖） | - |

---

## 八、性能与限制

### 8.1 性能数据

**单次查询延迟**（Mac CPU）：

| 阶段 | 耗时 |
|------|------|
| Query embedding | 200~500ms |
| FAISS 检索（18 条） | < 5ms |
| BM25 检索 | < 5ms |
| RRF 融合 | < 1ms |
| **Reranker（20 候选）** | **300~800ms** |
| LLM 生成 | 1~3s |
| **总计** | **2~4s** |

**内存占用**：

- FAISS 索引：< 1MB
- Reranker 模型：约 1.5GB（含 PyTorch 运行时）

### 8.2 当前限制

1. **知识库规模小**：仅 4 docx + 2 图片 + 1 视频，共 18 条记录
2. **Reranker 仅支持文本**：图片/视频靠独立向量通道匹配，不参与精排
3. **意图识别正则覆盖率有限**：边界 case 依赖 LLM 兜底
4. **无评估体系**：缺量化指标（Faithfulness、Context Recall 等）
5. **非流式输出**：LLM 生成是一次性返回，不是流式

---

## 九、后续优化方向

### 9.1 短期优化

- [ ] **全量知识库接入**：扩展到 `迪士尼RAG知识库（完整）`，含 PDF、PPT、DOC 多格式
- [ ] **Query 改写**：衔接前置的"联网搜索改写"模块，提升召回率
- [ ] **媒体权重补偿**：Reranker 后加媒体权重，解决文本压过媒体
- [ ] **流式输出**：LLM 改 SSE，改善交互体验
- [ ] **结果可视化**：用 Streamlit / Gradio 做 Web UI

### 9.2 中期优化

- [ ] **评估体系**：接入 RAGAS 框架，量化 Faithfulness / Answer Relevancy / Context Precision
- [ ] **Reranker 批处理**：batch_size 优化 + GPU 加速
- [ ] **缓存层**：Redis 缓存 hot query 的检索结果
- [ ] **多轮对话**：支持对话历史上下文

### 9.3 长期优化

- [ ] **分布式部署**：FAISS → Milvus / Qdrant，支持亿级向量
- [ ] **权限过滤**：企业场景的文档级 ACL，检索前过滤
- [ ] **Agent 化**：多步推理 + 工具调用
- [ ] **混合精度**：Reranker FP16 推理，降低内存占用

---

## 十、License

仅供学习交流。模型权重版权归原作者所有。

---

## 附录 A：常见问题（FAQ）

### Q1：报错 `401 Invalid API-key`

**原因**：key 和 endpoint 区域不匹配。

**解决**：

1. 确认控制台区域（新加坡 / 日本 / 中国内地）
2. 确认 `.env` 里的 key 属于该区域
3. 确认代码里的 endpoint 对应区域：
   - 新加坡：`ws-xxx.ap-southeast-1.maas.aliyuncs.com`
   - 日本：`ws-xxx.ap-northeast-1.maas.aliyuncs.com`

### Q2：报错 `Model not exist`

**原因**：该模型在业务空间未授权。

**解决**：

1. 访问百炼控制台 → 业务空间管理 → 默认业务空间
2. 搜索模型名（如 `qwen3.8-max`、`tongyi-embedding-vision-flash`）
3. 若"未授权"，点"编辑"→ 勾选"模型调用"→ 保存

### Q3：Reranker 加载报 `FileNotFoundError`

**原因**：模型未下载。

**解决**：

```bash
python download_reranker.py
```

### Q4：Mac 上 `Segmentation fault`

**原因**：torch + OpenMP 线程冲突。

**解决**：见 [5.9 节](#59-mac-兼容防-segfault)。

### Q5：图片/视频匹配不到

**原因**：阈值太高。

**解决**：`MEDIA_SIM_THRESHOLD` 设为 `0.10`（跨模态相似度天然低）。

---

## 附录 B：项目演进历程

| 版本 | 关键改进 |
|------|---------|
| V1 | 纯向量检索 + 简单 Prompt |
| V2 | 加 L2 归一化 + 余弦相似度 |
| V3 | 加正则意图识别 + 媒体匹配 |
| V4 | 两阶段分块（段落 + 句号） |
| **V5** | **加 BM25 双通道 + RRF 融合 + 本地 Reranker** |

---

## 附录 C：参考资料

- [Reciprocal Rank Fusion (RRF) 论文](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- [BAAI/bge-reranker-base (ModelScope)](https://modelscope.cn/models/BAAI/bge-reranker-base)
- [阿里云百炼文档](https://help.aliyun.com/zh/model-studio/)
- [FAISS Wiki](https://github.com/facebookresearch/faiss/wiki)