
# 🎡 迪士尼多模态 RAG 助手

> 面向企业客服场景的多模态检索增强生成（RAG）系统
> 支持 **文本 / 图片 / 视频** 混合检索 · **三阶段检索流水线** · 端到端可运行

**🌐 语言**: [English](./README.md) · **简体中文**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FAISS](https://img.shields.io/badge/FAISS-IndexFlatIP-0467DF)](https://github.com/facebookresearch/faiss)
[![Reranker](https://img.shields.io/badge/Reranker-bge--reranker--base-FF6B6B)](https://modelscope.cn/models/BAAI/bge-reranker-base)
[![LLM](https://img.shields.io/badge/LLM-qwen3.8--max-9B59B6)](https://help.aliyun.com/zh/model-studio/)

---

## 📌 项目一句话

**用向量 + BM25 双通道召回、RRF 融合、Cross-Encoder 精排，构建可处理文本/图片/视频混合查询的 RAG 系统。**

知识库 **70 条记录**（67 文本 chunks + 2 图片 + 1 视频），覆盖 docx / pdf / 图片 / 视频多格式。

在 4 个多模态测试用例上，**文本 Rerank 精确命中（0.8976）**，**媒体独立检索 100% 命中**（视频 0.1878、图片 0.1856），且**无幻觉**。

---

## 🎯 核心问题与解法

| 业务挑战 | 我的解法 |
|---------|---------|
| 单通道向量检索易召回"泛相关"噪声 | **双通道召回**：向量（语义）+ BM25（关键词） |
| 两路分数**量纲不同**，无法直接加权 | **RRF（倒数排名融合）**，只用排名信息，天然无量纲 |
| Bi-Encoder 语义偏差，精度不够 | **Cross-Encoder Reranker** 精排 Top-20 → Top-3 |
| **媒体被大量文本淹没**（跨模态相似度天然低） | **媒体独立检索通道**：全局扫描 + 阈值筛，不参与 Top-K 竞争 |
| 图片/视频跨模态，相似度天然低 | **分模态阈值**（媒体 0.10 vs 文本 0.3+） |
| LLM 幻觉 | **Prompt 约束 + 严格检索质量保证** |
| 服务端 500 中断批量入库 | **指数退避重试 + 单条跳过 + 主动限流** |

---

## 🏗️ 系统架构

```
                      用户 Query
                          │
             ┌────────────┴────────────┐
             │                         │
      ┌──────▼───────┐         ┌───────▼───────┐
      │  文本通道      │         │  媒体通道      │
      │              │         │   (独立)      │
      │  向量 + BM25  │         │  全局扫描      │
      │      ↓       │         │      ↓        │
      │  RRF 融合     │         │  阈值筛       │
      │      ↓       │         │      ↓        │
      │ Reranker 精排 │         │ Top-1 图片    │
      │      ↓       │         │ Top-1 视频    │
      │   Top-3      │         │              │
      └──────┬───────┘         └───────┬───────┘
             │                         │
             └────────────┬────────────┘
                          │
                  ┌───────▼────────┐
                  │  LLM 生成答案    │  ← 只基于文本
                  │  + 媒体附加展示   │  ← 媒体作参考
                  └────────────────┘
```

**技术栈**：Python 3.11 · FAISS · rank_bm25 + jieba · sentence-transformers · 阿里云百炼

### 双通道输出设计

系统采用 **"文本通道 + 媒体通道"独立并行**的设计，**不做融合**：

| 通道 | 输入 | 处理 | 输出 | 用途 |
|------|------|------|------|------|
| **文本通道** | Query | 向量 + BM25 → RRF → Reranker | Top-3 文本 chunk | **LLM 生成的依据** |
| **媒体通道** | Query + 意图 | 全局扫描 + 阈值筛 | Top-1 图片 + Top-1 视频 | **附加展示**（不参与生成） |

**为什么不做融合？**

1. **量纲不同**：Cross-Encoder 分数（0~1）vs 跨模态余弦相似度（0.1~0.2），无法直接比较
2. **职责不同**：文本是"生成源"，媒体是"参考资源"——**文本驱动答案，媒体补充展示**
3. **冲突处理**：当文本与媒体冲突时，**文本优先**（Reranker 精度远高于跨模态匹配）

### 媒体独立检索设计

**问题**：知识库从 18 条扩到 70 条后，媒体因跨模态相似度天然低（0.1~0.2），在文本 chunk（0.4~0.7）中被淹没在 Top-K 之外。

**解决方案**：媒体走**独立通道** —— 全局扫描该类型所有媒体，按阈值筛选，**不参与文本 Top-K 竞争**。

```python
def find_best_media(index, metadata, media_type, query, threshold=0.10):
    """全局扫描所有该类型媒体，取相似度最高的一个"""
    qvec = np.asarray([get_text_embedding(query)], dtype="float32")
    sims, idxs = index.search(qvec, index.ntotal)   # ← 扫描全库，不走 Top-K

    best = None
    for idx, sim in zip(idxs[0], sims[0]):
        m = metadata[idx]
        if m["type"] != media_type:                  # ← 只挑目标类型
            continue
        if float(sim) < threshold:
            continue
        if best is None or float(sim) > best["score"]:
            best = {"idx": int(idx), "score": float(sim), "metadata": m}
    return best
```

---

## ⚙️ 关键技术决策

### 1️⃣ 为什么用三阶段检索？

**单通道召回有偏差**：纯向量会被"泛化语义"带偏（"安全须知"跟任何迪士尼问题都有 0.5+ 相似度），纯 BM25 对同义词无能为力。

**三阶段各自解决一个问题**：

- **双通道** → 提升**召回率**：向量 + BM25 信号互补
- **RRF 融合** → 解决**量纲**：向量是 [0,1] 余弦，BM25 是 [0,+∞) 实数，**RRF 只用排名，天然无量纲**
- **Reranker** → 提升**精度**：Bi-Encoder 分别编码无交互，Cross-Encoder 拼接后每 token attend 对方

### 2️⃣ 为什么用 Cross-Encoder 精排？

| | Bi-Encoder | Cross-Encoder |
|--|-----------|---------------|
| 编码方式 | query、doc 分别编码 | `(query, doc)` 拼接后一起编码 |
| Token 交互 | ❌ | ✅ 每 token attend 对方 |
| 复杂度 | O(1) 近似 | O(N) |
| 用途 | 召回 | **精排** |

**实测对比**：

- Query: `"迪士尼门票退款"`
- Doc: `"门票退款政策：提前48小时可免费改期"`

| 模型 | 打分 |
|------|------|
| Bi-Encoder（余弦） | 0.69 |
| **Cross-Encoder** | **0.89** |

**精度提升一个量级**。工业界标准：Bi-Encoder 召回 Top-20 → Cross-Encoder 精排 Top-3。

### 3️⃣ 为什么 Reranker 本地部署？

- **数据合规**：企业场景下 `(query, doc)` 对含原文片段，传给第三方 API 有外泄风险
- **成本可控**：每 query 要算 20 对，API 按 token 计费，QPS 高成本爆炸
- **延迟稳定**：本地推理无网络抖动，P99 延迟可控

### 4️⃣ 多模态怎么处理？

**跨模态 embedding**：文本 / 图片 / 视频统一用 `tongyi-embedding-vision-flash` 编码到同一 **768 维**空间。

**关键工程细节** —— **跨模态相似度天然低**：

| 类型 | 典型相似度 |
|------|-----------|
| 文本 ↔ 文本 | 0.5 ~ 0.7 |
| 文本 ↔ 图片 | 0.1 ~ 0.2 |

**阈值必须分模态设**：文本 0.3+，媒体 **0.10**。统一用 0.3 会导致图片永远匹配不到。

### 5️⃣ 意图识别双层设计

**正则优先，LLM 兜底**：

- **正则**（0 延迟）：`"海报"`、`"视频"`、`"长什么样"` 高频词直接命中
- **LLM 兜底**（~500ms）：`"我想看那个聚会的图"` 这种隐式意图，调 `qwen3.8-flash` 判一次

**整体延迟低、准确率高。**

### 6️⃣ 分块策略

**两阶段分块**：

1. **按空行分段** → `re.split(r'\n\s*\n', text)`
2. **超长段落按句末标点切句** → `re.split(r'(?<=[。！？；.?!])', p)`
3. **贪心合并**成 ~300 字的 chunk
4. **加 50 字 overlap**

**保证每个 chunk 以完整句子开头**，避免语义被切断（例如避免把"未使用"切成"未使"+"用"）。

### 7️⃣ 批量入库的健壮性

- **指数退避重试**：服务端 500 / 429 时，等待 2s → 4s → 8s 重试
- **单条跳过**：某个 chunk 失败不中断整个建库流程
- **主动限流**：每 10 个 chunk 睡 0.5s，避免触发 QPS

---

## 📊 运行效果

### 知识库规模

| 类型 | 数量 | 来源 |
|------|------|------|
| **文本 chunks** | 67 | 4 docx + 1 pdf（香港迪士尼旅游攻略） |
| **图片** | 2 | 万圣节.jpeg + 聚在一起说奇妙.jpg |
| **视频** | 1 | 汽车剐蹭视频 |
| **总计** | **70** | — |

### 测试用例

| # | Query | 类型 | 文本 Rerank Top-1 | 媒体匹配 | 结果 |
|---|-------|------|------------------|---------|------|
| 1 | "迪士尼门票退款流程" | 文本 | **0.8976** | — | ✅ 输出完整退改规则 |
| 2 | "万圣节活动海报" | 图片 | — | **0.1756** | ✅ 命中万圣节.jpeg |
| 3 | "汽车剐蹭视频" | 视频 | — | **0.1878** | ✅ 精准命中 |
| 4 | "聚在一起说奇妙的海报" | 图片 | — | **0.1856** | ✅ 精准命中 |

**说明**：

- **文本 Rerank 分数**（0.8976）为 **Cross-Encoder 相关性分数**
- **媒体匹配分数**（0.1756）为 **跨模态余弦相似度** —— 量纲不同，**不直接比较**
- **两者独立输出**：文本进 LLM 生成，媒体作附加展示

### 关键输出示例

**测试 3（视频查询）—— 媒体独立检索精准命中**：

```
[Rerank Top-3] ★ 文本通道输出
  0.0001 | 和恒星之间刺激兴奋...
  0.0001 | 跑道，而采用的汽车更是第一次被迪士尼使用的电动车...
  0.0000 | 港海洋公园...

意图: 图片=False, 视频=True
  → 独立检索视频通道...
    [video 全局扫描] 共 1 个:
      0.1878 | [视频] 汽车剐蹭视频...
  → 命中视频: .../car.mp4 (sim=0.1878)

最终答案:
抱歉，我暂时没有这方面的资料。

[相关视频]: https://.../car.mp4
```

**测试 1（文本查询）—— 防幻觉生效**：

```
[Rerank Top-3] ★ 文本通道输出
  0.8976 | 春节、圣诞节）老人票需提前3天预约...30天内改期
  0.6178 | 籍游客用护照...证件原件
  0.5103 | 买相片光碟...外币兑换...

最终答案:
现有资料中未说明具体退款申请渠道、操作步骤或到账时间，
仅能查询到以下退票/退改规则：

- 老人票退改规则：未使用门票在有效期届满前 7天可免费退改；
  3天前办理收取 10% 手续费；48 小时内不退不改...
- 上海迪士尼门票退改规则：距入园日 48 小时内不可退改；
  突发疾病等特殊情况需提供二甲以上医院证明...
- 过期门票延期：30 天内可申请补差价延期，需支付 20% 手续费...
```

**有资料就给答案，没资料就承认。这是 RAG 相比纯 LLM 的核心优势。**

---

## 🔧 性能数据

**单次查询延迟**（Mac CPU）：

| 阶段 | 耗时 |
|------|------|
| Query embedding | 200 ~ 500ms |
| FAISS 检索（70 条） | < 5ms |
| BM25 检索 | < 5ms |
| RRF 融合 | < 1ms |
| **Reranker（20 候选）** | **300 ~ 800ms** |
| **媒体全局扫描（每类）** | **200 ~ 500ms** |
| LLM 生成 | 1 ~ 3s |
| **总计** | **2 ~ 5s** |

**内存占用**：

- FAISS 索引（70 条 768 维）：< 1MB
- Reranker 模型（含 PyTorch 运行时）：约 1.5GB

---

## 🎓 工程难点与解决

### 难点 1：区域 endpoint 匹配

**问题**：阿里云百炼有新加坡、日本、中国内地多区域，**key 和 endpoint 必须同区域**，否则 401。

**解决**：显式配置 endpoint，与 key 区域对齐；在控制台确认业务空间的模型授权状态。

### 难点 2：Mac 上 torch segfault

**问题**：`CrossEncoder.predict()` 在 Mac CPU 上因 **OpenMP 多线程冲突**段错误。

**解决**：

```python
# 文件顶部（import 之前）
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 加载时
torch.set_num_threads(1)
CrossEncoder(..., device="cpu")

# 推理时逐条 predict，避免批量并行
```

### 难点 3：分块切点落在句子中间

**问题**：机械按字符数切，导致"未使用"被切成"未使"+"用"，语义断裂。

**解决**：**两阶段分块** —— 先按空行分段，超长段落按句末标点切句，保证每个 chunk 以完整句子开头。

### 难点 4：知识库扩容后媒体检索失效

**问题**：知识库从 18 条扩到 70 条后，媒体在 Top-K 里**完全被文本挤掉**。

**根因**：媒体的跨模态相似度天然低（0.1~0.2），文本 chunk 相似度 0.4~0.7，**媒体竞争力不足**。

**解决**：**媒体走独立通道** —— 全局扫描该类型媒体，按阈值筛选，**不参与文本 Top-K 竞争**。

**这是多模态 RAG 的核心难点**：**不同模态的相似度不在同一量纲，需要分通道处理。**

### 难点 5：批量入库服务端 500 中断

**问题**：PDF 52 chunks 入库时服务端偶发 `InternalError`，单个失败中断整个流程。

**解决**：

- **指数退避重试**（2s → 4s → 8s）
- **单条跳过**（失败不中断）
- **主动限流**（每 10 个 chunk 睡 0.5s）

---

## 📁 项目结构

```
CASE-迪士尼RAG助手/
├── 4-disney_build_index.py      # 知识库构建（docx/pdf/图片/视频 + 重试 + 限流）
├── 5-disney_query.py            # 查询与 RAG 问答（双通道 + RRF + Rerank + 媒体独立）
├── download_reranker.py         # 下载本地 Reranker 模型
├── test_endpoint.py             # endpoint 连通性测试
├── disney_knowledge_base/       # 知识库
│   ├── *.docx                   # 文本知识（4 个）
│   ├── *.pdf                    # PDF 知识（1 个）
│   └── images/                  # 图片知识（2 个）
├── disney_index.faiss           # FAISS 向量索引（构建产物）
├── disney_metadata.json         # 元数据（构建产物）
└── models/                      # 本地 Reranker 权重
```

---

## 🚀 快速开始

### 1. 安装依赖

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
pdfplumber>=0.10.0
python-dotenv>=1.0.0
rank-bm25>=0.2.2
jieba>=0.42.1
sentence-transformers>=2.2.0
modelscope>=1.9.0
```

### 2. 配置 API Key

创建 `.env`：

```dotenv
DASHSCOPE_API_KEY=sk-你的新加坡区域key
```

获取 Key：[阿里云百炼控制台](https://bailian.console.aliyun.com/) → 区域选**新加坡** → API-KEY 管理。

### 3. 下载本地 Reranker

```bash
python download_reranker.py
```

下载 `BAAI/bge-reranker-base`（约 1.1GB）。

### 4. 构建知识库

```bash
python 4-disney_build_index.py
```

输出 `disney_index.faiss` + `disney_metadata.json`。

### 5. 运行 RAG 问答

```bash
python 5-disney_query.py
```

---

## 🔮 后续优化方向

### 短期

- [ ] **VLM 集成**：用 `qwen-vl-max` 让 LLM 真正"看图回答"，突破纯文本 LLM 的局限
- [ ] **Query 改写**：接入前置的"联网搜索改写"模块，提升召回率
- [ ] **流式输出**：LLM 改 SSE 改善交互体验
- [ ] **Web UI**：Streamlit / Gradio 展示结果

### 中期

- [ ] **评估体系**：接入 RAGAS 量化 Faithfulness / Answer Relevancy / Context Precision
- [ ] **Reranker 批处理**：batch_size 优化 + GPU 加速
- [ ] **缓存层**：Redis 缓存 hot query 的检索结果
- [ ] **多轮对话**：支持对话历史上下文
- [ ] **分类过滤**：利用 `category` 元数据做权限过滤 / 分面检索

### 长期

- [ ] **分布式部署**：FAISS → Milvus / Qdrant，支持亿级向量
- [ ] **权限过滤**：企业场景的文档级 ACL，检索前过滤
- [ ] **Agent 化**：多步推理 + 工具调用

---

## 📚 参考资料

- [Reciprocal Rank Fusion (RRF) 论文](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- [BAAI/bge-reranker-base (ModelScope)](https://modelscope.cn/models/BAAI/bge-reranker-base)
- [阿里云百炼文档](https://help.aliyun.com/zh/model-studio/)
- [FAISS Wiki](https://github.com/facebookresearch/faiss/wiki)

---

## 📄 License

仅供学习交流。模型权重版权归原作者所有。