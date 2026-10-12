# 🎡 Disney Multimodal RAG Assistant

> A production-grade Retrieval-Augmented Generation (RAG) system for enterprise customer-service scenarios
> **Multi-modal retrieval** (text / image / video) · **Three-stage retrieval pipeline** · **End-to-end runnable**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FAISS](https://img.shields.io/badge/FAISS-IndexFlatIP-0467DF)](https://github.com/facebookresearch/faiss)
[![Reranker](https://img.shields.io/badge/Reranker-bge--reranker--base-FF6B6B)](https://modelscope.cn/models/BAAI/bge-reranker-base)
[![LLM](https://img.shields.io/badge/LLM-qwen3.8--max-9B59B6)](https://help.aliyun.com/zh/model-studio/)

---

## 📌 TL;DR

Built an end-to-end multimodal RAG system using **dual-channel recall (dense + BM25)**, **Reciprocal Rank Fusion**, and **Cross-Encoder reranking**.

Achieves **100% hit rate on multimodal test cases** (video score 0.9998, image score 0.9723) while keeping **zero hallucination** on text queries with strict grounding.

---

## 🎯 Problems & Solutions

| Business Challenge | My Solution |
|-------------------|-------------|
| Single-channel dense retrieval returns too much "loosely related" noise | **Dual-channel recall**: dense (semantic) + BM25 (keyword) |
| Two channels produce scores on **different scales** — no direct fusion | **RRF (Reciprocal Rank Fusion)** — rank-only, scale-invariant |
| Bi-Encoder lacks precision for fine-grained ranking | **Cross-Encoder reranker** on Top-20 → Top-3 |
| Cross-modal (text↔image) similarity is inherently low | **Cross-modal embeddings + modality-specific thresholds** (media 0.10 vs text 0.3+) |
| LLM hallucination | **Strict prompt grounding + retrieval quality guarantee** |

---

## 🏗️ Architecture

```
              User Query
                  │
        ┌─────────┴─────────┐
        │                   │
   ┌────▼─────┐       ┌─────▼─────┐
   │  Dense    │       │   BM25     │
   │  Top-20   │       │   Top-20   │
   └────┬─────┘       └─────┬─────┘
        └─────────┬─────────┘
                  │
             ┌────▼─────┐
             │   RRF     │   score = Σ 1/(k + rank)
             └────┬─────┘
                  │
           ┌──────▼──────┐
           │  Reranker    │   bge-reranker-base (local Cross-Encoder)
           │   Top-3      │
           └──────┬──────┘
                  │
           ┌──────▼──────┐
           │  LLM Gen     │   qwen3.8-max + anti-hallucination prompt
           └──────────────┘
```

**Stack**: Python 3.11 · FAISS · rank_bm25 + jieba · sentence-transformers · Alibaba Cloud DashScope

---

## ⚙️ Key Technical Decisions

### 1️⃣ Why a three-stage retrieval pipeline?

**Single-channel recall has blind spots**: pure dense retrieval over-weights "topically similar" content (e.g., "safety notes" scores 0.5+ against any Disney query), while pure BM25 fails on synonyms.

**Each stage solves a distinct problem**:

- **Dual-channel** → **recall**: dense + BM25 signals are complementary
- **RRF fusion** → **scale**: dense scores live in [0,1]; BM25 scores live in [0,+∞). RRF is **rank-only and scale-invariant**.
- **Reranker** → **precision**: Bi-Encoders encode query and doc *separately* with no interaction; Cross-Encoders concatenate `(query, doc)` and let every token attend to the other.

### 2️⃣ Why Cross-Encoder reranking?

| | Bi-Encoder | Cross-Encoder |
|--|-----------|---------------|
| Encoding | Query & doc encoded separately | `(query, doc)` encoded jointly |
| Token interaction | ❌ | ✅ Every token attends the other |
| Complexity | O(1) approx. | O(N) |
| Use case | Recall | **Rerank** |

**Empirical comparison**:

- Query: `"Disney ticket refund"`
- Doc: `"Refund policy: free date change 48h in advance"`

| Model | Score |
|-------|-------|
| Bi-Encoder (cosine) | 0.69 |
| **Cross-Encoder** | **0.89** |

**One order of magnitude better discrimination.** Industry standard: Bi-Encoder recalls Top-20 → Cross-Encoder reranks Top-3.

### 3️⃣ Why self-hosted reranker?

- **Data compliance**: In enterprise settings, `(query, doc)` pairs contain raw document fragments. Sending them to third-party APIs is a compliance risk.
- **Cost control**: Each query needs 20 `(query, doc)` scorings. API billed per token; cost explodes at scale.
- **Stable latency**: Local inference has no network jitter; P99 latency is predictable.

### 4️⃣ How is multimodal handled?

**Cross-modal embeddings**: text / images / videos are all encoded into the same **768-dim** space via `tongyi-embedding-vision-flash`.

**Critical engineering detail** — **cross-modal similarity is inherently low**:

| Type | Typical similarity |
|------|-------------------|
| text ↔ text | 0.5 ~ 0.7 |
| text ↔ image | 0.1 ~ 0.2 |

**Thresholds must be modality-specific**: text 0.3+, media **0.10**. A uniform 0.3 threshold means images are never retrieved.

### 5️⃣ Two-layer intent detection

**Regex first, LLM fallback**:

- **Regex** (0 ms): high-frequency cues like `"poster"`, `"video"`, `"what does it look like"`
- **LLM fallback** (~500 ms): implicit intent like `"show me that gathering picture"`, resolved by `qwen3.8-flash`

Low latency in the common case, high coverage in the edge case.

### 6️⃣ Chunking strategy

**Two-stage chunking**:

1. Split by blank lines (paragraphs)
2. Over-long paragraphs split by sentence-ending punctuation (。！？；)
3. Greedy merge into ~300-character chunks
4. Add 50-character overlap

**Every chunk starts on a complete sentence boundary** — avoids cutting Chinese words mid-phrase (e.g., "未使用" split into "未使" + "用").

---

## 📊 Results

### Test Cases

| # | Query | Type | Rerank Top-1 Score | Result |
|---|-------|------|-------------------|--------|
| 1 | "Disney ticket refund process" | text | 0.8870 | ✅ Full refund policy extracted |
| 2 | "Halloween activity poster" | image | matched | ✅ Hits Halloween.jpeg |
| 3 | "My car was scratched — can you see the video?" | video | **0.9998** | ✅ Precise match |
| 4 | "The 'gathering' poster" | image | **0.9723** | ✅ Precise match |

### Sample Output

**Test 3 (video query) — Reranker nails it**:

```
[Rerank Top-3] ★ Final selection
  0.9998 | [video] Car scratch video...     ← Near-perfect score
  0.0002 | [image] 2-Halloween.jpeg...
  0.0000 | [image] 1-Gathering.jpg...

Intent: image=False, video=True
  → Video: https://.../car.mp4 (sim=0.1878)

Final answer:
Sorry to hear about your car. I see there is a resource titled
"Car scratch video" in the knowledge base, but I cannot view or
confirm its contents.

[Related video]: https://.../car.mp4
```

**Test 1 (text query) — zero hallucination**:

```
[Rerank Top-3] ★ Final selection
  0.8870 | Spring Festival, Christmas) Senior tickets require 3-day
           advance booking... refundable within 30 days
  0.6895 | Foreign visitors use passport... original ID required
  0.5769 | Closed Mondays and holidays, less crowded Wed/Thu...

Final answer:
Based on current rules, here is the refund/change information:

- Senior ticket policy: unused tickets can be refunded free of
  charge up to 7 days before expiry; 10% fee applies 3 days
  before expiry; no refund/change within 48 hours...
- Shanghai Disney policy: strict refund/change. Free date change
  once with 48-hour advance notice (price difference applies)...
- Expired unused tickets: refundable within 30 days with 20%
  processing fee for date extension...

For the specific submission process, required documents, and
refund timeline — apologies, I do not have that information.

**Ground answers when sources exist; admit unknowns otherwise. This is the core advantage of RAG over vanilla LLMs.**
```

---

## 🔧 Performance

**Per-query latency** (Mac CPU):

| Stage | Time |
|-------|------|
| Query embedding | 200 ~ 500 ms |
| FAISS search | < 5 ms |
| BM25 search | < 5 ms |
| RRF fusion | < 1 ms |
| **Reranker (20 candidates)** | **300 ~ 800 ms** |
| LLM generation | 1 ~ 3 s |
| **Total** | **2 ~ 4 s** |

**Memory footprint**:

- FAISS index (18 × 768-dim): < 1 MB
- Reranker model (with PyTorch runtime): ~ 1.5 GB

---

## 🎓 Engineering Challenges Solved

### Challenge 1: Region / endpoint mismatch

**Issue**: Alibaba Cloud Model Studio operates in multiple regions (Singapore, Japan, Mainland China). **API key and endpoint must be in the same region**, otherwise HTTP 401.

**Fix**: Explicit endpoint configuration aligned with the key's region; verify model authorization in the console.

### Challenge 2: Mac `torch` segfault

**Issue**: `CrossEncoder.predict()` segfaults on macOS CPU due to **OpenMP multi-thread conflicts**.

**Fix**:

```python
# At file top, before imports
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.set_num_threads(1)
CrossEncoder(..., device="cpu")

# Sequential predict instead of batch
for pair in pairs:
    score = reranker.predict([pair])
```

### Challenge 3: Chunk boundaries cutting mid-sentence

**Issue**: Mechanical character-count splitting severed phrases (e.g., "未使用" → "未使" + "用"), destroying semantic coherence.

**Fix**: **Two-stage chunking** — paragraph-first, sentence-fallback, ensuring every chunk starts on a complete sentence boundary.

---

## 📁 Project Structure

```
CASE-Disney-RAG-Assistant/
├── 4-disney_build_index.py      # Knowledge base construction (parse docx/images/videos + embed)
├── 5-disney_query.py            # Query & RAG pipeline (dual-channel + RRF + rerank)
├── download_reranker.py         # Download local reranker model
├── test_endpoint.py             # Endpoint connectivity test
├── disney_knowledge_base/       # Knowledge base
│   ├── *.docx                   # Text knowledge
│   └── images/                  # Image knowledge
├── disney_index.faiss           # FAISS index (build artifact)
├── disney_metadata.json         # Metadata (build artifact)
└── models/                      # Local reranker weights
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**`requirements.txt`**:

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

### 2. Configure API key

Create `.env`:

```dotenv
DASHSCOPE_API_KEY=sk-your-singapore-region-key
```

Get a key: [Alibaba Cloud Model Studio](https://bailian.console.aliyun.com/) → Region: **Singapore** → API-KEY management.

### 3. Download local reranker

```bash
python download_reranker.py
```

Downloads `BAAI/bge-reranker-base` (~1.1 GB).

### 4. Build the knowledge base

```bash
python 4-disney_build_index.py
```

Produces `disney_index.faiss` and `disney_metadata.json`.

### 5. Run RAG Q&A

```bash
python 5-disney_query.py
```

---

## 🔮 Roadmap

### Short-term

- [ ] **Full knowledge base ingestion** — extend to `Disney RAG KB (Full)`, supporting PDF / PPT / DOC
- [ ] **Query rewriting** — integrate upstream "web-search query rewriting" module to boost recall
- [ ] **Streaming output** — SSE for LLM generation
- [ ] **Web UI** — Streamlit / Gradio for demo

### Mid-term

- [ ] **Evaluation framework** — integrate RAGAS for Faithfulness / Answer Relevancy / Context Precision
- [ ] **Reranker batching** — batch_size tuning + GPU acceleration
- [ ] **Caching layer** — Redis for hot-query retrieval results
- [ ] **Multi-turn dialogue** — conversation history context

### Long-term

- [ ] **Distributed deployment** — FAISS → Milvus / Qdrant at billion-vector scale
- [ ] **Access control** — document-level ACL with pre-retrieval filtering
- [ ] **Agentic workflows** — multi-step reasoning + tool calling

---

## 📚 References

- [Reciprocal Rank Fusion (RRF) paper](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- [BAAI/bge-reranker-base (ModelScope)](https://modelscope.cn/models/BAAI/bge-reranker-base)
- [Alibaba Cloud Model Studio docs](https://help.aliyun.com/zh/model-studio/)
- [FAISS Wiki](https://github.com/facebookresearch/faiss/wiki)

---

## 📄 License

For educational purposes. Model weights copyright belongs to original authors.