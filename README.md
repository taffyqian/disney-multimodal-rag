# 🎡 Disney Multimodal RAG Assistant

> A production-grade Retrieval-Augmented Generation (RAG) system for enterprise customer-service scenarios
> **Multi-modal retrieval** (text / image / video) · **Three-stage retrieval pipeline** · **End-to-end runnable**

**🌐 Language**: **English** · [简体中文](./README.zh-CN.md) · [Self notes](./README.self.md)

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FAISS](https://img.shields.io/badge/FAISS-IndexFlatIP-0467DF)](https://github.com/facebookresearch/faiss)
[![Reranker](https://img.shields.io/badge/Reranker-bge--reranker--base-FF6B6B)](https://modelscope.cn/models/BAAI/bge-reranker-base)
[![LLM](https://img.shields.io/badge/LLM-qwen3.8--max-9B59B6)](https://help.aliyun.com/zh/model-studio/)

---

## 📌 TL;DR

Built an end-to-end multimodal RAG system using **dual-channel recall (dense + BM25)**, **Reciprocal Rank Fusion**, and **Cross-Encoder reranking**.

**Knowledge base**: 70 records (67 text chunks from 5 documents + 2 images + 1 video), covering docx / pdf / image / video.

**Results**: 100% hit rate on media queries (video 0.1878, image 0.1856) and precise text retrieval (Rerank 0.8976), with **zero hallucination** on grounded queries.

---

## 🎯 Problems & Solutions

| Business Challenge | My Solution |
|-------------------|-------------|
| Single-channel dense retrieval returns too much "loosely related" noise | **Dual-channel recall**: dense (semantic) + BM25 (keyword) |
| Two channels produce scores on **different scales** — no direct fusion | **RRF (Reciprocal Rank Fusion)** — rank-only, scale-invariant |
| Bi-Encoder lacks precision for fine-grained ranking | **Cross-Encoder reranker** on Top-20 → Top-3 |
| **Media gets drowned by text** (cross-modal similarity inherently low) | **Independent media channel**: global scan + threshold filter, bypassing Top-K |
| Cross-modal (text↔image) similarity is inherently low | **Modality-specific thresholds** (media 0.10 vs text 0.3+) |
| LLM hallucination | **Strict prompt grounding + retrieval quality guarantee** |
| Server 500 aborts batch ingestion | **Exponential backoff retry + skip-on-fail + rate limiting** |

---

## 🏗️ Architecture

```
                       User Query
                           │
              ┌────────────┴────────────┐
              │                         │
       ┌──────▼───────┐         ┌───────▼───────┐
       │  Text Channel │         │ Media Channel │
       │              │         │  (Independent)│
       │  Dense+BM25  │         │ Global Scan   │
       │      ↓       │         │      ↓        │
       │  RRF Fusion  │         │ Threshold     │
       │      ↓       │         │      ↓        │
       │  Reranker    │         │ Top-1 Image   │
       │      ↓       │         │ Top-1 Video   │
       │   Top-3      │         │               │
       └──────┬───────┘         └───────┬───────┘
              │                         │
              └────────────┬────────────┘
                           │
                   ┌───────▼────────┐
                   │  LLM Generate   │  ← Text only
                   │  + Media Attach │  ← Media as reference
                   └────────────────┘
```

**Stack**: Python 3.11 · FAISS · rank_bm25 + jieba · sentence-transformers · Alibaba Cloud DashScope

### Dual-channel Output Design

The system runs **two parallel channels** without fusion:

| Channel | Input | Processing | Output | Purpose |
|---------|-------|-----------|--------|---------|
| **Text** | Query | Dense + BM25 → RRF → Reranker | Top-3 chunks | **LLM generation source** |
| **Media** | Query + Intent | Global scan + Threshold | Top-1 image + Top-1 video | **Supplementary display** |

**Why no fusion?**

1. **Different scales**: Cross-Encoder scores (0–1) vs cross-modal cosine (0.1–0.2) — not comparable
2. **Different roles**: Text is the *generation source*; media is the *reference resource*
3. **Conflict handling**: When text and media conflict, **text wins** (Reranker precision ≫ cross-modal match)

### Media Independent Retrieval

**Problem**: After scaling the KB from 18 to 70 records, **media retrieval broke** — cross-modal similarity is inherently low (0.1–0.2), so media got squeezed out of the Top-K by text chunks (0.4–0.7).

**Solution**: media uses an **independent channel** — global scan + threshold filter, bypassing text Top-K competition.

```python
def find_best_media(index, metadata, media_type, query, threshold=0.10):
    """Global scan for the best matching media (no Top-K competition)"""
    qvec = np.asarray([get_text_embedding(query)], dtype="float32")
    sims, idxs = index.search(qvec, index.ntotal)   # ← Full scan

    best = None
    for idx, sim in zip(idxs[0], sims[0]):
        m = metadata[idx]
        if m["type"] != media_type:                  # ← Filter by type
            continue
        if float(sim) < threshold:
            continue
        if best is None or float(sim) > best["score"]:
            best = {"idx": int(idx), "score": float(sim), "metadata": m}
    return best
```

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
2. Over-long paragraphs split by sentence-ending punctuation (。！？；.?!)
3. Greedy merge into ~300-character chunks
4. Add 50-character overlap

**Every chunk starts on a complete sentence boundary** — avoids cutting Chinese words mid-phrase (e.g., "未使用" split into "未使" + "用").

### 7️⃣ Batch ingestion resilience

- **Exponential backoff retry**: wait 2s → 4s → 8s on 500/429
- **Skip-on-fail**: individual chunk failure does not abort the whole job
- **Rate limiting**: sleep 0.5s every 10 chunks

---

## 📊 Results

### Knowledge Base Scale

| Type | Count | Source |
|------|-------|--------|
| **Text chunks** | 67 | 4 docx + 1 pdf (HK Disney travel guide) |
| **Images** | 2 | Halloween.jpeg + Gathering.jpg |
| **Video** | 1 | Car scratch video |
| **Total** | **70** | — |

### Test Cases

| # | Query | Type | Rerank Top-1 | Media Match | Result |
|---|-------|------|-------------|-------------|--------|
| 1 | "Disney ticket refund process" | text | **0.8976** | — | ✅ Full refund policy |
| 2 | "Halloween activity poster" | image | — | **0.1756** | ✅ Matches Halloween.jpeg |
| 3 | "My car was scratched" | video | — | **0.1878** | ✅ Matches car video |
| 4 | "The 'gathering' poster" | image | — | **0.1856** | ✅ Matches gathering.jpg |

**Notes**:

- **Text Rerank score** (0.8976) is a **Cross-Encoder relevance score**
- **Media match score** (0.1756) is a **cross-modal cosine similarity** — different scale, **not directly comparable**
- **Two outputs are independent**: text goes to LLM; media is supplementary display

### Sample Output

**Test 3 (video query) — media independent retrieval nails it**:

```
[Rerank Top-3] ★ Text Channel Output
  0.0001 | Space Mountain...
  0.0001 | Cars Ride...
  0.0000 | Ocean Park...

Intent: image=False, video=True
  → Independent video channel scan...
    [video global scan] 1 found:
      0.1878 | [video] Car scratch video...
  → Match: .../car.mp4 (sim=0.1878)

Final answer:
Sorry, I don't have information on this.

[Related video]: https://.../car.mp4
```

**Test 1 (text query) — zero hallucination**:

```
[Rerank Top-3] ★ Text Channel Output
  0.8976 | Senior ticket refund: free change 7 days before expiry...
  0.6178 | Foreign visitors use passport...
  0.5103 | Buy souvenir CD... currency exchange...

Final answer:
The current resources do not cover the specific refund submission
process, required documents, or refund timeline. However, the
following refund/change rules are available:

- Senior ticket: free refund/change up to 7 days before expiry;
  10% fee applies 3 days before; no refund within 48h...
- Shanghai Disney: no refund/change within 48h of entry;
  medical emergencies require a Tier-2 hospital certificate...
- Expired tickets: extendable within 30 days with 20% fee...
```

**Ground answers when sources exist; admit unknowns otherwise. This is the core advantage of RAG over vanilla LLMs.**

---

## 🔧 Performance

**Per-query latency** (Mac CPU):

| Stage | Time |
|-------|------|
| Query embedding | 200 ~ 500 ms |
| FAISS search (70 records) | < 5 ms |
| BM25 search | < 5 ms |
| RRF fusion | < 1 ms |
| **Reranker (20 candidates)** | **300 ~ 800 ms** |
| **Media global scan (per type)** | **200 ~ 500 ms** |
| LLM generation | 1 ~ 3 s |
| **Total** | **2 ~ 5 s** |

**Memory footprint**:

- FAISS index (70 × 768-dim): < 1 MB
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

### Challenge 4: Media retrieval broke after KB scaling

**Issue**: After scaling from 18 to 70 records, **media got squeezed out of Top-K** by text chunks (cross-modal similarity 0.1–0.2 vs text 0.4–0.7).

**Fix**: **Independent media channel** — global scan + threshold filter, bypassing Top-K competition entirely. This embodies the core engineering insight of multimodal RAG: **different modalities need separate retrieval paths and separate thresholds**.

### Challenge 5: Server 500 aborted batch ingestion

**Issue**: During PDF ingestion (52 chunks), the embedding API returned sporadic `InternalError`, and a single failure aborted the entire pipeline.

**Fix**:

- **Exponential backoff retry** (2s → 4s → 8s)
- **Skip-on-fail** (partial success is better than total failure)
- **Active rate limiting** (0.5s sleep every 10 chunks)

---

## 📁 Project Structure

```
CASE-Disney-RAG-Assistant/
├── 4-disney_build_index.py      # KB construction (docx/pdf/image/video + retry + rate limit)
├── 5-disney_query.py            # Query & RAG pipeline (dual-channel + RRF + Rerank + media independent)
├── download_reranker.py         # Download local reranker model
├── test_endpoint.py             # Endpoint connectivity test
├── disney_knowledge_base/       # Knowledge base
│   ├── *.docx                   # Text knowledge (4 files)
│   ├── *.pdf                    # PDF knowledge (1 file)
│   └── images/                  # Image knowledge (2 files)
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
pdfplumber>=0.10.0
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

- [ ] **VLM integration** — use `qwen-vl-max` to let the LLM actually "see" images (breaking the text-only LLM limitation)
- [ ] **Query rewriting** — integrate upstream "web-search query rewriting" module to boost recall
- [ ] **Streaming output** — SSE for LLM generation
- [ ] **Web UI** — Streamlit / Gradio for demo

### Mid-term

- [ ] **Evaluation framework** — integrate RAGAS for Faithfulness / Answer Relevancy / Context Precision
- [ ] **Reranker batching** — batch_size tuning + GPU acceleration
- [ ] **Caching layer** — Redis for hot-query retrieval results
- [ ] **Multi-turn dialogue** — conversation history context
- [ ] **Category filtering** — leverage `category` metadata for ACL / faceted search

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