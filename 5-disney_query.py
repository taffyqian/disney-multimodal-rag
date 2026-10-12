# -*- coding: utf-8 -*-
"""
迪士尼RAG助手 V5.1 - 查询处理（Mac 防 segfault 版）

关键改动（修 Mac torch 段错误）：
- 文件顶部限制 OMP/MKL 线程为 1
- torch.set_num_threads(1)
- Reranker 显式 device="cpu"
- 逐条 predict 而非批量
"""
# ==================================================
# 【关键】线程限制必须放在所有 import 之前
# ==================================================
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 现在再 import 其他
import re
import json
import numpy as np
import faiss
import dashscope
import jieba
from pathlib import Path
from dotenv import load_dotenv
from http import HTTPStatus
from openai import OpenAI
from rank_bm25 import BM25Okapi

# 限制 torch 线程（在 import torch 之后、用之前）
import torch
torch.set_num_threads(1)

from sentence_transformers import CrossEncoder

# ---------- 环境 ----------
load_dotenv(Path(__file__).resolve().parent / ".env")

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("错误：请设置 'DASHSCOPE_API_KEY' 环境变量。")

DASHSCOPE_HTTP_BASE_URL = "https://ws-75fcsrpt6tqktl06.ap-southeast-1.maas.aliyuncs.com/api/v1"
OPENAI_COMPATIBLE_BASE_URL = "https://ws-75fcsrpt6tqktl06.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"

dashscope.api_key = DASHSCOPE_API_KEY
dashscope.base_http_api_url = DASHSCOPE_HTTP_BASE_URL

client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=OPENAI_COMPATIBLE_BASE_URL)

# ---------- 配置 ----------
BASE_DIR = Path(__file__).resolve().parent
MULTIMODAL_EMBEDDING_MODEL = "tongyi-embedding-vision-flash"
LLM_MODEL = "qwen3.8-max"

INDEX_FILE = BASE_DIR / "disney_index.faiss"
METADATA_FILE = BASE_DIR / "disney_metadata.json"

RERANKER_PATH = BASE_DIR / "models" / "models" / "BAAI--bge-reranker-base" / "snapshots" / "master"

# 检索参数
RECALL_TOP_K = 20
FUSION_TOP_K = 20
RERANK_TOP_N = 3
RRF_K = 60

# 媒体匹配阈值
MEDIA_SIM_THRESHOLD = 0.10

# 意图识别正则
IMAGE_PATTERNS = [r"图片", r"海报", r"照片", r"长什么样", r"给我.*图", r"看.*图", r"发.*图"]
VIDEO_PATTERNS = [r"视频", r"录像", r"影片", r"播放", r"看.*视频", r"给我.*视频"]
LLM_INTENT_FALLBACK = True

# Reranker 单例
_RERANKER = None


# ---------- 工具 ----------
def normalize(vec):
    v = np.asarray(vec, dtype="float32")
    n = np.linalg.norm(v)
    return (v / n if n > 0 else v).tolist()


def get_text_embedding(text):
    resp = dashscope.MultiModalEmbedding.call(
        model=MULTIMODAL_EMBEDDING_MODEL,
        input=[{"text": text}],
    )
    if resp.status_code != HTTPStatus.OK:
        raise Exception(f"Embedding失败: {resp.code} - {resp.message}")
    return normalize(resp.output["embeddings"][0]["embedding"])


def load_index():
    index = faiss.read_index(str(INDEX_FILE))
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"已加载索引: {index.ntotal} 条, 维度: {index.d}")
    return index, metadata


def build_bm25(metadata):
    text_metas = [m for m in metadata if m["type"] == "text"]
    corpus = [list(jieba.cut(m["content"])) for m in text_metas]
    return BM25Okapi(corpus), text_metas


def get_reranker():
    """懒加载本地 Cross-Encoder（Mac 防崩）"""
    global _RERANKER
    if _RERANKER is None:
        print(f"加载本地 Reranker: {RERANKER_PATH}")
        if not RERANKER_PATH.exists():
            raise FileNotFoundError(
                f"Reranker 模型不存在: {RERANKER_PATH}\n"
                "请先运行 download_reranker.py 下载模型"
            )
        _RERANKER = CrossEncoder(str(RERANKER_PATH), device="cpu", max_length=512)
        print("Reranker 就绪")
    return _RERANKER


# ---------- 意图识别 ----------
def detect_intent_by_regex(query):
    wi = any(re.search(p, query) for p in IMAGE_PATTERNS)
    wv = any(re.search(p, query) for p in VIDEO_PATTERNS)
    return wi, wv


def detect_intent_by_llm(query):
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{
                "role": "user",
                "content": (
                    "判断下面的用户问题是否需要返回【图片】或【视频】。\n"
                    "只输出 JSON，格式：{\"want_image\": true/false, \"want_video\": true/false}\n\n"
                    f"问题：{query}"
                ),
            }],
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        return bool(data.get("want_image")), bool(data.get("want_video"))
    except Exception:
        return False, False


def detect_media_intent(query):
    wi, wv = detect_intent_by_regex(query)
    if (wi or wv) or not LLM_INTENT_FALLBACK:
        return wi, wv
    return detect_intent_by_llm(query)


# ---------- 通道1：向量检索 ----------
def vector_search(query, index, metadata, top_k):
    qvec = np.asarray([get_text_embedding(query)], dtype="float32")
    sims, idxs = index.search(qvec, min(top_k, index.ntotal))
    results = []
    for idx, sim in zip(idxs[0], sims[0]):
        if idx == -1:
            continue
        results.append({"idx": int(idx), "score": float(sim),
                        "metadata": metadata[idx]})
    return results


# ---------- 通道2：BM25 检索 ----------
def bm25_search(query, bm25, text_metas, top_k):
    tokens = list(jieba.cut(query))
    scores = bm25.get_scores(tokens)
    ranked = sorted(enumerate(scores), key=lambda x: -x[1])[:top_k]
    results = []
    for i, s in ranked:
        m = text_metas[i]
        results.append({"idx": int(m["id"]), "score": float(s), "metadata": m})
    return results


# ---------- RRF 融合 ----------
def rrf_fusion(channel_results, k=RRF_K):
    rrf_scores = {}
    meta_map = {}
    for channel in channel_results:
        for rank, item in enumerate(channel, 1):
            idx = item["idx"]
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k + rank)
            meta_map[idx] = item["metadata"]
    ranked = sorted(rrf_scores.items(), key=lambda x: -x[1])
    return [
        {"idx": int(i), "rrf_score": float(s), "metadata": meta_map[i]}
        for i, s in ranked
    ]


# ---------- Reranker 精排（Mac 防崩：单条循环）----------
def rerank(query, candidates, top_n):
    if not candidates:
        return []
    try:
        reranker = get_reranker()
        scored = []
        for c in candidates:
            pair = (query, c["metadata"]["content"])
            s = reranker.predict([pair])   # 逐条，避免批量并行
            scored.append((c, float(s[0])))
        ranked = sorted(scored, key=lambda x: -x[1])[:top_n]
        return [
            {
                "idx": c["idx"],
                "rerank_score": s,
                "metadata": c["metadata"],
            }
            for c, s in ranked
        ]
    except Exception as e:
        print(f"  [Rerank 异常] {e}，回退 RRF 排序")
        return candidates[:top_n]


# ---------- 完整检索流水线 ----------
def search_pipeline(query, index, metadata, bm25, text_metas):
    print(f"\n[检索流水线]")

    vec_res = vector_search(query, index, metadata, RECALL_TOP_K)
    bm25_res = bm25_search(query, bm25, text_metas, RECALL_TOP_K)
    print(f"  Step1 召回: 向量 {len(vec_res)} | BM25 {len(bm25_res)}")

    fused = rrf_fusion([vec_res, bm25_res])
    print(f"  Step2 RRF: {len(fused)}")

    top_candidates = fused[:FUSION_TOP_K]
    reranked = rerank(query, top_candidates, RERANK_TOP_N)
    print(f"  Step3 Rerank: {len(reranked)}")

    return vec_res, bm25_res, fused, reranked


# ---------- 主流程 ----------
def rag_ask(query, index, metadata, bm25, text_metas, k=3):
    print(f"\n{'='*60}\nQuery: {query}\n{'='*60}")

    vec_res, bm25_res, fused, reranked = search_pipeline(
        query, index, metadata, bm25, text_metas
    )

    print(f"\n[向量 Top-5]")
    for r in vec_res[:5]:
        print(f"  {r['score']:.4f} | {r['metadata']['content'][:50]}...")

    print(f"\n[BM25 Top-5]")
    for r in bm25_res[:5]:
        print(f"  {r['score']:.4f} | {r['metadata']['content'][:50]}...")

    print(f"\n[RRF 融合 Top-5]")
    for r in fused[:5]:
        print(f"  {r['rrf_score']:.6f} | {r['metadata']['content'][:50]}...")

    print(f"\n[Rerank Top-{k}] ★ 最终入选")
    for r in reranked[:k]:
        print(f"  {r['rerank_score']:.4f} | {r['metadata']['content'][:50]}...")

    want_image, want_video = detect_media_intent(query)
    print(f"\n意图: 图片={want_image}, 视频={want_video}")

    matched_image = None
    if want_image:
        imgs = [r for r in vec_res
                if r["metadata"]["type"] == "image"
                and r["score"] >= MEDIA_SIM_THRESHOLD]
        if imgs:
            matched_image = max(imgs, key=lambda x: x["score"])
            print(f"  → 图片: {matched_image['metadata']['path']} (sim={matched_image['score']:.4f})")
        else:
            print(f"  → 用户要图片，无匹配（阈值 {MEDIA_SIM_THRESHOLD}）")

    matched_video = None
    if want_video:
        vids = [r for r in vec_res
                if r["metadata"]["type"] == "video"
                and r["score"] >= MEDIA_SIM_THRESHOLD]
        if vids:
            matched_video = max(vids, key=lambda x: x["score"])
            print(f"  → 视频: {matched_video['metadata']['url']} (sim={matched_video['score']:.4f})")
        else:
            print(f"  → 用户要视频，无匹配（阈值 {MEDIA_SIM_THRESHOLD}）")

    context = "\n\n".join(
        f"[资料{i+1} 来源: {r['metadata']['source']}]\n{r['metadata']['content']}"
        for i, r in enumerate(reranked[:k])
    )

    prompt = f"""你是一个迪士尼客服助手。请严格根据以下资料回答用户问题。
资料中没有的信息，请回答"抱歉，我暂时没有这方面的资料"，不要编造。

[资料]
{context}

[用户问题]
{query}
"""

    print("\n调用 LLM 生成答案...")
    completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "你是一个迪士尼客服助手。"},
            {"role": "user", "content": prompt},
        ],
    )
    answer = completion.choices[0].message.content

    if matched_image:
        answer += f"\n\n[相关图片]: {matched_image['metadata']['path']}"
    if matched_video:
        answer += f"\n\n[相关视频]: {matched_video['metadata']['url']}"

    print(f"\n最终答案:\n{answer}")
    return answer


# ---------- 主入口 ----------
if __name__ == "__main__":
    index, metadata = load_index()
    bm25, text_metas = build_bm25(metadata)
    print(f"BM25 索引: {len(text_metas)} 条文本")

    tests = [
        ("测试1: 文本查询", "我想了解一下迪士尼门票的退款流程"),
        ("测试2: 图片查询 - 万圣节海报", "最近万圣节的活动海报是什么"),
        ("测试3: 视频查询 - 汽车剐蹭", "我的汽车被剐蹭了，你能看到视频么？"),
        ("测试4: 图片查询 - 聚在一起", "聚在一起说奇妙的海报"),
    ]
    for title, q in tests:
        print(f"\n\n{'#'*60}\n{title}\n{'#'*60}")
        rag_ask(q, index, metadata, bm25, text_metas, k=3)