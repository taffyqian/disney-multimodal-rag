# -*- coding: utf-8 -*-
"""
迪士尼RAG助手 V5.1 - 知识库构建（docx + pdf，带重试 + 跳过 + 限流）

改动：
- 新增 PDF 解析（pdfplumber）
- Embedding 调用带指数退避重试（应对服务端 500 / 限流）
- 单个 chunk 失败不中断整个建库流程
- 每 10 个 chunk 主动限流（避免触发 QPS）
- 保留两阶段分块 / L2 归一化 / IndexFlatIP
"""
import os
import re
import time
import base64
import json
import numpy as np
import faiss
import dashscope
import pdfplumber
from pathlib import Path
from dotenv import load_dotenv
from http import HTTPStatus
from docx import Document as DocxDocument

# ---------- 环境 ----------
load_dotenv(Path(__file__).resolve().parent / ".env")

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("错误：请设置 'DASHSCOPE_API_KEY' 环境变量。")

dashscope.api_key = DASHSCOPE_API_KEY
dashscope.base_http_api_url = "https://ws-75fcsrpt6tqktl06.ap-southeast-1.maas.aliyuncs.com/api/v1"

# ---------- 配置 ----------
BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "disney_knowledge_base"
IMG_DIR = DOCS_DIR / "images"

MULTIMODAL_EMBEDDING_MODEL = "tongyi-embedding-vision-flash"  # 768 维

INDEX_FILE = BASE_DIR / "disney_index.faiss"
METADATA_FILE = BASE_DIR / "disney_metadata.json"

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

VIDEO_KNOWLEDGE = [
    {
        "url": "https://dataset-1255932437.cos.ap-nanjing.myqcloud.com/mp4/car.mp4",
        "description": "汽车剐蹭视频",
    }
]


# ---------- 归一化 ----------
def normalize(vec):
    v = np.asarray(vec, dtype="float32")
    n = np.linalg.norm(v)
    return (v / n if n > 0 else v).tolist()


# ---------- 文档解析 ----------
def parse_docx(file_path):
    """解析 DOCX：段落 + 表格（转 Markdown）"""
    doc = DocxDocument(file_path)
    all_text = []
    for element in doc.element.body:
        if element.tag.endswith("p"):
            text = ""
            for run in element.findall(
                ".//w:t",
                {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"},
            ):
                text += run.text or ""
            if text.strip():
                all_text.append(text.strip())
        elif element.tag.endswith("tbl"):
            table = [t for t in doc.tables if t._element is element][0]
            if table.rows:
                md = []
                header = [c.text.strip() for c in table.rows[0].cells]
                md.append("| " + " | ".join(header) + " |")
                md.append("|" + "---|" * len(header))
                for row in table.rows[1:]:
                    md.append("| " + " | ".join(c.text.strip() for c in row.cells) + " |")
                all_text.append("\n".join(md))
    return "\n".join(all_text)


def parse_pdf(file_path):
    """解析 PDF：逐页提取文本"""
    texts = []
    with pdfplumber.open(str(file_path)) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            t = page.extract_text() or ""
            if t.strip():
                texts.append(f"[Page {i}]\n{t.strip()}")
    return "\n\n".join(texts)


# ---------- 两阶段分块 ----------
def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    两阶段分块：
    1) 先按空行切段落
    2) 段落太长再按句末标点（。！？；.?!）切句
    3) 贪心合并成 ~chunk_size 的 chunk
    4) 加 overlap 上下文
    """
    # 阶段 1：按空行分段
    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    # 阶段 2：超长段落按句号切
    sentences = []
    for p in paragraphs:
        if len(p) <= chunk_size:
            sentences.append(p)
        else:
            parts = re.split(r'(?<=[。！？；.?!])', p)
            parts = [s.strip() for s in parts if s.strip()]
            sentences.extend(parts)

    # 阶段 3：贪心合并
    chunks = []
    buf = ""
    for s in sentences:
        if len(buf) + len(s) <= chunk_size:
            buf += s
        else:
            if buf:
                chunks.append(buf)
            if len(s) > chunk_size:
                for i in range(0, len(s), chunk_size):
                    chunks.append(s[i:i + chunk_size])
                buf = ""
            else:
                buf = s
    if buf:
        chunks.append(buf)

    # 阶段 4：加 overlap
    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            overlapped.append(chunks[i - 1][-overlap:] + chunks[i])
        chunks = overlapped

    return chunks


# ---------- Embedding（带重试）----------
def get_text_embedding(text, max_retries=3):
    """带重试的文本 Embedding（应对服务端 500 / 限流）"""
    # 清理：去掉控制字符、多余空白
    clean = re.sub(r'[\x00-\x1f\x7f]', ' ', text).strip()
    if not clean:
        raise ValueError("空文本，跳过")

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = dashscope.MultiModalEmbedding.call(
                model=MULTIMODAL_EMBEDDING_MODEL,
                input=[{"text": clean}],
            )
            if resp.status_code == HTTPStatus.OK:
                return normalize(resp.output["embeddings"][0]["embedding"])

            # 服务端错误 / 限流 → 重试
            if resp.status_code in (429, 500, 503):
                wait = 2 ** attempt  # 2s, 4s, 8s
                print(f"        [重试 {attempt}/{max_retries}] {resp.code}，等待 {wait}s...")
                time.sleep(wait)
                last_err = f"{resp.code} - {resp.message}"
                continue

            # 其他错误 → 直接抛
            raise Exception(f"文本Embedding失败: {resp.code} - {resp.message}")

        except Exception as e:
            last_err = str(e)
            if attempt == max_retries:
                break
            wait = 2 ** attempt
            print(f"        [重试 {attempt}/{max_retries}] {str(e)[:60]}，等待 {wait}s...")
            time.sleep(wait)

    raise Exception(f"文本Embedding失败：重试 {max_retries} 次后仍失败（{last_err}）")


def get_image_embedding(image_path):
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    ext = os.path.splitext(image_path)[1].lower().lstrip(".")
    if ext == "jpg":
        ext = "jpeg"
    image_data = f"data:image/{ext};base64,{b64}"
    resp = dashscope.MultiModalEmbedding.call(
        model=MULTIMODAL_EMBEDDING_MODEL,
        input=[{"image": image_data}],
    )
    if resp.status_code != HTTPStatus.OK:
        raise Exception(f"图片Embedding失败: {resp.code} - {resp.message}")
    return normalize(resp.output["embeddings"][0]["embedding"])


def get_video_embedding(video_url):
    resp = dashscope.MultiModalEmbedding.call(
        model=MULTIMODAL_EMBEDDING_MODEL,
        input=[{"video": video_url}],
    )
    if resp.status_code != HTTPStatus.OK:
        raise Exception(f"视频Embedding失败: {resp.code} - {resp.message}")
    vecs = [np.asarray(e["embedding"], dtype="float32") for e in resp.output["embeddings"]]
    vecs = [v / (np.linalg.norm(v) + 1e-12) for v in vecs]
    avg = np.mean(vecs, axis=0)
    avg = avg / (np.linalg.norm(avg) + 1e-12)
    return avg.tolist()


# ---------- 主流程 ----------
def build_and_save():
    print("\n--- 构建多模态知识库 V5.1 ---")
    print(f"Endpoint: {dashscope.base_http_api_url}")
    print(f"Embedding: {MULTIMODAL_EMBEDDING_MODEL}")
    print(f"分块: chunk={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}（按段落+句号切）")

    metadata_store = []
    all_vectors = []
    doc_id = 0
    total_chunks = 0
    total_success = 0

    # 1) 文档（docx + pdf）
    if DOCS_DIR.is_dir():
        for filename in os.listdir(DOCS_DIR):
            path = DOCS_DIR / filename
            if filename.startswith(".") or path.is_dir():
                continue

            # 按扩展名选择解析器
            if filename.endswith(".docx"):
                print(f"\n  处理文档: {filename}")
                full_text = parse_docx(str(path))
            elif filename.endswith(".pdf"):
                print(f"\n  处理文档: {filename} (PDF)")
                full_text = parse_pdf(str(path))
            else:
                continue

            chunks = split_text(full_text)
            total_chunks += len(chunks)
            print(f"    文档长度: {len(full_text)} 字符, 切分为 {len(chunks)} 个chunk")

            success = 0
            for i, chunk in enumerate(chunks, 1):
                try:
                    vec = get_text_embedding(chunk)
                    metadata_store.append({
                        "id": doc_id,
                        "source": filename,
                        "type": "text",
                        "content": chunk,
                    })
                    all_vectors.append(vec)
                    doc_id += 1
                    success += 1
                    total_success += 1
                    if i % 5 == 0 or i == len(chunks):
                        print(f"      [{i}/{len(chunks)}] ✓ (成功 {success})")
                except Exception as e:
                    print(f"      [{i}/{len(chunks)}] ✗ 跳过: {str(e)[:80]}")
                    continue

                # 限流：每 10 个 chunk 睡 0.5 秒
                if i % 10 == 0:
                    time.sleep(0.5)

            print(f"    → 成功入库 {success}/{len(chunks)} chunks")

    # 2) 图片
    if IMG_DIR.is_dir():
        print("\n  处理图片...")
        for img in os.listdir(IMG_DIR):
            if img.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
                path = str(IMG_DIR / img)
                print(f"    - {img}")
                try:
                    vec = get_image_embedding(path)
                    metadata_store.append({
                        "id": doc_id,
                        "source": f"图片: {img}",
                        "type": "image",
                        "path": path,
                        "content": f"[图片] {img}",
                    })
                    all_vectors.append(vec)
                    doc_id += 1
                    total_success += 1
                except Exception as e:
                    print(f"      ✗ 跳过: {str(e)[:80]}")

    # 3) 视频
    print("\n  处理视频...")
    for v in VIDEO_KNOWLEDGE:
        print(f"    - {v['description']}")
        try:
            vec = get_video_embedding(v["url"])
            metadata_store.append({
                "id": doc_id,
                "source": f"视频: {v['description']}",
                "type": "video",
                "url": v["url"],
                "description": v["description"],
                "content": f"[视频] {v['description']}",
            })
            all_vectors.append(vec)
            doc_id += 1
            total_success += 1
        except Exception as e:
            print(f"      ✗ 跳过: {str(e)[:80]}")

    if not all_vectors:
        print("⚠️ 没有可入库的数据")
        return

    # 维度一致性
    dims = {len(v) for v in all_vectors}
    assert len(dims) == 1, f"❌ 向量维度不一致: {dims}"
    dim = dims.pop()
    print(f"\n向量维度: {dim}（已 L2 归一化）")

    index = faiss.IndexFlatIP(dim)
    index.add(np.asarray(all_vectors, dtype="float32"))
    faiss.write_index(index, str(INDEX_FILE))
    print(f"✅ 索引已保存: {INDEX_FILE}")

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata_store, f, ensure_ascii=False, indent=2)
    print(f"✅ 元数据已保存: {METADATA_FILE}")

    tc = sum(1 for m in metadata_store if m["type"] == "text")
    ic = sum(1 for m in metadata_store if m["type"] == "image")
    vc = sum(1 for m in metadata_store if m["type"] == "video")
    print(f"\n===== 完成 =====")
    print(f"文本: {tc} | 图片: {ic} | 视频: {vc} | 总计: {len(metadata_store)}")
    print(f"处理 chunks: {total_success}/{total_chunks} 成功")


if __name__ == "__main__":
    build_and_save()