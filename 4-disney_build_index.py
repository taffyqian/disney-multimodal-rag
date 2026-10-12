# -*- coding: utf-8 -*-
"""迪士尼RAG助手 V4 - 知识库构建"""
import os
import re
import base64
import json
import numpy as np
import faiss
import dashscope
from pathlib import Path
from dotenv import load_dotenv
from http import HTTPStatus
from docx import Document as DocxDocument

load_dotenv(Path(__file__).resolve().parent / ".env")

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("错误：请设置 DASHSCOPE_API_KEY 环境变量。")

dashscope.api_key = DASHSCOPE_API_KEY
dashscope.base_http_api_url = "https://ws-75fcsrpt6tqktl06.ap-southeast-1.maas.aliyuncs.com/api/v1"

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "disney_knowledge_base"
IMG_DIR = DOCS_DIR / "images"
MULTIMODAL_EMBEDDING_MODEL = "tongyi-embedding-vision-flash"
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

def normalize(vec):
    v = np.asarray(vec, dtype="float32")
    n = np.linalg.norm(v)
    return (v / n if n > 0 else v).tolist()

def parse_docx(file_path):
    doc = DocxDocument(file_path)
    all_text = []
    for element in doc.element.body:
        if element.tag.endswith("p"):
            text = ""
            for run in element.findall(".//w:t",
                {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}):
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
                all_text.append("\\n".join(md))
    return "\\n".join(all_text)

def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    paragraphs = re.split(r'\\n\\s*\\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    sentences = []
    for p in paragraphs:
        if len(p) <= chunk_size:
            sentences.append(p)
        else:
            parts = re.split(r'(?<=[。！？；])', p)
            parts = [s.strip() for s in parts if s.strip()]
            sentences.extend(parts)
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
    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            overlapped.append(chunks[i - 1][-overlap:] + chunks[i])
        chunks = overlapped
    return chunks

def get_text_embedding(text):
    resp = dashscope.MultiModalEmbedding.call(
        model=MULTIMODAL_EMBEDDING_MODEL, input=[{"text": text}])
    if resp.status_code != HTTPStatus.OK:
        raise Exception(f"文本Embedding失败: {resp.code} - {resp.message}")
    return normalize(resp.output["embeddings"][0]["embedding"])

def get_image_embedding(image_path):
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    ext = os.path.splitext(image_path)[1].lower().lstrip(".")
    if ext == "jpg":
        ext = "jpeg"
    image_data = f"data:image/{ext};base64,{b64}"
    resp = dashscope.MultiModalEmbedding.call(
        model=MULTIMODAL_EMBEDDING_MODEL, input=[{"image": image_data}])
    if resp.status_code != HTTPStatus.OK:
        raise Exception(f"图片Embedding失败: {resp.code} - {resp.message}")
    return normalize(resp.output["embeddings"][0]["embedding"])

def get_video_embedding(video_url):
    resp = dashscope.MultiModalEmbedding.call(
        model=MULTIMODAL_EMBEDDING_MODEL, input=[{"video": video_url}])
    if resp.status_code != HTTPStatus.OK:
        raise Exception(f"视频Embedding失败: {resp.code} - {resp.message}")
    vecs = [np.asarray(e["embedding"], dtype="float32") for e in resp.output["embeddings"]]
    vecs = [v / (np.linalg.norm(v) + 1e-12) for v in vecs]
    avg = np.mean(vecs, axis=0)
    avg = avg / (np.linalg.norm(avg) + 1e-12)
    return avg.tolist()

def build_and_save():
    print("\\n--- 构建多模态知识库 V4 ---")
    print(f"Endpoint: {dashscope.base_http_api_url}")
    print(f"Embedding: {MULTIMODAL_EMBEDDING_MODEL}")

    metadata_store = []
    all_vectors = []
    doc_id = 0

    if DOCS_DIR.is_dir():
        for filename in os.listdir(DOCS_DIR):
            path = DOCS_DIR / filename
            if filename.startswith(".") or path.is_dir():
                continue
            if filename.endswith(".docx"):
                print(f"  处理文档: {filename}")
                text = parse_docx(str(path))
                chunks = split_text(text)
                print(f"    长度: {len(text)}, chunk 数: {len(chunks)}")
                for chunk in chunks:
                    vec = get_text_embedding(chunk)
                    metadata_store.append({"id": doc_id, "source": filename,
                                           "type": "text", "content": chunk})
                    all_vectors.append(vec)
                    doc_id += 1

    if IMG_DIR.is_dir():
        print("  处理图片...")
        for img in os.listdir(IMG_DIR):
            if img.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
                path = str(IMG_DIR / img)
                print(f"    - {img}")
                vec = get_image_embedding(path)
                metadata_store.append({"id": doc_id, "source": f"图片: {img}",
                                       "type": "image", "path": path,
                                       "content": f"[图片] {img}"})
                all_vectors.append(vec)
                doc_id += 1

    print("  处理视频...")
    for v in VIDEO_KNOWLEDGE:
        print(f"    - {v['description']}")
        vec = get_video_embedding(v["url"])
        metadata_store.append({"id": doc_id, "source": f"视频: {v['description']}",
                               "type": "video", "url": v["url"],
                               "description": v["description"],
                               "content": f"[视频] {v['description']}"})
        all_vectors.append(vec)
        doc_id += 1

    if not all_vectors:
        print("⚠️ 没有可入库的数据")
        return

    dims = {len(v) for v in all_vectors}
    assert len(dims) == 1, f"❌ 向量维度不一致: {dims}"
    dim = dims.pop()
    print(f"\\n向量维度: {dim}（已 L2 归一化）")

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
    print(f"\\n完成! 文本:{tc}, 图片:{ic}, 视频:{vc}")

if __name__ == "__main__":
    build_and_save()
