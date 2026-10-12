# download_reranker.py
from modelscope import snapshot_download
from pathlib import Path

cache_dir = Path(__file__).resolve().parent / "models"
cache_dir.mkdir(exist_ok=True)

model_dir = snapshot_download(
    "BAAI/bge-reranker-base",
    cache_dir=str(cache_dir),
)
print("模型已下载到:", model_dir)