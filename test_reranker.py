# test_reranker.py
from sentence_transformers import CrossEncoder
from pathlib import Path

MODEL_PATH = Path("models/models/BAAI--bge-reranker-base/snapshots/master").resolve()
print("模型路径:", MODEL_PATH)
print("路径存在:", MODEL_PATH.exists())

# 看目录里有什么
if MODEL_PATH.exists():
    for f in sorted(MODEL_PATH.iterdir()):
        size_mb = f.stat().st_size / 1024 / 1024 if f.is_file() else 0
        print(f"  {f.name}  ({size_mb:.1f} MB)")

reranker = CrossEncoder(str(MODEL_PATH))

query = "迪士尼门票退款"
docs = [
    "门票退款政策：提前48小时可免费改期",
    "迪士尼乐园有好多好玩的项目",
    "年卡分为宝石卡、珍珠卡",
]
scores = reranker.predict([(query, d) for d in docs])
print("\n分数:", scores)
print("（期望：第 1 条最高）")