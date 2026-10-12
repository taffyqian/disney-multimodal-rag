# test_endpoint.py
from dotenv import load_dotenv
load_dotenv()
import os
from openai import OpenAI

key = os.getenv("DASHSCOPE_API_KEY")
print("key 前缀:", key[:8] if key else None, "| 长度:", len(key) if key else 0)

BASE = "https://ws-75fcsrpt6tqktl06.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
client = OpenAI(api_key=key, base_url=BASE)

# 1. 测 LLM
print("\n--- 测 LLM ---")
resp = client.chat.completions.create(
    model="qwen3.8-max",
    messages=[{"role": "user", "content": "说你好"}],
)
print("✅ LLM:", resp.choices[0].message.content)

# 2. 测 Embedding
print("\n--- 测 Embedding ---")
emb = client.embeddings.create(
    model="tongyi-embedding-vision-flash",
    input=[{"text": "测试文本"}],
)
print("✅ Embedding 维度:", len(emb.data[0].embedding))