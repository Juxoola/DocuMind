"""
Полный E2E тест: RAG (как в /api/chat) + LLM (qwen3.5-4b на 58061).
"""
import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, r"C:\test")

import json, requests
import config
from src.rag_pipeline import retrieve_nodes, build_file_context, make_prompt

NOTEBOOK = "e104f1c8"
QUERY = """Что происходит с приоритетом потока из пула при его возврате обратно в пул?
Варианты:
A. Приоритет остаётся изменённым.
B. Приоритет устанавливается в Highest.
C. Восстанавливается приоритет Normal.
D. Приоритет устанавливается в Lowest.

Ответь одной буквой (A, B, C или D) с коротким обоснованием на основе контекста."""

LLM_URL = "http://127.0.0.1:58061/v1/chat/completions"
MODEL = "qwen3.5-4b-claude-4.6-os-auto-variable-heretic-uncensored-thinking.i1-iq4_xs.gguf"

# 1. RAG
print("=== STEP 1: RAG ===")
store = config.get_vector_store_func if hasattr(config, 'get_vector_store_func') else None
from src.rag_pipeline import get_vector_store
vstore = get_vector_store(NOTEBOOK)
all_metas = vstore._collection.get(include=["metadatas"])
file_names = sorted(set(m.get("file_name", "?") for m in all_metas.get("metadatas", [])))

nodes = retrieve_nodes(QUERY, NOTEBOOK, file_names)
print(f"RAG found {len(nodes)} nodes")
sources, context = build_file_context(nodes, NOTEBOOK)
print(f"Context: {len(context)} chars, {len(sources)} sources\n")

# 2. LLM call
print("=== STEP 2: LLM CALL ===")
prompt = make_prompt(QUERY, context, thinking_mode=False, max_tokens=2048)
print(f"Prompt: {len(prompt)} chars\n")

try:
    r = requests.post(LLM_URL, json={
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Ты — эксперт по .NET threading. Отвечай на основе КОНТЕКСТА, не выдумывай. Если ответа в контексте нет — скажи 'НЕ НАЙДЕНО'."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 1024,
        "stream": False,
    }, timeout=120)
    r.raise_for_status()
    data = r.json()
    answer = data["choices"][0]["message"]["content"]
    print("=== LLM ANSWER ===")
    print(answer)
except Exception as e:
    print(f"ERROR: {e}")
