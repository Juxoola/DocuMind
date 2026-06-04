"""Анализ F6 + что LLM получает на вход."""
import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, r"C:\test")

import config
from src.rag_pipeline import retrieve_nodes, build_file_context, get_vector_store

NOTEBOOK = "e104f1c8"
QUERY = "Что происходит с приоритетом потока из пула при его возврате обратно в пул?"

store = get_vector_store(NOTEBOOK)
all_metas = store._collection.get(include=["metadatas"])
file_names = sorted(set(m.get("file_name", "?") for m in all_metas.get("metadatas", [])))

print(f"RAG_TOP_K_PER_FILE = {config.RAG_TOP_K_PER_FILE}")
print(f"RAG_RERANK_POOL    = {config.RAG_RERANK_POOL}")
print(f"RAG_FINAL_TOP_N    = {config.RAG_FINAL_TOP_N}")
print(f"MIN_FINAL_CHUNKS   = {config.MIN_FINAL_CHUNKS}")
print(f"RERANK_SCORE_THRESHOLD = {config.RERANK_SCORE_THRESHOLD}")
print()

nodes = retrieve_nodes(QUERY, NOTEBOOK, file_names)
print(f"After retrieve_nodes: {len(nodes)} nodes")
scores = [n.score for n in nodes]
print(f"Scores: max={max(scores):.4f}, min={min(scores):.4f}, median={sorted(scores)[len(scores)//2]:.4f}")

# F6 manual calc
import statistics
median = statistics.median(scores)
mad = statistics.median([abs(s - median) for s in scores]) or 0.05
adaptive_thr = max(0.0, median - 2.0 * mad)
print(f"F6: median={median:.4f}, MAD={mad:.4f}, adaptive_thr={adaptive_thr:.4f}")
print(f"F6 cuts: {sum(1 for s in scores if s < adaptive_thr)} chunks below threshold")

# vs static 0.05
static_05 = config.RERANK_SCORE_THRESHOLD
print(f"Static 0.05: would keep {sum(1 for s in scores if s >= static_05)} chunks")

print("\n=== CONTEXT THAT GOES TO LLM ===")
sources, context = build_file_context(nodes, NOTEBOOK)
print(f"Context chars: {len(context)}")
print("---")
# Покажем первые 3000 chars
print(context[:3000])
print("---")
# Ищем упоминание приоритета
if "приоритет" in context.lower():
    pos = context.lower().find("приоритет")
    print(f"\n>>> НАЙДЕНО 'приоритет' в контексте на позиции {pos}:")
    print(context[max(0,pos-200):pos+500])
else:
    print("\n>>> СЛОВО 'приоритет' НЕ НАЙДЕНО в контексте!")
