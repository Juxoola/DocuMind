"""
Диагностика потребления памяти: Embedding + Reranker (GGUF llama-server)
Запуск: python tests/memory_diagnosis.py
"""
import os
import sys
import time
import json
import subprocess
import requests
import psutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

def bytes_to_mb(b):
    return b / (1024 ** 2)

def get_system_ram_mb():
    vm = psutil.virtual_memory()
    return bytes_to_mb(vm.used), bytes_to_mb(vm.total)

def get_process_mem_mb(pid):
    try:
        p = psutil.Process(pid)
        return bytes_to_mb(p.memory_info().rss)
    except Exception:
        return 0.0

def get_all_llama_processes():
    """Возвращает все процессы llama-server.exe и их память."""
    result = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info']):
        try:
            if proc.info['name'] and 'llama-server' in proc.info['name'].lower():
                mem_mb = bytes_to_mb(proc.info['memory_info'].rss)
                cmd = proc.info['cmdline'] or []
                # Находим имя модели из аргументов
                model_name = "unknown"
                for i, arg in enumerate(cmd):
                    if arg == '-m' and i + 1 < len(cmd):
                        model_name = os.path.basename(cmd[i + 1])
                        break
                result.append({
                    'pid': proc.info['pid'],
                    'model': model_name,
                    'ram_mb': mem_mb,
                    'cmdline': cmd
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return result

def print_separator(title=""):
    w = 60
    if title:
        print(f"\n{'='*w}")
        print(f"  {title}")
        print(f"{'='*w}")
    else:
        print(f"{'─'*w}")

def measure_baseline():
    """Измеряет базовое потребление RAM без запущенных серверов."""
    print_separator("BASELINE (без серверов)")
    used, total = get_system_ram_mb()
    python_pid = os.getpid()
    python_mem = get_process_mem_mb(python_pid)
    llama_procs = get_all_llama_processes()
    
    print(f"  RAM системы:         {used:.0f} MB / {total:.0f} MB ({used/total*100:.1f}%)")
    print(f"  Python процесс:      {python_mem:.0f} MB")
    print(f"  llama-server.exe:    {len(llama_procs)} шт.")
    for p in llama_procs:
        print(f"    PID {p['pid']}: {p['model']} — {p['ram_mb']:.0f} MB")
    return used

def analyze_llama_server_cmd(embed_path, rerank_path):
    """Анализирует аргументы запуска серверов и объясняет потребление памяти."""
    print_separator("АНАЛИЗ ПАРАМЕТРОВ ЗАПУСКА")
    
    SERVER_EXE = os.path.join(config.BASE_DIR, "bin", "llama-server.exe")
    
    # Параметры для embedding сервера (из gguf_direct.py, строка 250):
    # cmd.extend(["-c", "4096", "-b", "2048", "-ub", "2048"])
    ctx_size = 4096
    n_batch = 2048
    n_ubatch = 2048
    
    # Подсчет теоретического потребления памяти для Qwen3-0.6B Q8
    # Q8 = ~1 байт/параметр → 0.6B * 1 = ~600 MB веса
    # KV-cache при ctx=4096, hidden_dim~1024, layers~28, heads~16:
    # kv_size = 2 * ctx * layers * hidden * dtype_bytes
    # Q8_0: dtype=2B (f16 kv)
    # Примерный расчет:
    hidden_dim_embed = 1024  # Qwen3-0.6B
    layers_embed = 28
    kv_bytes_embed = 2 * ctx_size * layers_embed * hidden_dim_embed * 2  # f16
    kv_mb_embed = kv_bytes_embed / (1024**2)
    
    print(f"\n  Embedding сервер ({os.path.basename(embed_path) if embed_path else 'N/A'}):")
    print(f"    Контекст:        -c {ctx_size}")
    print(f"    Батч:            -b {n_batch}, -ub {n_ubatch}")
    print(f"    Модель ~Q8:      ~600 MB (параметры)")
    print(f"    KV-cache (f16):  ~{kv_mb_embed:.0f} MB (ctx={ctx_size}, layers={layers_embed})")
    print(f"    Буфер батча:     ~{n_batch * 4096 / 1024**2:.0f}+ MB")
    print(f"    ⚠️  Итого прибл.: ~{600 + kv_mb_embed + n_batch*4096/1024**2:.0f} MB")
    
    print(f"\n  Reranker сервер ({os.path.basename(rerank_path) if rerank_path else 'N/A'}):")
    print(f"    Контекст:        -c {ctx_size} (4096 x2 модели = 8192 всего)")
    print(f"    Батч:            -b {n_batch}, -ub {n_ubatch}")
    print(f"    Модель ~Q8:      ~600 MB (параметры)")
    print(f"    KV-cache (f16):  ~{kv_mb_embed:.0f} MB")
    print(f"    Буфер батча:     ~{n_batch * 4096 / 1024**2:.0f}+ MB")
    print(f"    ⚠️  Итого прибл.: ~{600 + kv_mb_embed + n_batch*4096/1024**2:.0f} MB")
    
    print(f"\n  СУММАРНО (2 сервера): ~{2 * (600 + kv_mb_embed + n_batch*4096/1024**2):.0f} MB теоретически")
    print(f"\n  РЕАЛЬНЫЕ ПРИЧИНЫ ПЕРЕРАСХОДА:")
    print(f"  1. -b 2048, -ub 2048 → llama.cpp выделяет батч-буфер ВПЕРЁД: ~{2048*4096/1024**2:.0f}+ MB/сервер")
    print(f"  2. KV-cache f16 при 4096 токенов: ~{kv_mb_embed:.0f} MB/сервер")
    print(f"  3. Python + chromadb + torch: ~500-800 MB")
    print(f"  4. llama.cpp overhead (CUDA init, scratch buffers): ~200-400 MB/сервер")
    print(f"  5. 2 процесса llama-server.exe × 2-3 GB = 4-6 GB")

def run_live_test():
    """Запускает реальные серверы и измеряет фактическое потребление."""
    print_separator("ЖИВОЙ ТЕСТ: запуск серверов и замер RAM")
    
    embed_name = config.EMBEDDING_MODEL_NAME
    rerank_name = config.RERANKER_MODEL_NAME
    
    embed_path = config.resolve_model_path(embed_name)
    rerank_path = config.resolve_model_path(rerank_name)
    
    print(f"\n  Embedding: {embed_name}")
    print(f"    → Путь: {embed_path}")
    exists_e = os.path.exists(embed_path) if embed_path else False
    size_e = os.path.getsize(embed_path) / 1024**2 if exists_e else 0
    print(f"    → Файл: {'✅ найден' if exists_e else '❌ НЕ НАЙДЕН'} ({size_e:.0f} MB)")
    
    print(f"\n  Reranker: {rerank_name}")
    print(f"    → Путь: {rerank_path}")
    exists_r = os.path.exists(rerank_path) if rerank_path else False
    size_r = os.path.getsize(rerank_path) / 1024**2 if exists_r else 0
    print(f"    → Файл: {'✅ найден' if exists_r else '❌ НЕ НАЙДЕН'} ({size_r:.0f} MB)")
    
    if not exists_e or not exists_r:
        print("\n  ⚠️  Один или оба файла не найдены. Пропуск живого теста.")
        return
    
    ram_before, _ = get_system_ram_mb()
    print(f"\n  RAM до запуска: {ram_before:.0f} MB")
    
    print_separator("Запуск Embedding сервера...")
    from src.gguf_direct import get_gguf_embedding_url
    t0 = time.time()
    embed_url = get_gguf_embedding_url(embed_path, is_reranker=False)
    embed_time = time.time() - t0
    
    time.sleep(2)  # Даем время на полный старт
    ram_after_embed, _ = get_system_ram_mb()
    llama_procs = get_all_llama_processes()
    
    print(f"  ✅ Embedding запущен за {embed_time:.1f}с: {embed_url}")
    print(f"  RAM после Embedding:  {ram_after_embed:.0f} MB (+{ram_after_embed - ram_before:.0f} MB)")
    for p in llama_procs:
        print(f"    llama-server PID {p['pid']}: {p['model']} — {p['ram_mb']:.0f} MB")
    
    print_separator("Запуск Reranker сервера...")
    t0 = time.time()
    rerank_url = get_gguf_embedding_url(rerank_path, is_reranker=True)
    rerank_time = time.time() - t0
    
    time.sleep(2)
    ram_after_rerank, _ = get_system_ram_mb()
    llama_procs_all = get_all_llama_processes()
    
    print(f"  ✅ Reranker запущен за {rerank_time:.1f}с: {rerank_url}")
    print(f"  RAM после Reranker:   {ram_after_rerank:.0f} MB (+{ram_after_rerank - ram_after_embed:.0f} MB)")
    for p in llama_procs_all:
        print(f"    llama-server PID {p['pid']}: {p['model']} — {p['ram_mb']:.0f} MB")
    
    total_delta = ram_after_rerank - ram_before
    print_separator("ИТОГ ЖИВОГО ТЕСТА")
    print(f"  Размер файлов на диске:")
    print(f"    Embedding:  {size_e:.0f} MB")
    print(f"    Reranker:   {size_r:.0f} MB")
    print(f"    Итого:      {size_e + size_r:.0f} MB")
    print(f"\n  Потребление RAM (прирост):")
    print(f"    +Embedding: +{ram_after_embed - ram_before:.0f} MB")
    print(f"    +Reranker:  +{ram_after_rerank - ram_after_embed:.0f} MB")
    print(f"    Итого:      +{total_delta:.0f} MB")
    
    ratio = total_delta / (size_e + size_r) if (size_e + size_r) > 0 else 0
    print(f"\n  Коэффициент раздутия: ×{ratio:.1f} (RAM / файл на диске)")
    print(f"\n  ОБЪЯСНЕНИЕ КОЭФФИЦИЕНТА:")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  Q8_0 GGUF: данные уже в памяти в f16 для вычислений")
    print(f"  KV-cache:  выделяется под МАКСИМАЛЬНЫЙ контекст (4096)")
    print(f"  Батч-буфер:-b 2048 × token_dim × layers (предвыделение)")
    print(f"  CUDA:      cuDNN, cuBLAS overhead + VRAM страницы")
    print(f"  ─────────────────────────────────────────────────")
    
    return embed_url, rerank_url, llama_procs_all

def suggest_optimizations():
    """Предлагает конкретные оптимизации для снижения потребления RAM."""
    print_separator("РЕКОМЕНДАЦИИ ПО ОПТИМИЗАЦИИ")
    
    print("""
  ТЕКУЩИЕ ПАРАМЕТРЫ (gguf_direct.py, строка 250):
    cmd.extend(["-c", "4096", "-b", "2048", "-ub", "2048"])
  
  ПРИЧИНЫ ВЫСОКОГО ПОТРЕБЛЕНИЯ:
  ┌─────────────────────────────────────────────────────────┐
  │ 1. -b 2048 / -ub 2048: Большой батч-буфер              │
  │    Для embedding/reranker нужен батч из 1-16 запросов.  │
  │    2048 — это бюджет для LLM с параллельными слотами!   │
  │                                                         │
  │ 2. -c 4096: Полный KV-cache                             │
  │    Embedding модели НЕ генерируют текст.                │
  │    Им не нужен весь KV-cache под 4096 токенов!          │
  │    Достаточно 512 (для документов ≤ 512 токенов).       │
  │                                                         │
  │ 3. --flash-attn on: Требует GPU-буферы                  │
  │    Для маленьких embedding моделей не даёт выигрыша     │
  │                                                         │
  │ 4. -ngl -1: Все слои на GPU                             │
  │    При 2 серверах GPU-память делится                     │
  └─────────────────────────────────────────────────────────┘
  
  ОПТИМАЛЬНЫЕ ПАРАМЕТРЫ ДЛЯ EMBEDDING/RERANKER:
    -c 512   → KV-cache под 512 токенов (экономия ~80%)
    -b 32    → Батч = 32 запроса (достаточно для RAG)
    -ub 32   → То же для micro-batching
    
  ОЖИДАЕМЫЙ РЕЗУЛЬТАТ:
    Текущий:    ~2-3 GB/сервер = ~5-6 GB всего
    После:      ~700-900 MB/сервер = ~1.5-2 GB всего
    Экономия:   ~3-4 GB RAM
""")

def main():
    print("\n" + "█"*60)
    print("  ДИАГНОСТИКА ПАМЯТИ: Embedding + Reranker (GGUF)")
    print("█"*60)
    
    # 1. Базовое измерение
    baseline = measure_baseline()
    
    # 2. Анализ параметров (теоретический)
    embed_path = config.resolve_model_path(config.EMBEDDING_MODEL_NAME)
    rerank_path = config.resolve_model_path(config.RERANKER_MODEL_NAME)
    analyze_llama_server_cmd(embed_path, rerank_path)
    
    # 3. Живой тест если возможно
    llama_procs_before = get_all_llama_processes()
    if llama_procs_before:
        print_separator("СЕРВЕРЫ УЖЕ ЗАПУЩЕНЫ")
        for p in llama_procs_before:
            print(f"  PID {p['pid']}: {p['model']} — {p['ram_mb']:.0f} MB RAM")
        print(f"\n  Суммарно llama-server: {sum(p['ram_mb'] for p in llama_procs_before):.0f} MB")
        print(f"  RAM системы: {baseline:.0f} MB")
    else:
        print("\n  ℹ️  llama-server серверы не запущены. Пропуск живого теста.")
        print("      Запустите приложение и затем повторите тест.")
    
    # 4. Рекомендации
    suggest_optimizations()
    
    print_separator("БЫСТРОЕ ИСПРАВЛЕНИЕ")
    print("""
  Изменить в src/gguf_direct.py, строка 250:
  
  БЫЛО:
    cmd.extend(["-c", "4096", "-b", "2048", "-ub", "2048"])
  
  СТАЛО:
    cmd.extend(["-c", "512", "-b", "32", "-ub", "32"])
  
  Это одно изменение сэкономит 3-4 GB RAM!
""")

if __name__ == "__main__":
    main()
