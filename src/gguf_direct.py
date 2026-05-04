"""
GGUF Direct API — использование llama-cpp-python напрямую без сервера.
Быстрее и надёжнее, чем запуск отдельного процесса.
"""

import os
import sys
from typing import Optional, Dict
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import completion_to_prompt, messages_to_prompt

import config

try:
    import llama_cpp
    def dummy_log_callback(level, message, user_data): pass
    llama_cpp.llama_log_set(dummy_log_callback, None)
except: pass

# Глобальный кэш загруженных моделей
_model_cache: Dict[str, LlamaCPP] = {}


def get_gguf_llm(
    gguf_path: str,
    mmproj_path: Optional[str] = None,
    ctx_size: Optional[int] = None,
    gpu_layers: Optional[int] = None,
    temperature: float = 0.1,
    max_tokens: int = 1024,
    n_threads: Optional[int] = None,
    n_batch: Optional[int] = None,
    flash_attn: bool = False,
) -> LlamaCPP:
    """
    Загружает GGUF модель через прямой API llama-cpp-python.
    Кэширует модель по пути — повторные вызовы возвращают уже загруженную модель.
    
    Args:
        gguf_path: абсолютный путь к .gguf файлу
        mmproj_path: путь к mmproj для multimodal моделей
        ctx_size: размер контекста
        gpu_layers: количество GPU слоёв (-1 = все)
        temperature: температура генерации
        max_tokens: максимум токенов в ответе
    
    Returns:
        LlamaCPP instance готовый к использованию
    """
    # Нормализуем путь
    gguf_path = os.path.normpath(gguf_path)
    cache_key = gguf_path
    
    # Если модель уже загружена — возвращаем из кэша
    if cache_key in _model_cache:
        print(f"[GGUF Direct] Используем кэшированную модель: {os.path.basename(gguf_path)}")
        return _model_cache[cache_key]
        
    # Если запрашивается НОВАЯ модель, выгружаем старые, чтобы освободить VRAM
    unload_all_models()
    
    if not os.path.exists(gguf_path):
        raise FileNotFoundError(f"GGUF модель не найдена: {gguf_path}")
    
    ctx = ctx_size or config.GGUF_CTX_SIZE
    ngl = gpu_layers if gpu_layers is not None else config.GGUF_GPU_LAYERS
    
    print(f"[GGUF Direct] Загрузка модели: {os.path.basename(gguf_path)}")
    print(f"[GGUF Direct] Параметры: ctx={ctx}, gpu_layers={ngl}, temp={temperature}")
    
    model_kwargs = {
        "n_gpu_layers": ngl,
        "n_ctx": ctx,
        "n_threads": n_threads if n_threads is not None else config.GGUF_THREADS,
        "n_batch": n_batch if n_batch is not None else 2048,
        "flash_attn": flash_attn,
    }
    
    # Добавляем mmproj если указан
    if mmproj_path and os.path.exists(mmproj_path):
        mmproj_path = os.path.normpath(mmproj_path)
        print(f"[GGUF Direct] Multimodal: {os.path.basename(mmproj_path)}")
        model_kwargs["clip_model_path"] = mmproj_path
    
    try:
        llm = LlamaCPP(
            model_path=gguf_path,
            temperature=temperature,
            max_new_tokens=max_tokens,
            context_window=ctx,
            model_kwargs=model_kwargs,
            verbose=False,
        )
        
        _model_cache[cache_key] = llm
        print(f"[GGUF Direct] Модель загружена успешно")
        return llm
        
    except Exception as e:
        print(f"[GGUF Direct] Ошибка загрузки: {e}")
        raise


def unload_model(gguf_path: str) -> bool:
    """
    Выгружает модель из памяти.
    
    Returns:
        True если модель была выгружена, False если её не было в кэше
    """
    gguf_path = os.path.normpath(gguf_path)
    if gguf_path in _model_cache:
        del _model_cache[gguf_path]
        print(f"[GGUF Direct] Модель выгружена: {os.path.basename(gguf_path)}")
        return True
    return False


def unload_all_models():
    """Выгружает все загруженные модели из памяти."""
    global _model_cache
    count = len(_model_cache)
    if count > 0:
        _model_cache.clear()
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[GGUF Direct] Выгружено моделей: {count}, память очищена.")


def get_loaded_models() -> list:
    """Возвращает список путей загруженных моделей."""
    return list(_model_cache.keys())
