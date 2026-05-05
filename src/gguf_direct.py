import os
import llama_cpp
from typing import Dict, Optional
import config

# Отключаем лишние логи llama-cpp
try:
    def dummy_log_callback(level, message, user_data):
        pass
    llama_cpp.llama_log_set(dummy_log_callback, None)
except: pass

# Глобальный кэш загруженных моделей
_model_cache: Dict[str, llama_cpp.Llama] = {}

def get_gguf_llm(gguf_path: str, mmproj_path: str = None, temperature: float = 0.3, ctx_size: int = 8192, 
                 gpu_layers: int = -1, n_threads: int = None, n_batch: int = 2048, flash_attn: bool = False, 
                 max_tokens: int = 2048, type_k: int = 2, type_v: int = 2) -> llama_cpp.Llama:
    """
    Загружает GGUF модель через прямой API llama-cpp-python.
    Кэширует модель по пути — повторные вызовы возвращают уже загруженную модель.
    """
    # Нормализуем путь
    gguf_path = os.path.normpath(gguf_path)
    cache_key = gguf_path
    
    # Если модель уже загружена — возвращаем из кэша
    if cache_key in _model_cache:
        print(f"[GGUF Direct] Используем кэшированную модель: {os.path.basename(gguf_path)}")
        return _model_cache[cache_key]
        
    # Если запрашивается НОВАЯ модель, выгружаем старые, чтобы освободить VRAM
    from src.rag_pipeline import unload_rag_models
    unload_rag_models()
    
    if not os.path.exists(gguf_path):
        raise FileNotFoundError(f"GGUF модель не найдена: {gguf_path}")

    print(f"[GGUF Direct] Загрузка модели: {os.path.basename(gguf_path)}")
    
    llama_kwargs = {
        "model_path": gguf_path,
        "n_ctx": ctx_size,
        "n_gpu_layers": gpu_layers,
        "n_threads": n_threads if n_threads is not None else config.GGUF_THREADS,
        "n_batch": n_batch if n_batch is not None else 2048,
        "flash_attn": flash_attn,
        "type_k": type_k,
        "type_v": type_v,
        "verbose": False
    }
    
    # Добавляем mmproj если указан (нужно для мультимодальности)
    if mmproj_path and os.path.exists(mmproj_path):
        from llama_cpp.llama_chat_format import Llava15ChatHandler
        # Передаем clip_model_path ТОЛЬКО в ChatHandler, чтобы избежать конфликтов при загрузке
        llama_kwargs["chat_handler"] = Llava15ChatHandler(clip_model_path=os.path.normpath(mmproj_path), verbose=False)
        print(f"[GGUF Direct] Подключен CLIP модуль: {os.path.basename(mmproj_path)}")
    
    llm = llama_cpp.Llama(**llama_kwargs)
    _model_cache[cache_key] = llm
    return llm

def unload_all_models():
    """Полная очистка всех загруженных моделей из кэша и VRAM"""
    global _model_cache
    for path, model in _model_cache.items():
        print(f"[GGUF Direct] Выгрузка модели: {os.path.basename(path)}")
        del model
    _model_cache = {}
    import gc
    gc.collect()

def get_loaded_models():
    """Возвращает список путей к загруженным моделям"""
    return list(_model_cache.keys())
