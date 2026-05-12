import os
import re
import subprocess
import time
import requests
import json
import signal
import threading
import logging
from typing import Dict, Optional, Tuple, List
import config

logger = logging.getLogger(__name__)
import torch

# Глобальный кэш запущенных серверов
_server_processes: Dict[str, subprocess.Popen] = {}
_server_ports: Dict[str, int] = {}
_server_configs: Dict[str, Dict] = {}
_lock = threading.Lock()

SERVER_EXE = os.path.join(config.BASE_DIR, "bin", "llama-server.exe")

def detect_model_family(gguf_path: str) -> str:
    """Определяет семейство модели по имени файла."""
    name = os.path.basename(gguf_path).lower()
    if any(x in name for x in ["qwen", "qwq"]):
        return "qwen"
    if "gemma" in name:
        if any(x in name for x in ["gemma-4", "gemma4", "gemma_4", "-4b", "-4e", "e4b"]):
            return "gemma4"
        return "gemma3"
    if any(x in name for x in ["deepseek", "-r1", "_r1"]):
        return "deepseek"
    if "llama" in name:
        return "llama"
    return "generic"

def is_server_ready(port: int) -> bool:
    """Проверяет, готов ли сервер принимать запросы."""
    try:
        r = requests.get(f"http://127.0.0.1:{port}/health", timeout=1)
        return r.status_code == 200
    except Exception:
        return False

def get_gguf_llm(
    gguf_path: str,
    mmproj_path: str = None,
    temperature: float = 0.1,
    ctx_size: int = None,
    gpu_layers: int = -1,
    n_threads: int = None,
    n_batch: int = 2048,
    flash_attn: bool = True,
    max_tokens: int = 4096,
    type_k: int = 2,
    type_v: int = 2,
    enable_thinking: bool = True,
    thinking_budget: int = 1024,
    n_parallel: int = 1,
    custom_args: Optional[List[str]] = None,
) -> str:
    """
    Запускает llama-server.exe для указанной модели.
    Возвращает URL сервера (например, http://127.0.0.1:49152).
    """
    # Нормализуем путь к модели (для Windows регистр не важен)
    gguf_path = os.path.normpath(config.resolve_model_path(gguf_path)).lower()
    if mmproj_path:
        mmproj_path = os.path.normpath(config.resolve_model_path(mmproj_path)).lower()
    
    # Собираем текущий конфиг запроса для сравнения
    # Нормализуем значения (None -> default), чтобы избежать ложных перезапусков
    current_config = {
        "mmproj": mmproj_path or None,
        "ctx_size": int(ctx_size or config.GGUF_CTX_SIZE),
        "gpu_layers": int(gpu_layers if gpu_layers is not None else -1),
        "n_batch": int(n_batch or 2048),
        "flash_attn": bool(flash_attn),
        "max_tokens": int(max_tokens or 4096),
        "type_k": int(type_k or 2),
        "type_v": int(type_v or 2),
        "enable_thinking": bool(enable_thinking),
        "thinking_budget": int(thinking_budget if thinking_budget is not None else 1024),
        "n_parallel": int(n_parallel or 1),
        "custom_args": custom_args if custom_args is not None else []
    }

    with _lock:
        if gguf_path in _server_processes:
            # Если процесс жив И конфиг совпадает — возвращаем URL
            if _server_processes[gguf_path].poll() is None and _server_configs.get(gguf_path) == current_config:
                return f"http://127.0.0.1:{_server_ports[gguf_path]}"
            else:
                print(f"[GGUF Server] Настройки изменились или сервер упал. Перезапуск {os.path.basename(gguf_path)}...")
                # Мы не удаляем его из _server_processes здесь, 
                # чтобы unload_all_models() ниже гарантированно его прибил.

        # Выгружаем другие модели ПЕРЕД запуском новой (экономия VRAM)
        # НО: делаем это только если мы РЕАЛЬНО запускаем новый сервер
        from src.rag_pipeline import unload_rag_models
        unload_rag_models()
        kill_stray_servers() # Принудительно чистим всё перед запуском нового
        unload_all_models() # Сбрасываем внутренний стейт

        if not os.path.exists(gguf_path):
            raise FileNotFoundError(f"GGUF модель не найдена: {gguf_path}")

        # Подбираем свободный порт
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(('', 0))
        port = s.getsockname()[1]
        s.close()

    # Для параллельной работы нужно расширить общий контекст, чтобы каждому слоту хватило места
    total_ctx = current_config["ctx_size"] * current_config["n_parallel"]
    
    # Маппинг типов квантования для llama-server
    CACHE_TYPE_MAP = {
        0: "f16",
        1: "f32",
        2: "q4_0", # q4_k часто мапится на q4_0 в простых версиях
        8: "q8_0"
    }
    # Пытаемся получить строковое значение, если пришло число
    type_k_str = CACHE_TYPE_MAP.get(current_config["type_k"], "f16")
    type_v_str = CACHE_TYPE_MAP.get(current_config["type_v"], "f16")

    cmd = [
        SERVER_EXE,
        "-m", gguf_path,
        "--port", str(port),
        "-c", str(total_ctx),
        "-ngl", str(current_config["gpu_layers"]),
        "-b", str(current_config["n_batch"]),
        "--parallel", str(current_config["n_parallel"]),
        "--cont-batching",
        "--jinja",
        "--cache-type-k", type_k_str,
        "--cache-type-v", type_v_str,
        "-n", str(current_config["max_tokens"])
    ]

    # Если рассуждения отключены — добавляем соответствующие флаги сервера
    if not current_config["enable_thinking"]:
        cmd.extend([
            "--reasoning", "off", 
            "--reasoning-format", "none", 
            "--reasoning-budget", "0"
        ])
    else:
        # Если включены — явно указываем это и задаем бюджет
        cmd.extend([
            "--reasoning", "on",
            "--reasoning-budget", str(current_config["thinking_budget"])
        ])
    
    if current_config["flash_attn"]:
        # Добавляем только если его нет в custom_args
        if not any("-fa" in str(arg) or "--flash-attn" in str(arg) for arg in (current_config["custom_args"] or [])):
            cmd.extend(["--flash-attn", "on"])
    
    if n_threads and n_threads > 0:
        cmd.extend(["-t", str(n_threads)])
    
    if current_config["mmproj"] and os.path.exists(current_config["mmproj"]):
        cmd.extend(["--mmproj", os.path.normpath(current_config["mmproj"])])
        print(f"[GGUF Server] С поддержкой Vision: {os.path.basename(current_config['mmproj'])}")

    if current_config["custom_args"]:
        # Добавляем кастомные аргументы в конец
        cmd.extend(current_config["custom_args"])

    print(f"[GGUF Server] Запуск: {os.path.basename(gguf_path)} на порту {port}...")
    
    # Запускаем процесс. На Windows используем CREATE_NO_WINDOW.
    # ВАЖНО: Мы перенаправляем вывод в DEVNULL, чтобы избежать переполнения буфера PIPE, 
    # которое вызывает зависание процесса llama-server на Windows.
    creationflags = 0x08000000 # CREATE_NO_WINDOW
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags
    )
    
    # Ждем готовности сервера (до 60 секунд)
    start_wait = time.time()
    while time.time() - start_wait < 60:
        if is_server_ready(port):
            print(f"[GGUF Server] Готов!")
            _server_processes[gguf_path] = process
            _server_ports[gguf_path] = port
            _server_configs[gguf_path] = current_config
            return f"http://127.0.0.1:{port}"
        
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise RuntimeError(f"Сервер упал при запуске:\n{stderr}")
            
        time.sleep(0.5)
    
    process.terminate()
    raise TimeoutError("Сервер не ответил за 60 секунд")

def stream_gguf_chat(
    llm_url: str,
    messages: list,
    enable_thinking: bool,
    max_tokens: int,
    temperature: float,
    repeat_penalty: float,
    top_p: float,
    min_p: float,
    model_family: str = "generic"
):
    """Стриминг через OpenAI-совместимый API сервера llama.cpp."""
    payload = {
        "messages": messages,
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "repeat_penalty": repeat_penalty,
        "top_p": top_p,
        "min_p": min_p,
    }
    
    # Определяем теги на основе семейства
    OPEN_TAG, CLOSE_TAG = ("<|channel|>", "<channel|>") if model_family == "gemma4" else ("<think>", "</think>")
    
    try:
        r = requests.post(
            f"{llm_url}/v1/chat/completions",
            json=payload,
            stream=True,
            timeout=60
        )
        
        is_thinking = False
        for line in r.iter_lines():
            if line:
                line_str = line.decode("utf-8")
                if line_str.startswith("data: "):
                    if line_str == "data: [DONE]":
                        break
                    try:
                        data = json.loads(line_str[6:])
                        delta = data["choices"][0]["delta"]
                        
                        # 1. Проверяем наличие reasoning_content (новый формат llama.cpp / OpenAI)
                        reasoning = delta.get("reasoning_content", "")
                        if reasoning:
                            if not is_thinking:
                                yield OPEN_TAG
                                is_thinking = True
                            yield reasoning
                            continue
                            
                        # 2. Проверяем наличие обычного контента
                        content = delta.get("content", "")
                        if content:
                            # Если пошел текст, но мы еще "думали" — закрываем тег
                            if is_thinking:
                                yield CLOSE_TAG
                                is_thinking = False
                            yield content
                    except Exception as e:
                        logger.debug(f"Ошибка парсинга SSE: {e}")
                        continue
        
        # На всякий случай закрываем тег в конце
        if is_thinking:
            yield CLOSE_TAG

    except Exception as e:
        print(f"[GGUF Stream] Ошибка: {e}")
        yield f"Ошибка связи с сервером: {e}"

def kill_stray_servers():
    """Убивает все запущенные процессы llama-server.exe в системе (Windows/Linux)."""
    print("[GGUF Server] Поиск и завершение сторонних процессов llama-server...")
    try:
        if os.name == 'nt':
            # /F - принудительно, /IM - по имени образа, /T - дерево процессов
            subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe", "/T"], capture_output=True)
        else:
            subprocess.run(["pkill", "-9", "llama-server"], capture_output=True)
    except Exception as e:
        print(f"[GGUF Server] Ошибка при очистке процессов: {e}")

def count_running_servers() -> int:
    """Считает количество запущенных процессов llama-server.exe в системе."""
    try:
        if os.name == 'nt':
            output = subprocess.check_output(['tasklist', '/FI', 'IMAGENAME eq llama-server.exe', '/NH'], text=True)
            return output.count("llama-server.exe")
        else:
            output = subprocess.check_output(['pgrep', '-c', 'llama-server'], text=True)
            return int(output.strip())
    except Exception:
        return 0

def unload_all_models():
    """Убивает все процессы серверов максимально надежно."""
    global _server_processes, _server_ports, _server_configs
    if not _server_processes:
        return

    print(f"[GGUF Server] Выгрузка всех моделей: {list(map(os.path.basename, _server_processes.keys()))}")
    
    for path, process in _server_processes.items():
        if process.poll() is None: # Если еще живой
            try:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
            except Exception as e:
                print(f"[GGUF Server] Ошибка при остановке {os.path.basename(path)}: {e}")
    
    _server_processes = {}
    _server_ports = {}
    _server_configs = {}
    
    # На Windows иногда процессы зависают, пробуем почистить по имени порта если нужно, 
    # но пока ограничимся gc.
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_loaded_models():
    """Возвращает список путей к запущенным моделям."""
    return list(_server_processes.keys())
