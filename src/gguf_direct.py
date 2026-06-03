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
# Глобальный кэш запущенных серверов
_server_processes: Dict[str, subprocess.Popen] = {}
_server_ports: Dict[str, int] = {}
_server_configs: Dict[str, Dict] = {}
_server_roles: Dict[str, str] = {} # gguf_path -> role
_lock = threading.Lock()

# Windows Job Object для гарантированного удаления дочерних процессов при выходе
_win32_job = None
if os.name == 'nt':
    try:
        import ctypes
        from ctypes import wintypes
        _win32_job = ctypes.windll.kernel32.CreateJobObjectW(None, None)
        # ExtendedLimitInformation = 9, LimitFlags offset is 16
        # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
        limit_info = (wintypes.DWORD * 36)() # 144 bytes for x64
        limit_info[4] = 0x2000 
        ctypes.windll.kernel32.SetInformationJobObject(_win32_job, 9, ctypes.byref(limit_info), ctypes.sizeof(limit_info))
    except Exception as e:
        print(f"[GGUF Server] Ошибка инициализации Windows Job Object: {e}")

def _assign_to_job(process):
    """Привязывает процесс к Job Object (только для Windows)."""
    if _win32_job is not None:
        try:
            import ctypes
            from ctypes import wintypes
            ctypes.windll.kernel32.AssignProcessToJobObject(_win32_job, wintypes.HANDLE(int(process._handle)))
        except Exception: pass


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
        "custom_args": custom_args if custom_args is not None else [],
        "mtp": "mtp" in os.path.basename(gguf_path).lower(),
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

        # Выгружаем другие LLM модели ПЕРЕД запуском новой (экономия VRAM)
        # НО: делаем это только если мы РЕАЛЬНО запускаем новый сервер
        from src.rag_pipeline import unload_rag_models
        unload_rag_models(hard=False)
        unload_all_models(role="llm") # Сбрасываем только другие LLM

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
        "-b", "512",
        "-ub", "256",
        "--parallel", str(current_config["n_parallel"]),
        "--cont-batching",
        "--jinja",
        "--cache-type-k", type_k_str,
        "--cache-type-v", type_v_str,
        "-n", str(current_config["max_tokens"])
    ]

    if current_config["mtp"]:
        cmd.extend(["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"])

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
    _assign_to_job(process)
    
    # Ждем готовности сервера (до 60 секунд)
    start_wait = time.time()
    while time.time() - start_wait < 60:
        if is_server_ready(port):
            print(f"[GGUF Server] Готов!")
            _server_processes[gguf_path] = process
            _server_ports[gguf_path] = port
            _server_configs[gguf_path] = current_config
            _server_roles[gguf_path] = "llm"
            return f"http://127.0.0.1:{port}"
        
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise RuntimeError(f"Сервер упал при запуске:\n{stderr}")
            
        time.sleep(0.5)
    
    process.terminate()
    raise TimeoutError("Сервер не ответил за 60 секунд")

def get_gguf_embedding_url(gguf_path: str, n_threads: int = None, is_reranker: bool = False) -> str:
    """Запускает llama-server для эмбеддингов или реранкера и возвращает URL."""
    global _server_processes, _server_ports, _server_configs, _server_roles
    
    role = "reranker" if is_reranker else "embedding"
    
    current_config = {
        "n_threads": n_threads,
        "is_reranker": is_reranker
    }

    with _lock:
        if gguf_path in _server_processes:
            if _server_processes[gguf_path].poll() is None and _server_configs.get(gguf_path) == current_config:
                return f"http://127.0.0.1:{_server_ports[gguf_path]}"
            else:
                print(f"[GGUF Server] Перезапуск {role} {os.path.basename(gguf_path)}...")
                
        # Выгружаем другие модели ТОЙ ЖЕ РОЛИ ПЕРЕД запуском новой
        from src.rag_pipeline import unload_rag_models
        unload_rag_models(hard=False)
        unload_all_models(role=role)

        if not os.path.exists(gguf_path):
            raise FileNotFoundError(f"GGUF модель не найдена: {gguf_path}")

        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(('', 0))
        port = s.getsockname()[1]
        s.close()

        cmd = [SERVER_EXE, "-m", gguf_path, "--port", str(port)]
        
        # Эмбеддинги требуют --embedding
        if not is_reranker:
            cmd.extend(["--embedding"])
            if "qwen" in os.path.basename(gguf_path).lower():
                cmd.extend(["--override-kv", "tokenizer.ggml.suffix_token_id=int:151643"])
        else:
            cmd.extend(["--reranking"])
            
        # Embedding/Reranker не генерируют текст авторегрессивно.
        # Параметры зависят от роли:
        #
        # Embedding: -c 2048, -b 2048
        #   Чанк обрабатывается целиком за один проход, -b 2048 гарантирует,
        #   что тексты длиннее 32 токенов (например, для Gemma) не вызовут ошибку 500.
        #
        # Reranker: -c 4096, -b 2048
        #   /v1/rerank оценивает каждую пару (query + doc) НЕЗАВИСИМО.
        #   -c 4096: достаточно для query(~50) + doc(2048) = 2100 токенов с запасом
        #   -b 2048: обрабатывать весь документ за один forward-pass.
        #   Без этого (-b 32) 2048-токенный чанк делится на 64 микро-части,
        #   pooling даёт неправильный score (0.0).
        if is_reranker:
            ctx, b_size = "4096", "2048"
        else:
            ctx, b_size = "2048", "2048"
        cmd.extend(["-c", ctx, "-b", b_size, "-ub", b_size])
        
        # Квантование KV-cache: q8_0 = 50% экономии памяти против f16.
        # Безопасно для embedding/reranker: они не генерируют текст авторегрессивно,
        # поэтому ошибки квантования не накапливаются от токена к токену.
        cmd.extend(["--cache-type-k", "q8_0", "--cache-type-v", "q8_0"])
        
        # Добавляем флаги оптимизации, как у LLM
        if config.GGUF_GPU_LAYERS != 0:
            cmd.extend(["-ngl", str(config.GGUF_GPU_LAYERS)])
        cmd.extend(["--flash-attn", "on"])
        
        if n_threads and n_threads > 0:
            cmd.extend(["-t", str(n_threads)])

        print(f"[GGUF Server] Запуск {role}: {os.path.basename(gguf_path)} на порту {port}...")
        
        creationflags = 0x08000000 # CREATE_NO_WINDOW
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags
        )
        _assign_to_job(process)
        
        start_wait = time.time()
        while time.time() - start_wait < 60:
            if is_server_ready(port):
                print(f"[GGUF Server] {role.capitalize()} готов!")
                _server_processes[gguf_path] = process
                _server_ports[gguf_path] = port
                _server_configs[gguf_path] = current_config
                _server_roles[gguf_path] = role
                return f"http://127.0.0.1:{port}"
            
            if process.poll() is not None:
                raise RuntimeError(f"{role.capitalize()} сервер упал при запуске")
                
            time.sleep(0.5)
        
        process.terminate()
        raise TimeoutError(f"{role.capitalize()} сервер не ответил за 60 секунд")

async def stream_gguf_chat(
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
    """Асинхронный стриминг через OpenAI-совместимый API сервера llama.cpp."""
    import httpx
    
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
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", f"{llm_url}/v1/chat/completions", json=payload) as r:
                r.raise_for_status()
                is_thinking = False
                
                async for line in r.aiter_lines():
                    if line:
                        line_str = line
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

def unload_all_models(role: str = None):
    """Убивает процессы серверов. Если указан role, выгружает только серверы с этой ролью."""
    global _server_processes, _server_ports, _server_configs, _server_roles
    if not _server_processes:
        return

    to_remove = []
    for path, process in _server_processes.items():
        if role is not None and _server_roles.get(path) != role:
            continue
            
        print(f"[GGUF Server] Выгрузка модели ({_server_roles.get(path, 'unknown')}): {os.path.basename(path)}")
        to_remove.append(path)
        
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
    for path in to_remove:
        _server_processes.pop(path, None)
        _server_ports.pop(path, None)
        _server_configs.pop(path, None)
        _server_roles.pop(path, None)
    
    # На Windows иногда процессы зависают, пробуем почистить по имени порта если нужно, 
    # но пока ограничимся gc.
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_loaded_models():
    """Возвращает список путей к запущенным LLM моделям (без эмбеддингов)."""
    return [path for path in _server_processes.keys() if _server_roles.get(path, "llm") == "llm"]

def get_active_llm_url() -> str | None:
    """Возвращает URL первого живого LLM-сервера (роль 'llm'), или None если нет.
    
    Используется для Query Expansion: позволяет использовать уже запущенный
    GGUF-сервер вместо LM Studio.
    """
    with _lock:
        for path, process in _server_processes.items():
            if _server_roles.get(path) == "llm" and process.poll() is None:
                port = _server_ports.get(path)
                if port:
                    return f"http://127.0.0.1:{port}"
    return None
