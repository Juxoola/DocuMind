import os
import sys
import subprocess
import time
import requests
import json
import threading
import logging
from typing import Dict, Optional, List
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

# Состояние последней загрузки LLM (для UI: idle/loading/ready/error)
# Используется при hot-swap модели через /api/preload-llm.
_llm_load_state: Dict = {
    "state": "idle",       # idle | loading | ready | error
    "model": None,         # path текущей/загружаемой модели
    "port": None,          # порт (когда ready)
    "task_id": None,       # уникальный ID фоновой задачи
    "started_at": None,    # time.time() начала загрузки
    "ready_at": None,      # time.time() готовности
    "last_load_seconds": None,  # последнее время загрузки (сек) для ETA
    "error": None,         # текст ошибки
    "phase": None,         # freeing | starting | loading_model | probing | ready
}

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
        except Exception as e:
            # F-fix #silent-except: Job Object — best-effort. Если привязка
            # не удалась (старый pywin32, нет прав), процесс не умрёт с
            # родителем, но в любом случае сработает kill_all в lifespan.
            # Логируем debug — это не критично, но полезно при диагностике.
            logger.debug(f"AssignProcessToJobObject failed (non-critical): {e}")


# Маппинг типов квантования KV-кэша для llama-server.
# Ключ — значение gguf_kv_quant из UI/Settings; значение — флаг --cache-type-k/v.
#
# F-fix #34: q4_k и q5_k НЕ поддерживаются llama.cpp для KV-cache.
# Допустимы только: f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1.
# Коммит cd2dd6e ошибочно добавил q4_k/q5_k в map — llama-server падал
# при старте с "retcode=1" и пустым stderr.
CACHE_TYPE_MAP = {
    0: "f16",
    1: "f32",
    2: "q4_0",
    3: "q4_1",
    4: "q4_0",  # было "q4_k" — невалидно. Fallback на q4_0.
    6: "q5_0",  # было "q5_k" — невалидно. Fallback на q5_0.
    8: "q8_0",
}

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

def _start_llm_server_sync(
    gguf_path: str,
    mmproj_path: str,
    current_config: Dict,
) -> str:
    """
    Внутренняя функция: реально запускает llama-server и ждёт готовности.
    Блокирует вызывающий поток на 10-60 секунд.
    Возвращает URL при успехе, бросает исключение при ошибке.
    """
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()

    # Для параллельной работы нужно расширить общий контекст, чтобы каждому слоту хватило места
    total_ctx = current_config["ctx_size"] * current_config["n_parallel"]

    type_k_str = CACHE_TYPE_MAP.get(current_config["type_k"], "f16")
    type_v_str = CACHE_TYPE_MAP.get(current_config["type_v"], "f16")

    cmd = [
        SERVER_EXE,
        "-m", gguf_path,
        "--port", str(port),
        "-c", str(total_ctx),
        "-ngl", str(current_config["gpu_layers"]),
        "-b", str(current_config["n_batch"]),
        "-ub", str(current_config["n_ubatch"]),
        "--parallel", str(current_config["n_parallel"]),
        "--cont-batching",
        "--jinja",
        "--cache-type-k", type_k_str,
        "--cache-type-v", type_v_str,
        "-n", str(current_config["max_tokens"])
    ]

    if current_config["mtp_enabled"]:
        cmd.extend(["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"])

    if not current_config["enable_thinking"]:
        cmd.extend([
            "--reasoning", "off",
            "--reasoning-format", "none",
            "--reasoning-budget", "0"
        ])
    else:
        cmd.extend([
            "--reasoning", "on",
            "--reasoning-budget", str(current_config["thinking_budget"])
        ])

    if current_config["flash_attn"]:
        if not any("-fa" in str(arg) or "--flash-attn" in str(arg) for arg in (current_config["custom_args"] or [])):
            cmd.extend(["--flash-attn", "on"])

    if current_config.get("_n_threads") and current_config["_n_threads"] > 0:
        cmd.extend(["-t", str(current_config["_n_threads"])])

    if current_config["mmproj"] and os.path.exists(current_config["mmproj"]):
        cmd.extend(["--mmproj", os.path.normpath(current_config["mmproj"])])
        print(f"[GGUF Server] С поддержкой Vision: {os.path.basename(current_config['mmproj'])}")

    if current_config["custom_args"]:
        cmd.extend(current_config["custom_args"])

    # F-fix #33: печатаем полную команду при старте, чтобы при падении
    # видеть какие параметры реально были переданы (а не гадать).
    print(f"[GGUF Server] Запуск: {os.path.basename(gguf_path)} на порту {port}...")
    print(f"[GGUF Server]   cmd: {' '.join(cmd)}")

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
            print(f"[GGUF Server] Готов!")
            with _lock:
                _server_processes[gguf_path] = process
                _server_ports[gguf_path] = port
                _server_configs[gguf_path] = current_config
                _server_roles[gguf_path] = "llm"
            return f"http://127.0.0.1:{port}"

        if process.poll() is not None:
            try:
                _, stderr = process.communicate(timeout=1)
            except Exception:
                stderr = b""
            # F-fix #31: process.communicate() может вернуть None для stderr,
            # если stream был DEVNULL и pipe уже закрыт. None.decode() → AttributeError.
            # F-fix #32: добавляем return code и pid в сообщение, иначе при OOM
            # или corrupt model файл непонятно что произошло.
            stderr_text = (stderr or b"").decode('utf-8', errors='ignore')[:500]
            retcode = process.returncode
            raise RuntimeError(
                f"Сервер llama-server упал при запуске (pid={process.pid}, retcode={retcode}). "
                f"Возможные причины: OOM GPU, повреждённый .gguf, отсутствует mmproj для vision, "
                f"несовместимая версия llama-server. stderr: {stderr_text}"
            )

        time.sleep(0.5)

    # F-fix #8: при таймауте запуска используем taskkill /F /T (см. unload_all_models).
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                capture_output=True, timeout=5
            )
        else:
            process.kill()
    except Exception as e:
        # F-fix #silent-except: первый terminate() упал. Пробуем kill().
        # Если и он упадёт — процесс скорее всего уже мёртв.
        logger.debug(f"terminate() failed: {e}, trying kill()")
        try: process.kill()
        except Exception as kill_err:
            logger.debug(f"kill() also failed (process likely already dead): {kill_err}")
    try: process.wait(timeout=5)
    except Exception as e:
        # wait() может таймаутить если процесс завис. Логируем — это
        # индикатор проблемы с завершением на Windows.
        logger.debug(f"process.wait() timed out (process may be stuck): {e}")
    raise TimeoutError("Сервер не ответил за 60 секунд")


def get_gguf_llm(
    gguf_path: str,
    mmproj_path: str = None,
    temperature: float = 0.1,
    ctx_size: int = None,
    gpu_layers: int = -1,
    n_threads: int = None,
    n_batch: int = 512,
    flash_attn: bool = True,
    max_tokens: int = 4096,
    type_k: int = 2,
    type_v: int = 2,
    enable_thinking: bool = True,
    thinking_budget: int = 1024,
    n_parallel: int = 1,
    custom_args: Optional[List[str]] = None,
    mtp_enabled: bool = False,
    n_ubatch: int = 256,
) -> str:
    """
    Запускает llama-server.exe для указанной модели (синхронно, блокирует).
    Возвращает URL сервера (например, http://127.0.0.1:49152).
    Для асинхронной загрузки с UI-прогрессом используй preload_gguf_llm().
    """
    # Нормализуем путь к модели
    gguf_path = os.path.normpath(config.resolve_model_path(gguf_path)).lower()
    if mmproj_path:
        mmproj_path = os.path.normpath(config.resolve_model_path(mmproj_path)).lower()

    current_config = {
        "mmproj": mmproj_path or None,
        "ctx_size": int(ctx_size or config.GGUF_CTX_SIZE),
        "gpu_layers": int(gpu_layers if gpu_layers is not None else -1),
        "n_batch": int(n_batch or 512),
        "flash_attn": bool(flash_attn),
        "max_tokens": int(max_tokens or 4096),
        "type_k": int(type_k or 2),
        "type_v": int(type_v or 2),
        "enable_thinking": bool(enable_thinking),
        "thinking_budget": int(thinking_budget if thinking_budget is not None else 1024),
        "n_parallel": int(n_parallel or 1),
        "custom_args": custom_args if custom_args is not None else [],
        "mtp_enabled": bool(mtp_enabled),
        "n_ubatch": int(n_ubatch or 256),
        "_n_threads": n_threads,
    }

    with _lock:
        if gguf_path in _server_processes:
            if _server_processes[gguf_path].poll() is None and _server_configs.get(gguf_path) == current_config:
                # Уже готов с тем же конфигом
                _llm_load_state.update({"state": "ready", "model": gguf_path, "port": _server_ports[gguf_path], "error": None})
                return f"http://127.0.0.1:{_server_ports[gguf_path]}"
            else:
                print(f"[GGUF Server] Настройки изменились или сервер упал. Перезапуск {os.path.basename(gguf_path)}...")

        # Помечаем состояние "loading" (если вызывающий не через preload)
        _llm_load_state.update({
            "state": "loading", "model": gguf_path, "port": None,
            "task_id": None, "started_at": time.time(), "ready_at": None,
            "error": None, "phase": "starting"
        })
        unload_rag_models_safe()
        unload_all_models(role="llm")

        if not os.path.exists(gguf_path):
            _llm_load_state.update({"state": "error", "error": f"Model not found: {gguf_path}"})
            raise FileNotFoundError(f"GGUF модель не найдена: {gguf_path}")

    # Запускаем вне _lock
    try:
        url = _start_llm_server_sync(gguf_path, mmproj_path, current_config)
        elapsed = time.time() - (_llm_load_state.get("started_at") or time.time())
        with _lock:
            _llm_load_state.update({
                "state": "ready", "port": _server_ports.get(gguf_path),
                "ready_at": time.time(), "last_load_seconds": elapsed,
                "phase": "ready", "error": None
            })
        return url
    except Exception as e:
        with _lock:
            _llm_load_state.update({"state": "error", "error": str(e)[:300], "phase": None})
        raise


def unload_rag_models_safe():
    """Best-effort unload of RAG models to free VRAM before LLM start."""
    try:
        from src.rag_pipeline import unload_rag_models
        unload_rag_models(hard=False)
    except Exception:
        pass


def preload_gguf_llm(
    gguf_path: str,
    mmproj_path: str = None,
    ctx_size: int = None,
    gpu_layers: int = -1,
    n_threads: int = None,
    n_batch: int = 512,
    flash_attn: bool = True,
    max_tokens: int = 4096,
    type_k: int = 2,
    type_v: int = 2,
    enable_thinking: bool = True,
    thinking_budget: int = 1024,
    n_parallel: int = 1,
    custom_args: Optional[List[str]] = None,
    mtp_enabled: bool = False,
    n_ubatch: int = 256,
) -> Dict:
    """
    Асинхронно запускает llama-server в фоне. Возвращает сразу.
    UI следит за прогрессом через get_llm_status() / stream_llm_status().
    """
    import uuid
    gguf_path = os.path.normpath(config.resolve_model_path(gguf_path)).lower()
    if mmproj_path:
        mmproj_path = os.path.normpath(config.resolve_model_path(mmproj_path)).lower()

    current_config = {
        "mmproj": mmproj_path or None,
        "ctx_size": int(ctx_size or config.GGUF_CTX_SIZE),
        "gpu_layers": int(gpu_layers if gpu_layers is not None else -1),
        "n_batch": int(n_batch or 512),
        "flash_attn": bool(flash_attn),
        "max_tokens": int(max_tokens or 4096),
        "type_k": int(type_k or 2),
        "type_v": int(type_v or 2),
        "enable_thinking": bool(enable_thinking),
        "thinking_budget": int(thinking_budget if thinking_budget is not None else 1024),
        "n_parallel": int(n_parallel or 1),
        "custom_args": custom_args if custom_args is not None else [],
        "mtp_enabled": bool(mtp_enabled),
        "n_ubatch": int(n_ubatch or 256),
        "_n_threads": n_threads,
    }

    task_id = str(uuid.uuid4())[:8]

    # Если уже загружена с тем же конфигом — мгновенный ready
    with _lock:
        if gguf_path in _server_processes:
            if _server_processes[gguf_path].poll() is None and _server_configs.get(gguf_path) == current_config:
                _llm_load_state.update({
                    "state": "ready", "model": gguf_path,
                    "port": _server_ports[gguf_path], "task_id": task_id,
                    "started_at": time.time(), "ready_at": time.time(),
                    "phase": "ready", "error": None
                })
                return {"status": "ready", "port": _server_ports[gguf_path], "task_id": task_id}

    # Помечаем loading и запускаем фоновый поток
    _llm_load_state.update({
        "state": "loading", "model": gguf_path, "port": None,
        "task_id": task_id, "started_at": time.time(),
        "ready_at": None, "error": None, "phase": "freeing"
    })

    def _worker():
        try:
            _llm_load_state["phase"] = "starting"
            unload_rag_models_safe()
            unload_all_models(role="llm")
            if not os.path.exists(gguf_path):
                raise FileNotFoundError(f"GGUF модель не найдена: {gguf_path}")
            _llm_load_state["phase"] = "loading_model"
            url = _start_llm_server_sync(gguf_path, mmproj_path, current_config)
            elapsed = time.time() - _llm_load_state["started_at"]
            _llm_load_state.update({
                "state": "ready",
                "port": _server_ports.get(gguf_path),
                "ready_at": time.time(),
                "last_load_seconds": elapsed,
                "phase": "ready",
                "error": None
            })
            print(f"[preload] OK: loaded in {elapsed:.1f}s")
        except Exception as e:
            _llm_load_state.update({
                "state": "error", "error": str(e)[:300], "phase": None
            })
            print(f"[preload] ERROR: {e}")

    thread = threading.Thread(target=_worker, daemon=True, name=f"preload-llm-{task_id}")
    thread.start()
    return {"status": "loading", "task_id": task_id, "model": os.path.basename(gguf_path)}


def get_llm_status() -> Dict:
    """Возвращает текущее состояние LLM для UI."""
    with _lock:
        state = _llm_load_state.copy()
    # Дополняем elapsed/eta
    if state.get("started_at") and state.get("state") == "loading":
        elapsed = time.time() - state["started_at"]
        state["elapsed"] = round(elapsed, 1)
        last = state.get("last_load_seconds")
        if last:
            state["eta"] = round(max(0, last - elapsed), 1)
        else:
            # ETA по типичной скорости: ~1с на 100MB
            try:
                size_mb = os.path.getsize(state["model"]) / (1024*1024) if state.get("model") else 0
                state["eta"] = round(max(5, size_mb / 100), 1)
            except Exception:
                state["eta"] = None
    elif state.get("state") == "ready":
        state["elapsed"] = round((state.get("ready_at") or time.time()) - (state.get("started_at") or time.time()), 1)
        state["eta"] = 0
    return state


def get_gguf_embedding_url(gguf_path: str, n_threads: int = None, is_reranker: bool = False, n_parallel: int = 1) -> str:
    """Запускает llama-server для эмбеддингов или реранкера и возвращает URL.

    Args:
        n_parallel: количество параллельных слотов на сервере.
            Должно совпадать с embed_batch_size в OpenAIEmbedding,
            иначе запросы сериализуются на сервере и рост HTTP overhead.
    """
    global _server_processes, _server_ports, _server_configs, _server_roles

    role = "reranker" if is_reranker else "embedding"

    n_parallel = max(1, int(n_parallel or 1))

    current_config = {
        "n_threads": n_threads,
        "is_reranker": is_reranker,
        "n_parallel": n_parallel,
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

        cmd = [SERVER_EXE, "-m", gguf_path, "--port", str(port), "--parallel", str(n_parallel)]

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
        # Embedding: -c 4096, -b 512, -ub 512
        #   v_splitter в process_pdf режет vision-описания на чанки по 2048 символов.
        #   Для русского текста 2048 chars ≈ 600-1500 токенов + prefix "Изображение PDF имя стр N: " ≈ 200-300 токенов.
        #   С -c 2048 длинные описания (3000+ символов) превышали контекст → 500 "Context size has been exceeded"
        #   и падение build_index. -c 4096 даёт запас с учётом prefix и Cyrillic-токенизации.
        #   -b 512 (вместо 2048): для 0.6B модели 2048 batch выделяет огромные pre-allocated attention buffers
        #   (~3-4GB на процесс через CUDA scratch). 512 хватает для чанков ≤2048 символов (≈1500 токенов).
        #
        # Reranker: -c 4096, -b 2048, -ub 2048
        #   /v1/rerank оценивает каждую пару (query + doc) НЕЗАВИСИМО.
        #   -c 4096: достаточно для query(~50) + doc(2048) = 2100 токенов с запасом
        #   -b 2048 / -ub 2048: llama.cpp использует "physical batch size" = -ub
        #   (micro-batch). Реальные замеры:
        #     * 10 чанков × ~80 токенов + query+template = 1025 токенов
        #     * с Query Expansion (3 запроса × 35 чанков) = 1048-1181 токенов
        #   -b 1024 не хватало — 1025 > 1024 → 500 "physical batch size".
        #   -b 2048 даёт 2x запас на крупные vision-описания и QE.
        #   VRAM: +150-250 MB на 0.6B модели (KV-cache scratch × 2).
        #   -ub 32 — pool делится на микро-части, pooling даёт неправильный score (0.0).
        if is_reranker:
            ctx, b_size, ub_size = "4096", "2048", "2048"
        else:
            ctx, b_size, ub_size = "4096", "512", "512"
        cmd.extend(["-c", ctx, "-b", b_size, "-ub", ub_size])
        
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

        print(f"[GGUF Server] Запуск {role}: {os.path.basename(gguf_path)} на порту {port} (parallel={n_parallel})...")
        # F-fix #33: печатаем полную cmd для диагностики (как в LLM-пути)
        print(f"[GGUF Server]   cmd: {' '.join(cmd)}")
        
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


def get_active_embedding_parallel(gguf_path: str = None) -> int:
    """Возвращает n_parallel запущенного embedding-сервера (по умолчанию 1).

    Если сервер не запущен или gguf_path не указан — возвращает 1.
    Используется init_settings для синхронизации embed_batch_size с --parallel.
    """
    with _lock:
        if gguf_path:
            cfg = _server_configs.get(gguf_path)
            if cfg and cfg.get("n_parallel"):
                return int(cfg["n_parallel"])
            return 1
        for path, role in _server_roles.items():
            if role == "embedding":
                cfg = _server_configs.get(path, {})
                return int(cfg.get("n_parallel", 1))
        return 1


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
    except Exception as e:
        # F-fix #silent-except: pgrep/tasklist упал. На Windows это часто
        # бывает если tasklist не в PATH. Возвращаем 0 (best guess) и логируем.
        logger.debug(f"count llama-server processes failed: {e}")
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
            # F-fix #8: на Windows terminate() оставляет orphan CUDA-контекст в драйвере
            # NVIDIA (наблюдаемая утечка ~2-3GB на каждый kill). Решение:
            # 1) сразу kill() вместо terminate()+wait — мягкое завершение не помогает
            # 2) дать процессу явно закрыть свой CUDA-контекст через /slots endpoint ДО kill
            # 3) на Windows дополнительно taskkill /F /T для надёжного убийства child-процессов
            try:
                port = _server_ports.get(path)
                if port:
                    try:
                        import requests as _r
                        _r.post(f"http://127.0.0.1:{port}/slots/0/clear", timeout=0.5)
                    except Exception as slot_err:
                        # F-fix #silent-except: slot clear — best-effort перед
                        # kill. Не критично, но полезно знать если падает.
                        logger.debug(f"slot clear failed: {slot_err}")
            except Exception as kill_err:
                # F-fix #silent-except: процесс уже мёртв или нет прав на taskkill.
                # Не критично для unload_all_models — мы и так пытаемся освободить VRAM.
                logger.debug(f"taskkill failed (process may be dead): {kill_err}")
            try:
                if sys.platform == "win32":
                    # /F — force, /T — вместе с дочерними процессами. Это единственный
                    # надёжный способ освободить CUDA-контекст на Windows.
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                        capture_output=True, timeout=5
                    )
                else:
                    process.kill()
                try: process.wait(timeout=5)
                except Exception: pass
            except Exception as e:
                print(f"[GGUF Server] Ошибка при остановке {os.path.basename(path)}: {e}")
    for path in to_remove:
        _server_processes.pop(path, None)
        _server_ports.pop(path, None)
        _server_configs.pop(path, None)
        _server_roles.pop(path, None)

    # F-fix #8: gc + empty_cache мало помогают на Windows (driver держит контекст
    # в ядре), но мы всё равно вызываем их + небольшой sleep даёт драйверу время
    # на освобождение контекста. Основная защита — taskkill /F /T выше.
    import gc
    import time as _time
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass
    _time.sleep(0.1)

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
