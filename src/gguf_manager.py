"""
GGUF Model Manager — управление жизненным циклом нативного llama-server.exe.
Используется для запуска основного сервера модели, выбранной пользователем.
"""

import json
import os
import socket
import subprocess
import threading
import time

import requests

import config

# Глобальное состояние сервера
_server_process: subprocess.Popen | None = None
_server_info: dict = {}
SERVER_EXE = os.path.join(config.BASE_DIR, "bin", "llama-server.exe")

# ── Persistent scan cache (mtime-keyed) ─────────────────────────────────
# os.walk по F:\llm с тысячами файлов на каждый resolve_model_path = секунды задержки.
# Кеш валидируется по mtime корневой директории + TTL 5 мин (на случай file-add в subdir).
_GGUF_CACHE_FILE = os.path.join(config.BASE_DIR, "_gguf_scan_cache.json")
_GGUF_CACHE_TTL_SEC = 300.0
_gguf_cache_lock = threading.Lock()


def _dir_mtime(root: str) -> float:
    """mtime директории + бонус за file-count (любое добавление/удаление меняет count)."""
    try:
        st = os.stat(root)
        return st.st_mtime
    except OSError:
        return 0.0


def _scan_gguf_dirs_uncached() -> list[dict]:
    """Полный os.walk без кеша. Дорогая операция — вызывать редко."""
    results = []
    search_dirs = [d.strip() for d in config.GGUF_SEARCH_DIRS.split(";") if d.strip()]

    for base_dir in search_dirs:
        if not os.path.exists(base_dir): continue
        for dirpath, dirnames, filenames in os.walk(base_dir):
            gguf_files = sorted([f for f in filenames if f.lower().endswith('.gguf')
                                 and not any(x in f.lower() for x in ['.mmproj', '.proj'])])
            mmproj_files = sorted([f for f in filenames if f.lower().endswith('.gguf')
                                   and any(x in f.lower() for x in ['.mmproj', '.proj'])])
            if gguf_files or mmproj_files:
                results.append({
                    "dir": dirpath,
                    "dir_name": os.path.basename(dirpath),
                    "gguf_files": gguf_files,
                    "mmproj_files": mmproj_files,
                })
    return results


def scan_gguf_dirs() -> list[dict]:
    """Сканирует директории на наличие GGUF и mmproj файлов.
    Использует persistent mtime-keyed cache — повторные вызовы при неизменных
    директориях практически бесплатны."""
    with _gguf_cache_lock:
        # Пробуем достать кеш
        cached = None
        try:
            if os.path.exists(_GGUF_CACHE_FILE):
                with open(_GGUF_CACHE_FILE, encoding="utf-8") as f:
                    cached = json.load(f)
        except Exception:
            cached = None

        # Валидация: проверяем mtime + TTL
        if cached:
            try:
                saved_at = float(cached.get("saved_at", 0))
                age = time.time() - saved_at
                cached_mtimes = cached.get("dir_mtimes", {}) or {}
                roots = [d.strip() for d in config.GGUF_SEARCH_DIRS.split(";") if d.strip()]
                roots_valid = all(
                    cached_mtimes.get(r) == _dir_mtime(r) for r in roots
                ) and len(cached_mtimes) == len(roots)
                if age < _GGUF_CACHE_TTL_SEC and roots_valid:
                    return cached.get("results", [])
            except Exception:
                pass

        # Cold path: полный os.walk + persist
        results = _scan_gguf_dirs_uncached()
        try:
            roots = [d.strip() for d in config.GGUF_SEARCH_DIRS.split(";") if d.strip()]
            payload = {
                "saved_at": time.time(),
                "dir_mtimes": {r: _dir_mtime(r) for r in roots},
                "results": results,
            }
            tmp = _GGUF_CACHE_FILE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            os.replace(tmp, _GGUF_CACHE_FILE)
        except Exception as e:
            print(f"[GGUF Manager] WARN: не удалось сохранить scan cache: {e}")
        return results


def invalidate_scan_cache():
    """Сбросить кеш (например, после добавления новой GGUF в поисковую директорию)."""
    with _gguf_cache_lock:
        try:
            if os.path.exists(_GGUF_CACHE_FILE):
                os.remove(_GGUF_CACHE_FILE)
        except Exception:
            pass


def find_gguf_by_name(filename: str) -> str | None:
    """Быстрый поиск файла по имени в GGUF_SEARCH_DIRS. Использует scan cache.
    Возвращает полный путь или None.

    Отличие от config.resolve_model_path: не делает fresh os.walk, всегда читает
    кешированный scan. config.resolve_model_path вызывает нас как fallback.
    """
    if not filename:
        return None
    name = os.path.basename(filename)
    for entry in scan_gguf_dirs():
        for f in (entry.get("gguf_files") or []) + (entry.get("mmproj_files") or []):
            if f == name:
                return os.path.join(entry["dir"], name)
    return None

def get_server_status() -> dict:
    """Проверяет статус запущенного сервера."""
    global _server_process, _server_info
    if _server_process is None or _server_process.poll() is not None:
        return {"running": False, "info": {}}

    try:
        url = f"http://127.0.0.1:{_server_info['port']}/health"
        r = requests.get(url, timeout=1)
        if r.status_code == 200:
            return {"running": True, "info": _server_info}
    except Exception: pass
    return {"running": True, "info": _server_info, "status": "initializing"}

def start_gguf_server(
    gguf_path: str,
    mmproj_path: str | None = None,
    ctx_size: int | None = None,
    gpu_layers: int | None = None,
    threads: int | None = None,
    mtp_enabled: bool = False,
) -> dict:
    """Запускает нативный llama-server.exe."""
    global _server_process, _server_info

    if _server_process: stop_gguf_server()

    # Поиск свободного порта
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()

    cmd = [
        SERVER_EXE,
        "-m", os.path.normpath(gguf_path),
        "--port", str(port),
        "-c", str(ctx_size or config.GGUF_CTX_SIZE),
        "-ngl", str(gpu_layers if gpu_layers is not None else config.GGUF_GPU_LAYERS),
        "-b", "512",
        "-ub", "256",
        "--parallel", "1",
        "--no-context-shift",
        "--jinja",
        "-n", "2048",
        "--flash-attn", "on"
    ]
    if mtp_enabled:
        cmd.extend(["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"])
    if mmproj_path:
        cmd.extend(["--mmproj", os.path.normpath(mmproj_path)])

    print(f"[GGUF Manager] Запуск сервера: {' '.join(cmd)}")

    # CREATE_NO_WINDOW
    _server_process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        creationflags=0x08000000
    )

    _server_info = {
        "port": port,
        "url": f"http://127.0.0.1:{port}",
        "model": os.path.basename(gguf_path)
    }

    # Ждем готовности
    for _ in range(config.GGUF_SERVER_STARTUP_TIMEOUT):
        time.sleep(1)
        try:
            # F-fix #21: requests.get без timeout = бесконечное ожидание.
            # Если сервер упал, connection hung → опросник висит 30 сек.
            if requests.get(f"http://127.0.0.1:{port}/health", timeout=config.GGUF_HEALTH_CHECK_TIMEOUT).status_code == 200:
                print(f"[GGUF Manager] Сервер готов на порту {port}")
                return {"status": "ok", "url": f"http://127.0.0.1:{port}/v1", "info": _server_info}
        except Exception: pass
        if _server_process.poll() is not None:
            return {"status": "error", "msg": "Процесс сервера завершился ошибкой"}

    return {"status": "error", "msg": "Таймаут запуска сервера"}

def stop_gguf_server() -> dict:
    global _server_process, _server_info
    if _server_process:
        _server_process.terminate()
        try:
            _server_process.wait(timeout=config.GGUF_SERVER_STOP_TIMEOUT)
        except Exception:
            # F-fix #17: bare except ловит BaseException (включая KeyboardInterrupt
            # и SystemExit), что маскирует Ctrl+C от пользователя. Ловим только
            # реальные ошибки ожидания процесса.
            try: _server_process.kill()
            except Exception: pass
    _server_process = None
    _server_info = {}
    return {"status": "ok"}

def get_gguf_server_url() -> str | None:
    status = get_server_status()
    if status["running"]:
        return f"{status['info']['url']}/v1"
    return None
