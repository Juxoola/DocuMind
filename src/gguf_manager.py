"""
GGUF Model Manager — сканирование директорий с GGUF моделями,
управление жизненным циклом llama-cpp-python OpenAI-совместимого сервера.
"""

import os
import glob
import subprocess
import time
import requests
import signal
import sys
from typing import Optional, List, Dict

import config

# ── Глобальное состояние сервера ──
_server_process: Optional[subprocess.Popen] = None
_server_info: Dict = {}  # текущая конфигурация запущенного сервера


def scan_gguf_dirs() -> List[Dict]:
    """
    Сканирует все GGUF_SEARCH_DIRS и возвращает список найденных моделей.
    Группирует файлы по директории: основные .gguf и .mmproj (или .proj) файлы.
    
    Returns:
        [
            {
                "dir": "F:/llm/mradermacher/Huihui-Qwen3.5-4B-abliterated-i1-GGUF",
                "dir_name": "Huihui-Qwen3.5-4B-abliterated-i1-GGUF",
                "gguf_files": ["Huihui-Qwen3.5-4B-abliterated.i1-IQ4_XS.gguf", ...],
                "mmproj_files": ["Huihui-Qwen3.5-4B-abliterated.mmproj-Q8_0.gguf"],
            },
            ...
        ]
    """
    results = []
    search_dirs_raw = config.GGUF_SEARCH_DIRS
    
    # Поддержка разделителя ; для нескольких директорий
    search_dirs = [d.strip() for d in search_dirs_raw.split(";") if d.strip()]
    
    # Конвертация Windows путей для WSL (если мы в WSL)
    def to_native_path(p: str) -> str:
        """Конвертирует путь — если мы в WSL и путь вида X:/..., конвертирует в /mnt/x/..."""
        if sys.platform == "linux" and len(p) >= 2 and p[1] == ':':
            drive = p[0].lower()
            rest = p[2:].replace('\\', '/')
            return f"/mnt/{drive}{rest}"
        return p.replace('\\', '/')
    
    for base_dir in search_dirs:
        native_base = to_native_path(base_dir)
        
        # Ищем все поддиректории, содержащие .gguf файлы
        if not os.path.exists(native_base):
            continue
        
        for dirpath, dirnames, filenames in os.walk(native_base):
            gguf_files = sorted([f for f in filenames if f.lower().endswith('.gguf')
                                 and not f.lower().endswith('.mmproj.gguf')
                                 and not f.lower().endswith('.proj.gguf')])
            mmproj_files = sorted([f for f in filenames if f.lower().endswith('.gguf')
                                   and ('.mmproj' in f.lower() or '.proj' in f.lower())])
            
            if gguf_files or mmproj_files:
                results.append({
                    "dir": dirpath,
                    "dir_name": os.path.basename(dirpath),
                    "gguf_files": gguf_files,
                    "mmproj_files": mmproj_files,
                })
    
    return results


def get_server_status() -> Dict:
    """Проверяет статус llama-cpp-python сервера."""
    global _server_process, _server_info
    
    if _server_process is None:
        return {"running": False, "info": {}}
    
    # Проверяем, жив ли процесс
    poll = _server_process.poll()
    if poll is not None:
        # Процесс завершился
        _server_process = None
        _server_info = {}
        return {"running": False, "info": {}}
    
    # Пингуем сервер
    try:
        url = f"http://{config.GGUF_SERVER_HOST}:{config.GGUF_SERVER_PORT}/v1/models"
        r = requests.get(url, timeout=2)
        if r.ok:
            return {"running": True, "info": _server_info, "models": r.json()}
    except:
        pass
    
    return {"running": True, "info": _server_info, "models": None}


def start_gguf_server(
    gguf_path: str,
    mmproj_path: Optional[str] = None,
    ctx_size: Optional[int] = None,
    gpu_layers: Optional[int] = None,
    threads: Optional[int] = None,
) -> Dict:
    """
    Запускает llama-cpp-python как OpenAI-совместимый сервер.
    
    Args:
        gguf_path: абсолютный путь к .gguf файлу модели
        mmproj_path: абсолютный путь к mmproj .gguf файлу (опционально)
        ctx_size: размер контекста
        gpu_layers: количество GPU слоёв
        threads: количество потоков
    
    Returns:
        {"status": "ok", "url": "..."} или {"status": "error", "msg": "..."}
    """
    global _server_process, _server_info
    
    # Если сервер уже запущен — сначала останавливаем
    if _server_process is not None:
        stop_gguf_server()
        time.sleep(2)
    
    # Проверяем, свободен ли порт
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((config.GGUF_SERVER_HOST, config.GGUF_SERVER_PORT))
        sock.close()
    except OSError:
        sock.close()
        return {"status": "error", "msg": f"Порт {config.GGUF_SERVER_PORT} уже занят. Остановите предыдущий сервер."}
    
    # Нормализуем пути (конвертируем все слэши в нативный формат ОС)
    gguf_path = os.path.normpath(gguf_path)
    if mmproj_path:
        mmproj_path = os.path.normpath(mmproj_path)
    
    print(f"[GGUF] DEBUG: gguf_path = {repr(gguf_path)}")
    print(f"[GGUF] DEBUG: mmproj_path = {repr(mmproj_path)}")
    
    if not gguf_path or not os.path.exists(gguf_path):
        return {"status": "error", "msg": f"Файл модели не найден: {gguf_path}"}
    
    ctx = ctx_size or config.GGUF_CTX_SIZE
    ngl = gpu_layers if gpu_layers is not None else config.GGUF_GPU_LAYERS
    nthreads = threads or config.GGUF_THREADS
    
    cmd = [
        sys.executable, "-m", "llama_cpp.server",
        "--model", gguf_path,
        "--host", config.GGUF_SERVER_HOST,
        "--port", str(config.GGUF_SERVER_PORT),
        "--ctx_size", str(ctx),
        "--n_gpu_layers", str(ngl),
    ]
    
    if nthreads > 0:
        cmd.extend(["--n_threads", str(nthreads)])
    
    if mmproj_path and os.path.exists(mmproj_path):
        cmd.extend(["--mmproj", mmproj_path])
    
    print(f"[GGUF] Запуск сервера: {' '.join(cmd)}")
    
    try:
        _server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError:
        return {"status": "error", "msg": "llama-cpp-python не установлен. Установите: pip install llama-cpp-python[server]"}
    except Exception as e:
        return {"status": "error", "msg": f"Ошибка запуска: {e}"}
    
    _server_info = {
        "gguf_path": gguf_path,
        "mmproj_path": mmproj_path,
        "model_name": os.path.basename(gguf_path),
        "ctx_size": ctx,
        "gpu_layers": ngl,
        "threads": nthreads,
        "url": f"http://{config.GGUF_SERVER_HOST}:{config.GGUF_SERVER_PORT}/v1",
    }
    
    # Ждём, пока сервер поднимется (максимум 60 секунд)
    print("[GGUF] Ожидание запуска сервера...")
    for attempt in range(60):
        time.sleep(1)
        try:
            url = f"http://{config.GGUF_SERVER_HOST}:{config.GGUF_SERVER_PORT}/v1/models"
            r = requests.get(url, timeout=2)
            if r.ok:
                print(f"[GGUF] Сервер запущен за {attempt + 1}с: {_server_info['url']}")
                return {"status": "ok", "url": _server_info["url"], "info": _server_info}
        except:
            pass
        
        # Проверяем, не упал ли процесс
        if _server_process.poll() is not None:
            stdout_data = _server_process.stdout.read() if _server_process.stdout else ""
            stderr_data = _server_process.stderr.read() if _server_process.stderr else ""
            full_log = f"STDOUT:\n{stdout_data}\n\nSTDERR:\n{stderr_data}"
            _server_process = None
            _server_info = {}
            print(f"[GGUF] Полный лог ошибки:\n{full_log}")
            return {"status": "error", "msg": f"Сервер упал при запуске. Лог:\n{full_log[-3000:]}"}
    
    _server_process = None
    _server_info = {}
    return {"status": "error", "msg": "Таймаут ожидания запуска сервера (60с)"}


def stop_gguf_server() -> Dict:
    """Останавливает llama-cpp-python сервер."""
    global _server_process, _server_info
    
    if _server_process is None:
        return {"status": "ok", "msg": "Сервер не запущен"}
    
    try:
        _server_process.terminate()
        try:
            _server_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _server_process.kill()
            _server_process.wait(timeout=5)
    except Exception as e:
        print(f"[GGUF] Ошибка остановки: {e}")
    
    _server_process = None
    _server_info = {}
    print("[GGUF] Сервер остановлен")
    return {"status": "ok", "msg": "Сервер остановлен"}


def get_gguf_server_url() -> Optional[str]:
    """Возвращает URL запущенного GGUF сервера или None."""
    status = get_server_status()
    if status["running"]:
        return status["info"].get("url")
    return None
