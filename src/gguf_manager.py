"""
GGUF Model Manager — управление жизненным циклом нативного llama-server.exe.
Используется для запуска основного сервера модели, выбранной пользователем.
"""

import os
import subprocess
import time
import requests
import json
import socket
from typing import Optional, List, Dict
import config

# Глобальное состояние сервера
_server_process: Optional[subprocess.Popen] = None
_server_info: Dict = {}
SERVER_EXE = os.path.join(config.BASE_DIR, "bin", "llama-server.exe")

def scan_gguf_dirs() -> List[Dict]:
    """Сканирует директории на наличие GGUF и mmproj файлов."""
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

def get_server_status() -> Dict:
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
    mmproj_path: Optional[str] = None,
    ctx_size: Optional[int] = None,
    gpu_layers: Optional[int] = None,
    threads: Optional[int] = None,
) -> Dict:
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
    for _ in range(30):
        time.sleep(1)
        try:
            if requests.get(f"http://127.0.0.1:{port}/health").status_code == 200:
                print(f"[GGUF Manager] Сервер готов на порту {port}")
                return {"status": "ok", "url": f"http://127.0.0.1:{port}/v1", "info": _server_info}
        except Exception: pass
        if _server_process.poll() is not None:
            return {"status": "error", "msg": "Процесс сервера завершился ошибкой"}
            
    return {"status": "error", "msg": "Таймаут запуска сервера"}

def stop_gguf_server() -> Dict:
    global _server_process, _server_info
    if _server_process:
        _server_process.terminate()
        try: _server_process.wait(timeout=5)
        except: _server_process.kill()
    _server_process = None
    _server_info = {}
    return {"status": "ok"}

def get_gguf_server_url() -> Optional[str]:
    status = get_server_status()
    if status["running"]:
        return f"{status['info']['url']}/v1"
    return None
