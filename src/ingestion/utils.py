1|"""Утилиты ингеста: безопасный вывод, GPU cleanup, subprocess registry, исключения."""
2|
3|import gc
4|import inspect as _inspect_module
5|import logging
6|import os
7|import sys
8|import threading
9|import warnings
10|
11|import requests
12|import requests.adapters
13|import torch
14|
15|import config
16|
17|logger = logging.getLogger(__name__)
# Фикс inspect.stack() для whisperx, подавление шумных предупреждений библиотек
18|
19|warnings.filterwarnings("ignore", message="Module 'speechbrain")
20|warnings.filterwarnings("ignore", message="torchcodec is not installed")
21|warnings.filterwarnings("ignore", message="TensorFloat-32")
22|warnings.filterwarnings("ignore", message=".*speechbrain.*deprecated", category=UserWarning)
23|warnings.filterwarnings("ignore", message=".*Lightning automatically upgraded.*")
24|
25|logging.getLogger("lightning.pytorch.utilities.migration").setLevel(logging.ERROR)
26|logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
27|logging.getLogger("whisperx").setLevel(logging.WARNING)
28|
29|_orig_getmodule = _inspect_module.getmodule
30|
31|
32|def _safe_getmodule(obj, filename=None):
33|    try:
34|        return _orig_getmodule(obj, filename)
35|    except Exception:
36|        return None
37|
38|
39|_inspect_module.getmodule = _safe_getmodule
40|
41|try:
42|    lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
43|    if os.path.exists(lib_dir):
44|        os.add_dll_directory(lib_dir)
45|except Exception:
46|    pass
47|
48|_http_session = requests.Session()
49|_http_session.mount(
50|    "http://",
51|    requests.adapters.HTTPAdapter(
52|        pool_connections=config.HTTP_POOL_SIZE_INGEST,
53|        pool_maxsize=config.HTTP_POOL_SIZE_INGEST,
54|    ),
55|)
56|_http_session.mount(
57|    "https://",
58|    requests.adapters.HTTPAdapter(
59|        pool_connections=config.HTTP_POOL_SIZE_INGEST,
60|        pool_maxsize=config.HTTP_POOL_SIZE_INGEST,
61|    ),
62|)
63|
64|_active_subprocesses: dict = {}
65|
66|
67|def register_subprocess(notebook_id, popen):
68|    _active_subprocesses.setdefault(notebook_id, []).append(popen)
69|
70|
71|def unregister_subprocess(notebook_id, popen):
72|    lst = _active_subprocesses.get(notebook_id)
73|    if lst and popen in lst:
74|        try:
75|            lst.remove(popen)
76|        except Exception as e:
77|            logger.debug(f"unregister_subprocess: popen already removed: {e}")
78|    if lst is not None and not lst:
79|        _active_subprocesses.pop(notebook_id, None)
80|
81|
82|# Убивает все ffmpeg/subprocess процессы блокнота — используется при cancel
def kill_subprocesses(notebook_id):
83|
84|    procs = _active_subprocesses.pop(notebook_id, [])
85|    for p in procs:
86|        try:
87|            if p.poll() is None:
88|                p.terminate()
89|        except Exception:
90|            pass
91|    return len(procs)
92|
93|
94|class IngestionCancelled(Exception):
95|
96|    pass
97|
98|
99|
100|def _safe_print(msg):
101|
102|    try:
103|        logger.info(msg)
104|    except UnicodeEncodeError:
105|        try:
106|            sys.stdout.buffer.write((str(msg) + "\n").encode("utf-8", errors="replace"))
107|            sys.stdout.buffer.flush()
108|        except Exception:
109|            logger.error(msg.encode("ascii", errors="replace").decode("ascii"))
110|
111|
112|# Очистка видеопамяти перед тяжёлыми задачами (выгрузка RAG-моделей)
def cleanup_gpu():
113|
114|    try:
115|        from src.rag_pipeline import unload_rag_models
116|        unload_rag_models(hard=False)
117|        gc.collect()
118|        if torch.cuda.is_available():
119|            torch.cuda.empty_cache()
120|        logger.info("[GPU] Память полностью очищена для анализа.")
121|    except Exception as e:
122|        logger.error(f"[GPU] Ошибка при очистке: {e}")
123|
124|
125|def format_seconds(s):
126|    h = int(s // 3600)
127|    m = int((s % 3600) // 60)
128|    sec = int(s % 60)
129|    return f"{h}:{m:02d}:{sec:02d}" if h > 0 else f"{m}:{sec:02d}"
130|