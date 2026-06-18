"""Trampoline для обратной совместимости: реэкспорт функций сканирования GGUF."""

from src.gguf.scanner import (
    _dir_mtime,
    _scan_gguf_dirs_uncached,
    find_gguf_by_name,
    invalidate_scan_cache,
    scan_gguf_dirs,
)
from src.gguf.state import _gguf_cache_lock

__all__ = [
    "_dir_mtime",
    "_gguf_cache_lock",
    "_scan_gguf_dirs_uncached",
    "find_gguf_by_name",
    "invalidate_scan_cache",
    "scan_gguf_dirs",
]
