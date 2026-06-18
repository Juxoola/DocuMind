"""Хранилище закладок (bookmarks.json) для вопросов/ответов по блокноту."""

import logging
import os
import threading
import time
import uuid
from collections import OrderedDict

import orjson

import config

logger = logging.getLogger(__name__)
_WRITE_LOCKS_MAXSIZE = 50
_write_locks: OrderedDict = OrderedDict()
_locks_guard = threading.Lock()


def _lock_for(notebook_id: str) -> threading.Lock:
    with _locks_guard:
        lock = _write_locks.get(notebook_id)
        if lock is None:
            lock = threading.Lock()
            if len(_write_locks) >= _WRITE_LOCKS_MAXSIZE:
                _write_locks.popitem(last=False)
            _write_locks[notebook_id] = lock
        else:
            _write_locks.move_to_end(notebook_id)
        return lock


def _bookmarks_path(notebook_id: str) -> str:
    return os.path.join(config.get_notebook_paths(notebook_id)["base"], "bookmarks.json")


def _read_bookmarks(notebook_id: str) -> list:
    path = _bookmarks_path(notebook_id)
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data = orjson.loads(f.read())
        if not isinstance(data, list):
            logger.warning(f"[BOOKMARKS] Неверный формат в {path}, ожидался list — обнуляю")
            return []
        return data
    except (orjson.JSONDecodeError, OSError) as e:
        logger.warning(f"[BOOKMARKS] Не удалось прочитать {path}: {e}")
        return []


def _write_bookmarks(notebook_id: str, bookmarks: list) -> None:
    path = _bookmarks_path(notebook_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(orjson.dumps(bookmarks, option=orjson.OPT_INDENT_2).decode())
    os.replace(tmp, path)


def list_bookmarks(notebook_id: str) -> list:
    with _lock_for(notebook_id):
        items = _read_bookmarks(notebook_id)
    return sorted(items, key=lambda b: b.get("created_at", 0), reverse=True)


def get_bookmark(notebook_id: str, bookmark_id: str) -> dict | None:
    with _lock_for(notebook_id):
        for b in _read_bookmarks(notebook_id):
            if b.get("id") == bookmark_id:
                return b
    return None


def create_bookmark(notebook_id: str, payload: dict) -> dict:
    with _lock_for(notebook_id):
        items = _read_bookmarks(notebook_id)
        bm = {
            "id": uuid.uuid4().hex[:12],
            "created_at": time.time(),
            "question": (payload.get("question") or "").strip(),
            "answer": (payload.get("answer") or "").strip(),
            "sources": payload.get("sources") or [],
            "model": payload.get("model") or "",
            "answer_mode": payload.get("answer_mode") or "concise",
            "thinking_mode": bool(payload.get("thinking_mode", False)),
            "title": (payload.get("title") or "").strip(),
            "tags": [
                t.strip() for t in (payload.get("tags") or []) if isinstance(t, str) and t.strip()
            ],
            "status": "ok",
        }
        if not bm["question"] or not bm["answer"]:
            raise ValueError("question и answer обязательны")
        items.append(bm)
        _write_bookmarks(notebook_id, items)
    return bm


def update_bookmark(notebook_id: str, bookmark_id: str, patch: dict) -> dict | None:
    with _lock_for(notebook_id):
        items = _read_bookmarks(notebook_id)
        for i, b in enumerate(items):
            if b.get("id") == bookmark_id:
                if "title" in patch:
                    b["title"] = (patch.get("title") or "").strip()
                if "tags" in patch:
                    b["tags"] = [
                        t.strip()
                        for t in (patch.get("tags") or [])
                        if isinstance(t, str) and t.strip()
                    ]
                items[i] = b
                _write_bookmarks(notebook_id, items)
                return b
    return None


def delete_bookmark(notebook_id: str, bookmark_id: str) -> bool:
    with _lock_for(notebook_id):
        items = _read_bookmarks(notebook_id)
        new_items = [b for b in items if b.get("id") != bookmark_id]
        if len(new_items) == len(items):
            return False
        _write_bookmarks(notebook_id, new_items)
    return True


def mark_stale_for_file(notebook_id: str, file_name: str) -> int:
    with _lock_for(notebook_id):
        items = _read_bookmarks(notebook_id)
        updated = 0
        for b in items:
            if b.get("status") == "stale":
                continue
            sources = b.get("sources") or []
            if any(s.get("file_name") == file_name for s in sources):
                b["status"] = "stale"
                updated += 1
        if updated:
            _write_bookmarks(notebook_id, items)
    return updated
