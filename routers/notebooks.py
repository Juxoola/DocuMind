"""
Роутер: CRUD ноутбуков + миграция старых данных.
"""
#
# Файл: notebooks.py — создание, удаление и список блокнотов (notebooks),
# а также миграция данных из старой плоской структуры в именованные блокноты.
#

import gc
import json
import logging
import os
import re
import time
import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import config

from .shared import robust_rmtree

logger = logging.getLogger(__name__)
router = APIRouter(tags=["notebooks"])

_NB_ID_PATTERN = re.compile(r"^[a-f0-9]{8}$")


# Валидация notebook_id: строго 8 hex-символов — защита от path traversal.
def validate_nb_id(nb_id: str) -> str:
    nb_id = nb_id.strip()
    if not _NB_ID_PATTERN.match(nb_id):
        logger.warning(f"Некорректный notebook_id: {nb_id!r}")
        raise HTTPException(status_code=400, detail="Некорректный ID блокнота")
    return nb_id


# Миграция старых данных (data/, chroma_db/, images/) в блокнот 'default'.
def migrate_old_data():
    try:
        old_data = os.path.join(config.BASE_DIR, "data")
        old_db = os.path.join(config.BASE_DIR, "chroma_db")
        old_imgs = os.path.join(config.BASE_DIR, "images")
        if not (os.path.exists(old_data) or os.path.exists(old_db) or os.path.exists(old_imgs)):
            return
        logger.info("Обнаружены старые данные. Миграция в ноутбук 'default'...")
        import shutil

        paths = config.get_notebook_paths("default")
        os.makedirs(paths["base"], exist_ok=True)
        if os.path.exists(old_data):
            shutil.move(old_data, paths["data"])
        if os.path.exists(old_db):
            shutil.move(old_db, paths["chroma_db"])
        if os.path.exists(old_imgs):
            shutil.move(old_imgs, paths["images"])
        with open(os.path.join(paths["base"], "meta.json"), "w", encoding="utf-8") as f:
            json.dump({"id": "default", "name": "Мой первый блокнот", "created_at": time.time()}, f)
    except Exception as e:
        logger.warning(f"Ошибка миграции (продолжаем без неё): {e}")


# Список всех блокнотов — читает meta.json из каждой поддиректории notebooks/.
@router.get("/api/notebooks")
async def get_notebooks():
    nbs = []
    if os.path.exists(config.NOTEBOOKS_DIR):
        for entry in os.listdir(config.NOTEBOOKS_DIR):
            if entry.startswith(".") or not _NB_ID_PATTERN.match(entry):
                continue
            meta_path = os.path.join(config.NOTEBOOKS_DIR, entry, "meta.json")
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, encoding="utf-8") as f:
                        nbs.append(json.load(f))
                except Exception as e:
                    logger.debug(f"Не удалось прочитать {meta_path}: {e}")
    return nbs


class CreateNotebookRequest(BaseModel):
    name: str


@router.post("/api/notebooks")
async def create_notebook(req: CreateNotebookRequest):
    nb_id = str(uuid.uuid4())[:8]
    paths = config.get_notebook_paths(nb_id)
    os.makedirs(paths["data"], exist_ok=True)
    os.makedirs(paths["chroma_db"], exist_ok=True)
    os.makedirs(paths["images"], exist_ok=True)
    meta = {"id": nb_id, "name": req.name, "created_at": time.time()}
    with open(os.path.join(paths["base"], "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    return meta


# Удаление блокнота: закрывает ChromaDB-клиент, отменяет BM25-перестроение, удаляет директорию.
@router.delete("/api/notebooks/{nb_id}")
async def delete_notebook(nb_id: str):
    nb_id = validate_nb_id(nb_id)
    paths = config.get_notebook_paths(nb_id)
    base_path = paths["base"]

    if not os.path.exists(base_path):
        raise HTTPException(status_code=404, detail="Блокнот не найден")

    try:
        from src.rag_pipeline import close_notebook_client

        close_notebook_client(nb_id)
    except Exception as e:
        logger.debug(f"[delete_notebook] close_notebook_client: {e}")

    try:
        from src.rag_pipeline import cancel_bm25_rebuild

        cancel_bm25_rebuild(nb_id)
    except Exception as e:
        logger.debug(f"[delete_notebook] cancel_bm25_rebuild: {e}")

    gc.collect()
    gc.collect()

    success, err_msg = robust_rmtree(base_path)
    if not success:
        logger.error(f"[delete_notebook] Не удалось удалить {base_path}: {err_msg}")
        raise HTTPException(
            status_code=503,
            detail=err_msg or "Не удалось удалить блокнот. Попробуйте позже.",
        )

    logger.info(f"Блокнот {nb_id} удалён.")
    return {"status": "ok"}
