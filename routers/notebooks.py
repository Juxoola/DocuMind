"""
Роутер: CRUD ноутбуков + миграция старых данных.
"""
import os
import json
import time
import uuid
import gc
import shutil
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import config
from .shared import robust_rmtree

logger = logging.getLogger(__name__)
router = APIRouter(tags=["notebooks"])


# ── Миграция ──

def migrate_old_data():
    """Перенос старых data/chroma_db/images в default-блокнот."""
    try:
        old_data = os.path.join(config.BASE_DIR, "data")
        old_db = os.path.join(config.BASE_DIR, "chroma_db")
        old_imgs = os.path.join(config.BASE_DIR, "images")
        if not (os.path.exists(old_data) or os.path.exists(old_db) or os.path.exists(old_imgs)):
            return
        print("Обнаружены старые данные. Миграция в ноутбук 'default'...")
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
        print(f"[migrate_old_data] Ошибка миграции (продолжаем без неё): {e}")


# ── Эндпоинты ──

@router.get("/api/notebooks")
async def get_notebooks():
    nbs = []
    if os.path.exists(config.NOTEBOOKS_DIR):
        for nb_id in os.listdir(config.NOTEBOOKS_DIR):
            meta_path = os.path.join(config.NOTEBOOKS_DIR, nb_id, "meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    nbs.append(json.load(f))
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


@router.delete("/api/notebooks/{nb_id}")
async def delete_notebook(nb_id: str):
    from src.rag_pipeline import close_all_clients, cancel_bm25_rebuild
    close_all_clients()
    try:
        cancel_bm25_rebuild(nb_id)
    except Exception as e:
        logger.debug(f"[delete_notebook] cancel_bm25_rebuild: {e}")
    gc.collect()
    gc.collect()
    paths = config.get_notebook_paths(nb_id)
    if os.path.exists(paths["base"]):
        success, err_msg = robust_rmtree(paths["base"])
        if not success:
            raise HTTPException(status_code=503, detail=err_msg or "Не удалось удалить ноутбук.")
    return {"status": "ok"}
