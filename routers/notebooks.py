"""Роутер: CRUD ноутбуков + миграция старых данных."""

import asyncio
import json
import logging
import os
import re
import time
import uuid

import aiofiles
import aiofiles.os
import orjson
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import config

from .shared import robust_rmtree

logger = logging.getLogger(__name__)
router = APIRouter(tags=["notebooks"])

_NB_ID_PATTERN = re.compile(r"^[a-f0-9]{8}$")


def validate_nb_id(nb_id: str) -> str:
    nb_id = nb_id.strip()
    if not _NB_ID_PATTERN.match(nb_id):
        logger.warning(f"Некорректный notebook_id: {nb_id!r}")
        raise HTTPException(status_code=400, detail="Некорректный ID блокнота")
    return nb_id


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


@router.get("/api/notebooks")
async def get_notebooks():
    nbs = []
    if os.path.exists(config.NOTEBOOKS_DIR):
        for entry in await aiofiles.os.listdir(config.NOTEBOOKS_DIR):
            if entry.startswith(".") or not _NB_ID_PATTERN.match(entry):
                continue
            meta_path = os.path.join(config.NOTEBOOKS_DIR, entry, "meta.json")
            if os.path.exists(meta_path):
                try:
                    async with aiofiles.open(meta_path, encoding="utf-8") as f:
                        nbs.append(orjson.loads(await f.read()))
                except Exception as e:
                    logger.debug(f"Не удалось прочитать {meta_path}: {e}")
    return nbs


class CreateNotebookRequest(BaseModel):
    name: str


@router.post("/api/notebooks")
async def create_notebook(req: CreateNotebookRequest):
    nb_id = str(uuid.uuid4())[:8]
    paths = config.get_notebook_paths(nb_id)
    await aiofiles.os.makedirs(paths["data"], exist_ok=True)
    await aiofiles.os.makedirs(paths["chroma_db"], exist_ok=True)
    await aiofiles.os.makedirs(paths["images"], exist_ok=True)
    meta = {"id": nb_id, "name": req.name, "created_at": time.time()}
    async with aiofiles.open(os.path.join(paths["base"], "meta.json"), "wb") as f:
        await f.write(orjson.dumps(meta))
    return meta


@router.delete("/api/notebooks/{nb_id}")
async def delete_notebook(nb_id: str):
    nb_id = validate_nb_id(nb_id)
    paths = config.get_notebook_paths(nb_id)
    base_path = paths["base"]

    if not os.path.exists(base_path):
        raise HTTPException(status_code=404, detail="Блокнот не найден")

    try:
        from src.rag.indexing import close_notebook_client

        close_notebook_client(nb_id)
    except Exception as e:
        logger.debug(f"[delete_notebook] close_notebook_client: {e}")

    try:
        from src.rag.bm25 import cancel_bm25_rebuild

        cancel_bm25_rebuild(nb_id)
    except Exception as e:
        logger.debug(f"[delete_notebook] cancel_bm25_rebuild: {e}")

    def _do_delete(path: str) -> None:
        success, err_msg = robust_rmtree(path)
        if not success:
            raise RuntimeError(err_msg or f"Не удалось удалить {path}")

    try:
        await asyncio.to_thread(_do_delete, base_path)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    logger.info(f"Блокнот {nb_id} удалён.")
    return {"status": "ok"}
