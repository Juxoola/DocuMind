"""Роутер: CRUD закладок (Q&A)."""

import logging

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from routers.notebooks import validate_nb_id
from src.bookmarks import (
    create_bookmark,
    delete_bookmark,
    get_bookmark,
    list_bookmarks,
    update_bookmark,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["bookmarks"])


# ── Получение закладок ──
@router.get("/api/bookmarks")
async def api_list_bookmarks(notebook_id: str = Query(...)):

    notebook_id = validate_nb_id(notebook_id)
    return {"bookmarks": await list_bookmarks(notebook_id)}


@router.get("/api/bookmarks/{bookmark_id}")
async def api_get_bookmark(bookmark_id: str, notebook_id: str = Query(...)):

    notebook_id = validate_nb_id(notebook_id)
    bm = await get_bookmark(notebook_id, bookmark_id)
    if bm is None:
        raise HTTPException(status_code=404, detail="Закладка не найдена")
    return bm


class CreateBookmarkRequest(BaseModel):
    notebook_id: str
    question: str
    answer: str
    sources: list[dict] = []
    model: str | None = ""
    answer_mode: str | None = "concise"
    thinking_mode: bool | None = False
    title: str | None = ""
    tags: list[str] = []


# ── Создание и обновление закладок ──
@router.post("/api/bookmarks")
async def api_create_bookmark(req: CreateBookmarkRequest):

    req.notebook_id = validate_nb_id(req.notebook_id)
    try:
        return await create_bookmark(req.notebook_id, req.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


class UpdateBookmarkRequest(BaseModel):
    notebook_id: str
    title: str | None = None
    tags: list[str] | None = None


@router.patch("/api/bookmarks/{bookmark_id}")
async def api_update_bookmark(bookmark_id: str, req: UpdateBookmarkRequest):

    req.notebook_id = validate_nb_id(req.notebook_id)
    patch = {k: v for k, v in req.model_dump().items() if k != "notebook_id" and v is not None}
    bm = await update_bookmark(req.notebook_id, bookmark_id, patch)
    if bm is None:
        raise HTTPException(status_code=404, detail="Закладка не найдена")
    return bm


# ── Удаление закладки ──
@router.delete("/api/bookmarks/{bookmark_id}")
async def api_delete_bookmark(bookmark_id: str, notebook_id: str = Query(...)):

    notebook_id = validate_nb_id(notebook_id)
    ok = await delete_bookmark(notebook_id, bookmark_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Закладка не найдена")
    return {"status": "ok"}
