"""
Роутер: CRUD закладок (Q&A).
"""
import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from src.bookmarks import (
    list_bookmarks, get_bookmark, create_bookmark,
    update_bookmark, delete_bookmark, mark_stale_for_file,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["bookmarks"])


@router.get("/api/bookmarks")
async def api_list_bookmarks(notebook_id: str = Query(...)):
    return {"bookmarks": list_bookmarks(notebook_id)}


@router.get("/api/bookmarks/{bookmark_id}")
async def api_get_bookmark(bookmark_id: str, notebook_id: str = Query(...)):
    bm = get_bookmark(notebook_id, bookmark_id)
    if bm is None:
        raise HTTPException(status_code=404, detail="Закладка не найдена")
    return bm


class CreateBookmarkRequest(BaseModel):
    notebook_id: str
    question: str
    answer: str
    sources: List[dict] = []
    model: Optional[str] = ""
    answer_mode: Optional[str] = "concise"
    thinking_mode: Optional[bool] = False
    title: Optional[str] = ""
    tags: List[str] = []


@router.post("/api/bookmarks")
async def api_create_bookmark(req: CreateBookmarkRequest):
    try:
        return create_bookmark(req.notebook_id, req.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


class UpdateBookmarkRequest(BaseModel):
    notebook_id: str
    title: Optional[str] = None
    tags: Optional[List[str]] = None


@router.patch("/api/bookmarks/{bookmark_id}")
async def api_update_bookmark(bookmark_id: str, req: UpdateBookmarkRequest):
    patch = {k: v for k, v in req.model_dump().items() if k != "notebook_id" and v is not None}
    bm = update_bookmark(req.notebook_id, bookmark_id, patch)
    if bm is None:
        raise HTTPException(status_code=404, detail="Закладка не найдена")
    return bm


@router.delete("/api/bookmarks/{bookmark_id}")
async def api_delete_bookmark(bookmark_id: str, notebook_id: str = Query(...)):
    ok = delete_bookmark(notebook_id, bookmark_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Закладка не найдена")
    return {"status": "ok"}
