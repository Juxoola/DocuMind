"""Тесты модуля bookmarks: CRUD-операции над JSON-файлом закладок."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestBookmarks:
    def _make_payload(self, **overrides):
        return {
            "notebook_id": "test_nb",
            "question": "What is RAG?",
            "answer": "Retrieval Augmented Generation is...",
            "sources": [{"file_name": "doc.pdf", "page": 5, "snippet": "RAG is..."}],
            "model": "test-model",
            "answer_mode": "concise",
            "thinking_mode": False,
            "title": "",
            "tags": ["rag", "ai"],
            **overrides,
        }

    # ── Создание закладок ──
    @pytest.mark.asyncio
    async def test_create_bookmark(self, temp_notebooks_dir):
        from src.bookmarks import create_bookmark

        bm = await create_bookmark("test_nb", self._make_payload())
        assert "id" in bm
        assert bm["question"] == "What is RAG?"
        assert bm["status"] == "ok"
        assert bm["tags"] == ["rag", "ai"]

    @pytest.mark.asyncio
    async def test_create_raises_on_missing_fields(self, temp_notebooks_dir):
        from src.bookmarks import create_bookmark

        with pytest.raises(ValueError, match="question и answer обязательны"):
            await create_bookmark("test_nb", {"question": "", "answer": ""})

    # ── Чтение и список закладок ──
    @pytest.mark.asyncio
    async def test_list_bookmarks(self, temp_notebooks_dir):
        from src.bookmarks import create_bookmark, list_bookmarks

        await create_bookmark("test_nb", self._make_payload(question="Q1"))
        await create_bookmark("test_nb", self._make_payload(question="Q2"))
        items = await list_bookmarks("test_nb")
        assert len(items) == 2
        assert items[0]["created_at"] >= items[1]["created_at"]

    @pytest.mark.asyncio
    async def test_list_empty(self, temp_notebooks_dir):
        from src.bookmarks import list_bookmarks

        assert await list_bookmarks("nonexistent") == []

    @pytest.mark.asyncio
    async def test_get_bookmark(self, temp_notebooks_dir):
        from src.bookmarks import create_bookmark, get_bookmark

        bm = await create_bookmark("test_nb", self._make_payload())
        found = await get_bookmark("test_nb", bm["id"])
        assert found is not None
        assert found["id"] == bm["id"]

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, temp_notebooks_dir):
        from src.bookmarks import get_bookmark

        assert await get_bookmark("test_nb", "fake_id_12345") is None

    # ── Обновление закладок ──
    @pytest.mark.asyncio
    async def test_update_bookmark(self, temp_notebooks_dir):
        from src.bookmarks import create_bookmark, update_bookmark

        bm = await create_bookmark("test_nb", self._make_payload())
        updated = await update_bookmark(
            "test_nb", bm["id"], {"title": "New Title", "tags": ["updated"]}
        )
        assert updated is not None
        assert updated["title"] == "New Title"
        assert updated["tags"] == ["updated"]
        assert updated["question"] == "What is RAG?"

    @pytest.mark.asyncio
    async def test_update_nonexistent(self, temp_notebooks_dir):
        from src.bookmarks import update_bookmark

        assert await update_bookmark("test_nb", "no_such_id", {"title": "X"}) is None

    # ── Удаление закладок ──
    @pytest.mark.asyncio
    async def test_delete_bookmark(self, temp_notebooks_dir):
        from src.bookmarks import create_bookmark, delete_bookmark, get_bookmark

        bm = await create_bookmark("test_nb", self._make_payload())
        assert await delete_bookmark("test_nb", bm["id"]) is True
        assert await get_bookmark("test_nb", bm["id"]) is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, temp_notebooks_dir):
        from src.bookmarks import delete_bookmark

        assert await delete_bookmark("test_nb", "no_such") is False

    @pytest.mark.asyncio
    async def test_mark_stale(self, temp_notebooks_dir):
        from src.bookmarks import create_bookmark, list_bookmarks, mark_stale_for_file

        await create_bookmark("test_nb", self._make_payload(question="About doc"))
        await create_bookmark(
            "test_nb", self._make_payload(question="Other", sources=[{"file_name": "other.pdf"}])
        )

        count = await mark_stale_for_file("test_nb", "doc.pdf")
        assert count == 1

        items = await list_bookmarks("test_nb")
        doc_bm = next(b for b in items if b["question"] == "About doc")
        other_bm = next(b for b in items if b["question"] == "Other")
        assert doc_bm["status"] == "stale"
        assert other_bm["status"] == "ok"

    @pytest.mark.asyncio
    async def test_mark_stale_idempotent(self, temp_notebooks_dir):
        from src.bookmarks import create_bookmark, mark_stale_for_file

        await create_bookmark("test_nb", self._make_payload(sources=[{"file_name": "doc.pdf"}]))
        assert await mark_stale_for_file("test_nb", "doc.pdf") == 1
        assert await mark_stale_for_file("test_nb", "doc.pdf") == 0

    # ── Edge cases: повреждённый JSON, фильтрация тегов ──
    @pytest.mark.asyncio
    async def test_corrupted_json(self, temp_notebooks_dir):
        from src.bookmarks import list_bookmarks

        nb_path = os.path.join(temp_notebooks_dir, "test_corrupt")
        os.makedirs(nb_path, exist_ok=True)
        with open(os.path.join(nb_path, "bookmarks.json"), "w") as f:
            f.write("{invalid json")
        assert await list_bookmarks("test_corrupt") == []

    @pytest.mark.asyncio
    async def test_tags_filtered(self, temp_notebooks_dir):
        from src.bookmarks import create_bookmark

        bm = await create_bookmark(
            "test_nb", self._make_payload(tags=["valid", "", "  ", None, "also-valid"])
        )
        assert bm["tags"] == ["valid", "also-valid"]
