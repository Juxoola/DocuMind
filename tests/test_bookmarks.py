"""
Тесты src/bookmarks.py.

bookmarks.py — чистый CRUD над JSON-файлом, без внешних зависимостей.
Тесты используют временную директорию через conftest.py.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestBookmarks:
    """Полный цикл: create → list → get → update → delete → mark_stale."""

    def _make_payload(self, **overrides):
        """Базовый payload для создания закладки."""
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

    def test_create_bookmark(self, temp_notebooks_dir):
        """Создание закладки возвращает объект с id."""
        from src.bookmarks import create_bookmark

        bm = create_bookmark("test_nb", self._make_payload())
        assert "id" in bm
        assert bm["question"] == "What is RAG?"
        assert bm["status"] == "ok"
        assert bm["tags"] == ["rag", "ai"]

    def test_create_raises_on_missing_fields(self, temp_notebooks_dir):
        """Без question или answer — ValueError."""
        from src.bookmarks import create_bookmark

        with pytest.raises(ValueError, match="question и answer обязательны"):
            create_bookmark("test_nb", {"question": "", "answer": ""})

    def test_list_bookmarks(self, temp_notebooks_dir):
        """После создания закладка появляется в списке."""
        from src.bookmarks import create_bookmark, list_bookmarks

        create_bookmark("test_nb", self._make_payload(question="Q1"))
        create_bookmark("test_nb", self._make_payload(question="Q2"))
        items = list_bookmarks("test_nb")
        assert len(items) == 2
        # Новые сверху (sorted by created_at desc)
        assert items[0]["created_at"] >= items[1]["created_at"]

    def test_list_empty(self, temp_notebooks_dir):
        """Для неизвестного ноутбука — пустой список."""
        from src.bookmarks import list_bookmarks

        assert list_bookmarks("nonexistent") == []

    def test_get_bookmark(self, temp_notebooks_dir):
        """get_bookmark по id возвращает закладку."""
        from src.bookmarks import create_bookmark, get_bookmark

        bm = create_bookmark("test_nb", self._make_payload())
        found = get_bookmark("test_nb", bm["id"])
        assert found is not None
        assert found["id"] == bm["id"]

    def test_get_nonexistent(self, temp_notebooks_dir):
        """Несуществующий id → None."""
        from src.bookmarks import get_bookmark

        assert get_bookmark("test_nb", "fake_id_12345") is None

    def test_update_bookmark(self, temp_notebooks_dir):
        """update_bookmark обновляет title и tags."""
        from src.bookmarks import create_bookmark, update_bookmark

        bm = create_bookmark("test_nb", self._make_payload())
        updated = update_bookmark("test_nb", bm["id"], {"title": "New Title", "tags": ["updated"]})
        assert updated is not None
        assert updated["title"] == "New Title"
        assert updated["tags"] == ["updated"]
        # question/answer не должны меняться
        assert updated["question"] == "What is RAG?"

    def test_update_nonexistent(self, temp_notebooks_dir):
        """Обновление несуществующей → None."""
        from src.bookmarks import update_bookmark

        assert update_bookmark("test_nb", "no_such_id", {"title": "X"}) is None

    def test_delete_bookmark(self, temp_notebooks_dir):
        """Удаление возвращает True, закладка исчезает."""
        from src.bookmarks import create_bookmark, delete_bookmark, get_bookmark

        bm = create_bookmark("test_nb", self._make_payload())
        assert delete_bookmark("test_nb", bm["id"]) is True
        assert get_bookmark("test_nb", bm["id"]) is None

    def test_delete_nonexistent(self, temp_notebooks_dir):
        """Удаление несуществующей → False."""
        from src.bookmarks import delete_bookmark

        assert delete_bookmark("test_nb", "no_such") is False

    def test_mark_stale(self, temp_notebooks_dir):
        """mark_stale_for_file помечает закладки для указанного файла."""
        from src.bookmarks import create_bookmark, list_bookmarks, mark_stale_for_file

        create_bookmark("test_nb", self._make_payload(question="About doc"))
        create_bookmark(
            "test_nb", self._make_payload(question="Other", sources=[{"file_name": "other.pdf"}])
        )

        count = mark_stale_for_file("test_nb", "doc.pdf")
        assert count == 1

        items = list_bookmarks("test_nb")
        doc_bm = next(b for b in items if b["question"] == "About doc")
        other_bm = next(b for b in items if b["question"] == "Other")
        assert doc_bm["status"] == "stale"
        assert other_bm["status"] == "ok"

    def test_mark_stale_idempotent(self, temp_notebooks_dir):
        """Повторный mark_stale не увеличивает счётчик."""
        from src.bookmarks import create_bookmark, mark_stale_for_file

        create_bookmark("test_nb", self._make_payload(sources=[{"file_name": "doc.pdf"}]))
        assert mark_stale_for_file("test_nb", "doc.pdf") == 1
        assert mark_stale_for_file("test_nb", "doc.pdf") == 0

    def test_corrupted_json(self, temp_notebooks_dir):
        """Битый JSON → пустой список (не падает)."""
        from src.bookmarks import list_bookmarks

        nb_path = os.path.join(temp_notebooks_dir, "test_corrupt")
        os.makedirs(nb_path, exist_ok=True)
        with open(os.path.join(nb_path, "bookmarks.json"), "w") as f:
            f.write("{invalid json")
        assert list_bookmarks("test_corrupt") == []

    def test_tags_filtered(self, temp_notebooks_dir):
        """Пустые и не-строковые теги отбрасываются."""
        from src.bookmarks import create_bookmark

        bm = create_bookmark(
            "test_nb", self._make_payload(tags=["valid", "", "  ", None, "also-valid"])
        )
        assert bm["tags"] == ["valid", "also-valid"]
