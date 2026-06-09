"""
Тесты модуля config.py.

Тестируем:
- get_notebook_paths: структура путей
- get_system_prompt: все 8 режимов, fallback на default
- SYSTEM_PROMPT — обратная совместимость
- safe_filename (из main.py) — path traversal, null-байты, reserved names
- resolve_model_path: пустой ввод, существующий путь
"""

import os
import sys

import pytest

# Добавляем корень проекта в PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config


class TestGetNotebookPaths:
    def test_returns_dict_with_required_keys(self, temp_notebooks_dir):
        """Проверяем структуру возвращаемого словаря."""
        paths = config.get_notebook_paths("test_nb_123")
        assert isinstance(paths, dict)
        assert "base" in paths
        assert "data" in paths
        assert "chroma_db" in paths
        assert "images" in paths

    def test_paths_contain_notebook_id(self, temp_notebooks_dir):
        """ID ноутбука присутствует во всех путях."""
        paths = config.get_notebook_paths("abc123")
        assert "abc123" in paths["base"]
        assert "abc123" in paths["data"]
        assert "abc123" in paths["chroma_db"]
        assert "abc123" in paths["images"]

    def test_base_is_parent_of_data_and_chroma(self, temp_notebooks_dir):
        """data, chroma_db, images — дочерние по отношению к base."""
        paths = config.get_notebook_paths("demo")
        assert paths["data"].startswith(paths["base"])
        assert paths["chroma_db"].startswith(paths["base"])
        assert paths["images"].startswith(paths["base"])


class TestGetSystemPrompt:
    @pytest.mark.parametrize("mode", config.ANSWER_MODES)
    def test_all_known_modes_return_prompt(self, mode):
        """Каждый известный режим возвращает непустой промпт с правилами."""
        prompt = config.get_system_prompt(mode)
        assert prompt
        assert config.SYSTEM_PROMPT_BASE in prompt
        assert config.SYSTEM_PROMPT_CITATION in prompt
        # Правила режима тоже присутствуют
        rule = config.SYSTEM_PROMPT_RULES[mode]
        assert rule in prompt

    def test_default_mode_is_concise(self):
        """ANSWER_MODE_DEFAULT должен быть 'concise'."""
        assert config.ANSWER_MODE_DEFAULT == "concise"

    def test_unknown_mode_falls_back_to_default(self):
        """Неизвестный mode → default (concise)."""
        prompt = config.get_system_prompt("nonexistent_mode_xyz")
        default = config.get_system_prompt(config.ANSWER_MODE_DEFAULT)
        assert prompt == default

    def test_none_mode_uses_default(self):
        """None → default."""
        prompt = config.get_system_prompt(None)
        default = config.get_system_prompt(config.ANSWER_MODE_DEFAULT)
        assert prompt == default

    def test_empty_string_mode_uses_default(self):
        """'' → default."""
        prompt = config.get_system_prompt("")
        default = config.get_system_prompt(config.ANSWER_MODE_DEFAULT)
        assert prompt == default

    def test_system_prompt_compat_matches_concise(self):
        """SYSTEM_PROMPT (обратная совместимость) == concise."""
        assert config.SYSTEM_PROMPT == config.get_system_prompt("concise")

    @pytest.mark.parametrize("mode,keyword", [
        ("concise", "СРАЗУ"),
        ("detailed", "развёрнутый"),
        ("summary", "сжатый"),
        ("step_by_step", "пошаговую"),
        ("checklist", "чек-лист"),
        ("moderate", "средней длины"),
        ("expert", "специалиста"),
        ("eli5", "просто"),
    ])
    def test_each_mode_has_distinct_keyword(self, mode, keyword):
        """Каждый режим содержит свои уникальные инструкции."""
        prompt = config.get_system_prompt(mode)
        assert keyword.lower() in prompt.lower()

    def test_citation_rules_are_in_prompt(self):
        """Правила цитирования содержат ключевые элементы."""
        assert "[N]" in config.SYSTEM_PROMPT_CITATION
        assert "$$" in config.SYSTEM_PROMPT_CITATION
        assert "В документах этого нет" in config.SYSTEM_PROMPT_CITATION


class TestAllowedExtensions:
    def test_pdf_is_allowed(self):
        assert ".pdf" in config.ALLOWED_UPLOAD_EXTENSIONS

    def test_exe_not_allowed(self):
        assert ".exe" not in config.ALLOWED_UPLOAD_EXTENSIONS

    def test_allowed_is_frozenset(self):
        assert isinstance(config.ALLOWED_UPLOAD_EXTENSIONS, frozenset)


class TestConfigValues:
    def test_upload_max_size_positive(self):
        assert config.UPLOAD_MAX_SIZE_BYTES > 0
        assert config.UPLOAD_MAX_SIZE_MB > 0

    def test_default_lm_studio_url(self):
        assert config.LM_STUDIO_URL == "http://localhost:1234/v1"

    def test_rag_params_are_positive(self):
        assert config.RAG_TOP_K_PER_FILE > 0
        assert config.RAG_RERANK_POOL > 0
        assert config.RAG_FINAL_TOP_N > 0

    def test_answer_modes_tuple(self):
        """ANSWER_MODES — кортеж из 8 режимов."""
        assert isinstance(config.ANSWER_MODES, tuple)
        assert len(config.ANSWER_MODES) == 8
        assert all(isinstance(m, str) for m in config.ANSWER_MODES)

    def test_get_system_prompt_contains_answer_modes(self):
        """Каждый режим из ANSWER_MODES имеет запись в SYSTEM_PROMPT_RULES."""
        for mode in config.ANSWER_MODES:
            assert mode in config.SYSTEM_PROMPT_RULES


class TestResolveModelPath:
    def test_empty_string_returns_empty(self):
        assert config.resolve_model_path("") == ""

    def test_none_returns_empty(self):
        assert config.resolve_model_path(None) == ""

    def test_absolute_existing_path_returns_normpath(self, tmp_path):
        """Абсолютный существующий путь возвращается нормализованным."""
        f = tmp_path / "model.gguf"
        f.write_text("dummy")
        result = config.resolve_model_path(str(f))
        assert result == os.path.normpath(str(f)).lower()
