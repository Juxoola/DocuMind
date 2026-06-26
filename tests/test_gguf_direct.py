"""Тесты src/gguf/ — модели, состояние, сервер.

Тестируем чистые функции без запуска серверов:
- detect_model_family: определение семейства по имени файла
- CACHE_TYPE_MAP: корректность KV-cache типов
- _llm_load_state: структура состояния
- is_server_ready: (мокаем requests)

Тяжёлые зависимости (subprocess, requests) замоканы.
"""

import asyncio
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestDetectModelFamily:
    """Определение семейства модели по имени файла."""

    def _detect(self, name):
        """Helper — вызываем detect_model_family после импорта."""
        from src.gguf.models import detect_model_family

        return detect_model_family(name)

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("qwen2.5-7b.Q4_K_M.gguf", "qwen"),
            ("qwq-32b-q4_k_m.gguf", "qwen"),
            ("Qwen3-14B-Q8_0.gguf", "qwen"),
            ("QwQ-32B-Preview-Q4_K_M.gguf", "qwen"),
        ],
    )
    def test_qwen_family(self, name, expected):
        """Модели Qwen/QwQ → 'qwen'."""
        assert self._detect(name) == expected

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("gemma-3-12b-it-Q4_K_M.gguf", "gemma3"),
            ("gemma3-27b-Q8_0.gguf", "gemma3"),
            ("gemma-4-9b-it-Q4_K_M.gguf", "gemma4"),
            ("Gemma-4-27B-Q8_0.gguf", "gemma4"),
            ("gemma_4_9b.gguf", "gemma4"),
            ("gemma4-9b-Q4_K_M.gguf", "gemma4"),
        ],
    )
    def test_gemma_family(self, name, expected):
        """Определение семейства Gemma: Gemma 3 → 'gemma3', Gemma 4 → 'gemma4'."""
        assert self._detect(name) == expected

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("deepseek-v3-Q4_K_M.gguf", "deepseek"),
            ("DeepSeek-R1-UD-Q5_K_M.gguf", "deepseek"),
            ("deepseek_r1_7b.gguf", "deepseek"),
        ],
    )
    def test_deepseek_family(self, name, expected):
        """Определение семейства DeepSeek → 'deepseek'."""
        assert self._detect(name) == expected

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("llama-3.2-3b.Q4_K_M.gguf", "llama"),
            ("Llama-3.1-8B-Q8_0.gguf", "llama"),
        ],
    )
    def test_llama_family(self, name, expected):
        """Определение семейства LLaMA → 'llama'."""
        assert self._detect(name) == expected

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("Mistral-7B-v0.3-Q4_K_M.gguf", "generic"),
            ("", "generic"),
            ("some_random_model.gguf", "generic"),
        ],
    )
    def test_generic_family(self, name, expected):
        """Всё остальное → 'generic'."""
        assert self._detect(name) == expected

    def test_case_insensitive(self):
        """Имена проверяются без учёта регистра."""
        from src.gguf.models import detect_model_family

        assert detect_model_family("QWEN2-7B.Q4_K_M.gguf") == "qwen"
        assert detect_model_family("DEEPSEEK-R1.Q4_K_M.gguf") == "deepseek"
        assert detect_model_family("GEMMA-3-12B.Q4_K_M.gguf") == "gemma3"

    def test_gemma4_variants(self):
        """Проверка всех вариантов детекции Gemma-4."""
        from src.gguf.models import detect_model_family

        for name in [
            "gemma-4-9b.gguf",
            "gemma4-9b.gguf",
            "gemma_4-9b.gguf",
            "gemma-4b-Q4_K_M.gguf",
            "gemma-4e-Q8_0.gguf",
            "gemma_e4b.gguf",
        ]:
            assert detect_model_family(name) == "gemma4", f"Failed for {name}"


class TestCacheTypeMap:
    """Карта типов KV-кэша — валидность значений для llama-server."""

    def test_all_values_are_valid(self):
        """Все значения в CACHE_TYPE_MAP допустимы для llama-server --cache-type-k/v."""
        from src.gguf.state import CACHE_TYPE_MAP

        valid = {"f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"}
        for key, val in CACHE_TYPE_MAP.items():
            assert val in valid, (
                f"Ключ {key}: значение '{val}' не поддерживается llama-server. Допустимы: {valid}"
            )

    def test_has_all_expected_keys(self):
        """Ожидаемые ключи 0-8 присутствуют."""
        from src.gguf.state import CACHE_TYPE_MAP

        for k in (0, 1, 2, 3, 4, 6, 8):
            assert k in CACHE_TYPE_MAP, f"Ключ {k} отсутствует в CACHE_TYPE_MAP"

    def test_no_q4k_or_q5k(self):
        """q4_k и q5_k НЕ ДОЛЖНЫ быть в map (llama-server падает)."""
        from src.gguf.state import CACHE_TYPE_MAP

        for val in CACHE_TYPE_MAP.values():
            assert "q4_k" not in val.lower(), f"q4_k найден: {val}"
            assert "q5_k" not in val.lower(), f"q5_k найден: {val}"

    def test_default_is_q4_0(self):
        """Ключ 2 (дефолт) → 'q4_0' или 'q8_0' (зависит от версии)."""
        from src.gguf.state import CACHE_TYPE_MAP

        assert CACHE_TYPE_MAP[2] in ("q4_0", "q8_0")


class TestLlmLoadState:
    """Структура состояния загрузки LLM."""

    def test_initial_state_structure(self):
        """_llm_load_state содержит все ожидаемые ключи."""
        from src.gguf.state import _llm_load_state

        for key in ("state", "model", "port", "task_id", "started_at", "error", "phase"):
            assert key in _llm_load_state, f"Ключ '{key}' отсутствует в _llm_load_state"

    def test_initial_state_idle(self):
        """Начальное состояние — idle."""
        from src.gguf.state import _llm_load_state

        assert _llm_load_state["state"] == "idle"

    def test_get_llm_status_returns_state_keys(self):
        """get_llm_status возвращает все ключи."""
        from src.gguf.server import get_llm_status

        status = asyncio.run(get_llm_status())
        for key in ("state", "phase", "model", "port", "error"):
            assert key in status, f"Ключ '{key}' отсутствует в get_llm_status()"


class TestServerReady:
    """Проверка is_server_ready (с моком async httpx)."""

    def test_ready_when_200(self):
        with patch("src.gguf.server.httpx.AsyncClient") as MockAC:
            from unittest.mock import AsyncMock

            mock_ctx = AsyncMock()
            mock_ctx.__aenter__.return_value.get = AsyncMock(
                return_value=MagicMock(status_code=200)
            )
            MockAC.return_value = mock_ctx
            from src.gguf.server import is_server_ready

            assert asyncio.run(is_server_ready(8081)) is True

    def test_not_ready_when_not_200(self):
        with patch("src.gguf.server.httpx.AsyncClient") as MockAC:
            from unittest.mock import AsyncMock

            mock_ctx = AsyncMock()
            mock_ctx.__aenter__.return_value.get = AsyncMock(
                return_value=MagicMock(status_code=503)
            )
            MockAC.return_value = mock_ctx
            from src.gguf.server import is_server_ready

            assert asyncio.run(is_server_ready(8081)) is False

    def test_not_ready_when_exception(self):
        with patch("src.gguf.server.httpx.AsyncClient") as MockAC:
            from unittest.mock import AsyncMock

            mock_ctx = AsyncMock()
            mock_ctx.__aenter__.return_value.get = AsyncMock(
                side_effect=Exception("Connection refused")
            )
            MockAC.return_value = mock_ctx
            from src.gguf.server import is_server_ready

            assert asyncio.run(is_server_ready(8081)) is False

    def test_not_ready_when_timeout(self):
        with patch("src.gguf.server.httpx.AsyncClient") as MockAC:
            from unittest.mock import AsyncMock

            mock_ctx = AsyncMock()
            mock_ctx.__aenter__.return_value.get = AsyncMock(side_effect=TimeoutError("timeout"))
            MockAC.return_value = mock_ctx
            from src.gguf.server import is_server_ready

            assert asyncio.run(is_server_ready(8081)) is False
