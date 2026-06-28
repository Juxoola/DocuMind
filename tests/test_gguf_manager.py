"""Тесты модуля gguf_manager: сканирование директорий, кеширование, поиск моделей."""

import asyncio
import os
import sys
import time
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config


# ── Сканирование директорий на наличие GGUF-моделей ──
class TestScanGgufDirs:
    def test_empty_dirs(self, tmp_path):
        from src.gguf.scanner import scan_gguf_dirs

        with patch.object(config, "GGUF_SEARCH_DIRS", str(tmp_path)):
            from src.gguf.scanner import invalidate_scan_cache

            asyncio.run(invalidate_scan_cache())
            results = asyncio.run(scan_gguf_dirs())
            assert results == []

    def test_finds_gguf_files(self, tmp_path):
        from src.gguf.scanner import invalidate_scan_cache, scan_gguf_dirs

        subdir = tmp_path / "models"
        subdir.mkdir()
        (subdir / "model.q4_k_m.gguf").write_text("dummy")
        (subdir / "model.mmproj.q4_k_m.gguf").write_text("dummy")

        with patch.object(config, "GGUF_SEARCH_DIRS", str(tmp_path)):
            asyncio.run(invalidate_scan_cache())
            results = asyncio.run(scan_gguf_dirs())

            assert len(results) == 1
            entry = results[0]
            assert "model.q4_k_m.gguf" in entry["gguf_files"]
            assert "model.mmproj.q4_k_m.gguf" in entry["mmproj_files"]

    def test_cache_hit(self, tmp_path):
        from src.gguf.scanner import invalidate_scan_cache, scan_gguf_dirs

        subdir = tmp_path / "models"
        subdir.mkdir()
        (subdir / "model.gguf").write_text("dummy")

        with patch.object(config, "GGUF_SEARCH_DIRS", str(tmp_path)):
            asyncio.run(invalidate_scan_cache())
            r1 = asyncio.run(scan_gguf_dirs())
            r2 = asyncio.run(scan_gguf_dirs())
            assert r1 == r2

    def test_cache_invalidation(self, tmp_path):
        from src.gguf.scanner import invalidate_scan_cache, scan_gguf_dirs

        subdir = tmp_path / "models"
        subdir.mkdir()
        (subdir / "model.gguf").write_text("dummy")

        with patch.object(config, "GGUF_SEARCH_DIRS", str(tmp_path)):
            asyncio.run(invalidate_scan_cache())
            r1 = asyncio.run(scan_gguf_dirs())
            assert len(r1) == 1
            (subdir / "new_model.gguf").write_text("dummy2")
            time.sleep(1.1)

            asyncio.run(invalidate_scan_cache())
            r2 = asyncio.run(scan_gguf_dirs())
            assert len(r2) == 1
            assert "new_model.gguf" in r2[0]["gguf_files"]

    def test_scan_nonexistent_dir(self):
        from src.gguf.scanner import invalidate_scan_cache, scan_gguf_dirs

        with patch.object(config, "GGUF_SEARCH_DIRS", "/nonexistent/path"):
            asyncio.run(invalidate_scan_cache())
            results = asyncio.run(scan_gguf_dirs())
            assert results == []

    def test_multiple_search_dirs(self, tmp_path):
        from src.gguf.scanner import invalidate_scan_cache, scan_gguf_dirs

        d1 = tmp_path / "dir1"
        d2 = tmp_path / "dir2"
        d1.mkdir()
        d2.mkdir()
        (d1 / "a.gguf").write_text("a")
        (d2 / "b.gguf").write_text("b")

        with patch.object(config, "GGUF_SEARCH_DIRS", f"{d1};{d2}"):
            asyncio.run(invalidate_scan_cache())
            results = asyncio.run(scan_gguf_dirs())
            assert len(results) == 2
            all_files = []
            for r in results:
                all_files.extend(r["gguf_files"])
            assert sorted(all_files) == sorted(["a.gguf", "b.gguf"])


# ── Поиск моделей по имени файла ──
class TestFindGgufByName:
    def test_find_existing(self, tmp_path):
        from src.gguf.scanner import find_gguf_by_name, invalidate_scan_cache

        (tmp_path / "target.gguf").write_text("data")

        with patch.object(config, "GGUF_SEARCH_DIRS", str(tmp_path)):
            asyncio.run(invalidate_scan_cache())
            result = asyncio.run(find_gguf_by_name("target.gguf"))
            assert result is not None
            assert result.endswith("target.gguf")

    def test_find_nonexistent(self, tmp_path):
        from src.gguf.scanner import find_gguf_by_name, invalidate_scan_cache

        with patch.object(config, "GGUF_SEARCH_DIRS", str(tmp_path)):
            asyncio.run(invalidate_scan_cache())
            assert asyncio.run(find_gguf_by_name("no_such_file.gguf")) is None

    def test_find_empty_name(self, tmp_path):
        from src.gguf.scanner import find_gguf_by_name

        assert asyncio.run(find_gguf_by_name("")) is None
        assert asyncio.run(find_gguf_by_name(None)) is None

    def test_find_mmproj(self, tmp_path):
        from src.gguf.scanner import find_gguf_by_name, invalidate_scan_cache

        (tmp_path / "mmproj-model.Q4_K_M.gguf").write_text("data")

        with patch.object(config, "GGUF_SEARCH_DIRS", str(tmp_path)):
            asyncio.run(invalidate_scan_cache())
            result = asyncio.run(find_gguf_by_name("mmproj-model.Q4_K_M.gguf"))
            assert result is not None
            assert "mmproj" in result


# ── Время модификации директорий ──
class TestDirMtime:
    def test_existing_dir(self, tmp_path):
        from src.gguf.scanner import _dir_mtime

        mtime = asyncio.run(_dir_mtime(str(tmp_path)))
        assert isinstance(mtime, float)
        assert mtime > 0

    def test_nonexistent_dir(self):
        from src.gguf.scanner import _dir_mtime

        assert asyncio.run(_dir_mtime("/nonexistent_path_xyz")) == 0.0

    def test_dir_mtime_changes_on_modification(self, tmp_path):
        from src.gguf.scanner import _dir_mtime

        before = asyncio.run(_dir_mtime(str(tmp_path)))
        time.sleep(1.1)
        (tmp_path / "new_file.txt").write_text("test")
        after = asyncio.run(_dir_mtime(str(tmp_path)))
        assert after >= before


# ── Инвалидация кэша сканирования ──
class TestInvalidateCache:
    def test_invalidate_removes_cache_file(self, tmp_path):
        from src.gguf.scanner import invalidate_scan_cache

        fake_cache = str(tmp_path / "_gguf_scan_cache.json")
        with open(fake_cache, "w") as f:
            f.write("{}")

        with patch("src.gguf.scanner._GGUF_CACHE_FILE", fake_cache):
            asyncio.run(invalidate_scan_cache())
            assert not os.path.exists(fake_cache)

    def test_invalidate_no_cache_file(self):
        from src.gguf.scanner import invalidate_scan_cache

        asyncio.run(invalidate_scan_cache())
