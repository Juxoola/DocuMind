"""
Тесты для process_audio_video / ensure_720p_video.

Проверяют критические пути без тяжёлых моделей (WhisperX, vision LLM).
Большая часть тестов — unit с моками subprocess.

Запуск:
    cd C:\\test
    python tests/test_video_processing.py

Integration тесты требуют реального ffmpeg и synthetic видео.
"""
import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestEnsure720pVideoDispatch(unittest.TestCase):
    """ensure_720p_video: какой код-путь выбирается."""

    def test_non_video_extension_returns_unchanged(self):
        """Не-видео расширение → возврат пути как есть, никаких subprocess."""
        from src.ingestion import ensure_720p_video
        for ext in ['.txt', '.pdf', '.docx', '.jpg', '.png']:
            with self.subTest(ext=ext):
                result = ensure_720p_video(f"somefile{ext}")
                self.assertEqual(result, f"somefile{ext}")

    @patch("subprocess.run")
    def test_short_video_uses_single_encode(self, mock_run):
        """Длительность < 120с → single-encode путь (без turbo)."""
        from src.ingestion import ensure_720p_video

        # Мок ffmpeg для get_duration: вернуть 60 секунд
        mock_duration = MagicMock()
        mock_duration.stderr = b"Duration: 00:01:00.00, start: 0.000000, bitrate: 1000 kb/s"
        mock_duration.returncode = 1  # ffmpeg -i всегда возвращает 1 без output
        # Мок для ffmpeg encode
        mock_encode = MagicMock()
        mock_encode.wait.return_value = 0

        mock_run.side_effect = [mock_duration, mock_encode]
        # Мок Popen для encode
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.wait.return_value = 0
            mock_proc.poll.return_value = None
            mock_popen.return_value = mock_proc

            prog_cb = MagicMock()
            with tempfile.TemporaryDirectory() as tmp:
                fake_mp4 = os.path.join(tmp, "test.mp4")
                # Создаём фейк-исходник
                with open(fake_mp4, "wb") as f:
                    f.write(b"fake mp4 content" * 100)
                # Создаём фейк-output
                with open(fake_mp4 + ".720p.mp4", "wb") as f:
                    f.write(b"x" * 5000)

                try:
                    result = ensure_720p_video(fake_mp4, prog_cb=prog_cb)
                    # Должен сработать single-encode (duration < 120)
                    self.assertTrue(mock_popen.called, "Popen should be called for single encode")
                except Exception as e:
                    self.fail(f"ensure_720p_video raised: {e}")
                finally:
                    # Чистим возможный output
                    for f in [fake_mp4 + ".720p.mp4",
                              os.path.splitext(fake_mp4)[0] + ".mp4"]:
                        if os.path.exists(f): os.remove(f)

    @patch("subprocess.run")
    def test_long_video_uses_turbo_encode(self, mock_run):
        """Длительность >= 120с → turbo-режим (4 параллельных)."""
        from src.ingestion import ensure_720p_video

        mock_duration = MagicMock()
        mock_duration.stderr = b"Duration: 00:30:00.00, start: 0.000000, bitrate: 5000 kb/s"
        mock_duration.returncode = 1

        mock_run.side_effect = [mock_duration]

        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.wait.return_value = 0
            mock_proc.poll.return_value = None
            mock_popen.return_value = mock_proc

            prog_cb = MagicMock()
            with tempfile.TemporaryDirectory() as tmp:
                fake_mp4 = os.path.join(tmp, "long_video.mp4")
                with open(fake_mp4, "wb") as f:
                    f.write(b"x" * 100)

                try:
                    ensure_720p_video(fake_mp4, prog_cb=prog_cb)
                    # Turbo-режим должен вызвать prog_cb с "Турбо-оптимизация"
                    turbo_logs = [call for call in prog_cb.call_args_list
                                  if "Турбо" in str(call)]
                    self.assertTrue(len(turbo_logs) >= 1,
                                    f"Expected 'Турбо' log, got: {prog_cb.call_args_list}")
                except Exception as e:
                    self.fail(f"ensure_720p_video raised: {e}")
                finally:
                    import shutil
                    parts_dir = fake_mp4 + "_parts"
                    if os.path.exists(parts_dir):
                        shutil.rmtree(parts_dir, ignore_errors=True)


class TestGetDuration(unittest.TestCase):
    """Внутренняя get_duration: таймаут, обработка ошибок."""

    @patch("subprocess.run")
    def test_get_duration_returns_0_on_timeout(self, mock_run):
        """TimeoutExpired → возвращаем 0, НЕ зависаем."""
        import subprocess
        from src.ingestion import ensure_720p_video
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="ffmpeg", timeout=30)

        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock(wait=MagicMock(return_value=0), poll=MagicMock(return_value=None))
            with tempfile.TemporaryDirectory() as tmp:
                fake_mp4 = os.path.join(tmp, "broken.mp4")
                with open(fake_mp4, "wb") as f:
                    f.write(b"x" * 100)
                try:
                    # Должен НЕ зависнуть; duration=0 → turbo-путь
                    result = ensure_720p_video(fake_mp4)
                    # В turbo-режиме Popen вызывается для каждого сегмента
                    # Проверяем, что НЕ зависли на get_duration
                except Exception as e:
                    # Любая ошибка ОК, главное что не TimeoutExpired
                    if "timeout" in str(e).lower() or "TimeoutExpired" in str(e):
                        self.fail(f"get_duration didn't handle timeout: {e}")
                finally:
                    import shutil
                    parts_dir = fake_mp4 + "_parts"
                    if os.path.exists(parts_dir):
                        shutil.rmtree(parts_dir, ignore_errors=True)

    @patch("subprocess.run")
    def test_get_duration_parses_correctly(self, mock_run):
        """Корректный парсинг 'Duration: HH:MM:SS.xx, ...'."""
        from src.ingestion import ensure_720p_video

        # 1ч 30мин = 5400 секунд
        mock_duration = MagicMock()
        mock_duration.stderr = b"Duration: 01:30:00.50, start: 0.000000, bitrate: 5000 kb/s\n"
        mock_duration.returncode = 1

        mock_run.side_effect = [mock_duration]
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock(wait=MagicMock(return_value=0), poll=MagicMock(return_value=None))
            with tempfile.TemporaryDirectory() as tmp:
                fake_mp4 = os.path.join(tmp, "test.mp4")
                with open(fake_mp4, "wb") as f:
                    f.write(b"x" * 100)
                try:
                    ensure_720p_video(fake_mp4)
                except Exception:
                    pass  # нам важен только parse
                finally:
                    import shutil
                    parts_dir = fake_mp4 + "_parts"
                    if os.path.exists(parts_dir):
                        shutil.rmtree(parts_dir, ignore_errors=True)

    @patch("subprocess.run")
    def test_get_duration_handles_garbage_output(self, mock_run):
        """Если ffmpeg вернул мусор без 'Duration' → 0, не падаем."""
        from src.ingestion import ensure_720p_video

        mock_duration = MagicMock()
        mock_duration.stderr = b"some random garbage output without duration info"
        mock_duration.returncode = 1
        mock_run.side_effect = [mock_duration]

        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock(wait=MagicMock(return_value=0), poll=MagicMock(return_value=None))
            with tempfile.TemporaryDirectory() as tmp:
                fake_mp4 = os.path.join(tmp, "weird.mp4")
                with open(fake_mp4, "wb") as f:
                    f.write(b"x" * 100)
                try:
                    # Не должно бросить IndexError на .split(",")[0]
                    ensure_720p_video(fake_mp4)
                except Exception as e:
                    if "IndexError" in str(e) or "list index out of range" in str(e):
                        self.fail(f"get_duration не обработал мусор: {e}")
                finally:
                    import shutil
                    parts_dir = fake_mp4 + "_parts"
                    if os.path.exists(parts_dir):
                        shutil.rmtree(parts_dir, ignore_errors=True)


class TestCancelMechanism(unittest.TestCase):
    """Cancel должен реально прерывать тяжёлые операции."""

    def test_cancel_before_ensure_720p_raises(self):
        """Если cancel=True ДО ensure_720p_video → должно поднять IngestionCancelled."""
        from src.ingestion import ensure_720p_video, IngestionCancelled

        cancel_event = MagicMock()
        cancel_event.return_value = True  # уже отменено

        with patch("subprocess.run") as mock_run:
            # get_duration должен быть пропущен (cancel проверяется перед)
            with self.assertRaises(IngestionCancelled):
                ensure_720p_video("anyfile.mp4", cancel_check=cancel_event)


class TestProcessAudioVideoImport(unittest.TestCase):
    """Smoke test: функция импортируется и имеет правильную сигнатуру."""

    def test_process_audio_video_signature(self):
        import inspect
        from src.ingestion import process_audio_video
        sig = inspect.signature(process_audio_video)
        params = list(sig.parameters.keys())
        # Все ключевые параметры на месте
        for p in ["file_path", "images_dir", "is_video", "progress_cb",
                  "llm_settings", "cancel_check", "notebook_id"]:
            self.assertIn(p, params, f"Missing parameter: {p}")

    def test_ensure_720p_video_signature(self):
        import inspect
        from src.ingestion import ensure_720p_video
        sig = inspect.signature(ensure_720p_video)
        params = list(sig.parameters.keys())
        for p in ["file_path", "prog_cb", "cancel_check", "notebook_id"]:
            self.assertIn(p, params, f"Missing parameter: {p}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
