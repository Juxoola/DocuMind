# Миграция на прямой API llama-cpp-python

## Что изменилось

Проект переведён с серверного режима llama-cpp-python на прямой Python API.

### Было (старый подход)
- Запуск отдельного процесса `python -m llama_cpp.server`
- Долгий старт сервера (10-30 секунд)
- Управление процессом через subprocess
- HTTP запросы к локальному серверу
- Endpoints: `/api/gguf-server/start`, `/api/gguf-server/stop`, `/api/gguf-server/status`

### Стало (новый подход)
- Прямое использование LlamaCPP через llama-index
- Модель загружается при первом запросе (lazy loading)
- Модели кэшируются в памяти
- Нет сетевых запросов, всё в одном процессе
- Endpoints: `/api/gguf-loaded`, `/api/gguf-unload`

## Новые файлы

- `src/gguf_direct.py` — модуль для работы с GGUF через прямой API
  - `load_gguf_model()` — загрузить модель в память
  - `get_gguf_llm()` — получить LLM объект (с кэшированием)
  - `unload_gguf_model()` — выгрузить модель из памяти
  - `get_loaded_models()` — список загруженных моделей

## Изменённые файлы

### main.py
- Убраны импорты `start_gguf_server`, `stop_gguf_server`
- Добавлен импорт `gguf_direct`
- Обновлён `upload_file()` — передаёт настройки GGUF в ingestion
- Обновлён `chat()` — использует прямой API через `get_gguf_llm()`
- Убраны endpoints `/api/gguf-server/*`
- Добавлены endpoints `/api/gguf-loaded`, `/api/gguf-unload`
- Обновлён shutdown handler — вызывает `unload_all_models()`

### src/ingestion.py
- Обновлена `describe_image_with_lmstudio()` — поддержка GGUF Direct API
- Проверяет флаг `use_gguf_direct` в `llm_settings`
- Для multimodal использует `get_gguf_llm()` с mmproj

### frontend/src/components/SettingsModal.jsx
- Убраны функции `startServer()`, `stopServer()`, `fetchGgufServerStatus()`
- Добавлены `selectModel()`, `unloadAllModels()`, `fetchGgufLoadedModels()`
- Убраны параметры сервера (контекст, GPU слои, потоки)
- Упрощён UI — просто выбор модели без запуска/остановки

## Как использовать

1. Открыть настройки → вкладка "Локальные GGUF"
2. Выбрать модель из списка (кнопка "Выбрать")
3. Сохранить настройки
4. При первом запросе модель загрузится автоматически
5. Последующие запросы используют кэшированную модель

## Преимущества

✅ Быстрый старт — нет задержки на запуск сервера
✅ Проще — нет управления процессами
✅ Надёжнее — нет сетевых запросов
✅ Меньше зависимостей — не нужен starlette-context

## Недостатки

⚠️ Модель остаётся в памяти GPU/RAM пока работает приложение
⚠️ Нельзя использовать модель из других приложений одновременно

## Откат на старый подход

Если нужно вернуться к серверному режиму:

```bash
git revert HEAD~2  # Откатить последние 2 коммита
```

Или использовать коммит `6dd2208` (последний с серверным подходом).

## Коммиты

- `fe3aca2` — Переход на прямой API llama-cpp-python
- `5c60308` — Обновлён фронтенд для работы с прямым API GGUF
