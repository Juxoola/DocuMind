# DocuMind

**Локальный аналог Google NotebookLM** — RAG-система для работы с документами на полностью локальном стеке. FastAPI + React + локальные GGUF-модели.

![Python](https://img.shields.io/badge/python-3.11+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.136-green)
![React](https://img.shields.io/badge/React-19-61DAFB)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## Возможности

**📄 Поддержка форматов**
PDF, DOCX, PPTX, TXT, MD, PNG/JPG/WEBP, MP4/AVI/MKV, MP3/WAV/M4A — всё грузится и индексируется.

**🔍 Гибридный поиск (RAG)**
- Векторный поиск через ChromaDB
- Keyword retrieval через BM25
- Reciprocal Rank Fusion для объединения результатов
- Опциональный реранкер (локальный GGUF)

**🧠 Локальные модели**
- Эмбеддинги: Qwen3-Embedding через llama.cpp (GGUF)
- Реранкер: Qwen3-Reranker через llama.cpp (GGUF)
- LLM: любой OpenAI-совместимый сервер (LM Studio, Ollama, llama.cpp)

**💬 Чат с документами**
- Ответы с цитированием источников
- Расширение запроса (query expansion)
- Markdown-разметка, формулы KaTeX, подсветка кода

**🔖 Организация**
- Notebooks (блокноты) — изолированные проекты
- Закладки на фрагменты документов
- Просмотрщик исходных документов встроенный

**🔒 Полная приватность**
- Ничего не уходит в интернет
- Все модели и данные — локально
- Опционально: внешний LLM через LM Studio / любой OpenAI endpoint

---

## Быстрый старт

### Требования

| Компонент | Версия |
|-----------|--------|
| Python | 3.11+ |
| Node.js | 18+ |
| llama-server.exe | последний релиз llama.cpp |

### Установка

```powershell
# 1. Клонировать и открыть папку проекта
cd C:\DocuMind

# 2. Установить все зависимости (Python + frontend + .env)
.\setup.bat

# 3. Поместить GGUF-модели в папку models/
#    (Qwen3-Embedding-0.6B-v2.Q8_0.gguf + Qwen3-Reranker-0.6B-v2.Q8_0.gguf)

# 4. Положить llama-server.exe в bin/
#    (скачать с https://github.com/ggml-org/llama.cpp/releases)

# 5. Настроить LLM-сервер (LM Studio / Ollama) на порту 1234,
#    или указать другой URL в .env
```

### Запуск

```powershell
# Полный запуск (backend + frontend) — одно окно
.\run-merged.bat

# Только backend (для разработки фронтенда отдельно)
.\run-merged.bat -NoFrontend

# Или через PowerShell напрямую
.\run-merged.ps1
```

**Или установи ярлыки на рабочем столе:**
```powershell
.\create_shortcuts.bat
```

После запуска:
- **Backend API:** http://localhost:8000/docs
- **Frontend:** http://localhost:5173

---

## Архитектура

```
┌──────────────────────────────────────────────────┐
│                  Frontend (Vite + React)          │
│  NotebookSelector → MainApp → ChatArea           │
│                           → DocumentViewer        │
│                           → Sidebar              │
│                           → SettingsModal        │
└──────────────────────┬───────────────────────────┘
                       │ HTTP (axios)
                       ▼
┌──────────────────────────────────────────────────┐
│              Backend (FastAPI)                    │
│                                                   │
│  routers/                                        │
│  ├── notebooks.py    — CRUD блокнотов            │
│  ├── files.py        — загрузка/удаление файлов  │
│  ├── chat.py         — RAG-чат с LLM             │
│  ├── gguf.py         — управление GGUF-серверами │
│  ├── bookmarks.py    — закладки                  │
│  └── settings.py     — настройки                 │
│                                                   │
│  src/                                            │
│  ├── rag/            — RAG-пайплайн              │
│  ├── gguf/           — менеджер GGUF-процессов   │
│  └── ingestion/      — обработка файлов          │
└──────────┬───────────────┬───────────────────────┘
           │               │
           ▼               ▼
┌──────────────┐    ┌──────────────┐
│  ChromaDB    │    │  BM25 (disk) │
│  (векторы)   │    │  (keywords)  │
└──────────────┘    └──────────────┘
           │               │
           └───────┬───────┘
                   ▼
┌──────────────────────────────────────┐
│         LLM / Embedding Engine       │
│  ┌──────────┐  ┌──────────────────┐  │
│  │ LM Studio│  │ llama-server.exe │  │
│  │ (Ollama) │  │ (GGUF локально) │  │
│  └──────────┘  └──────────────────┘  │
└──────────────────────────────────────┘
```

### Компоненты

- **routers/** — 6 APIRouter: notebooks, files, chat, gguf, bookmarks, settings
- **src/rag/** — пайплайн: эмбеддинги → ChromaDB → BM25 → RRF → реранкер → LLM
- **src/gguf/** — менеджер процессов llama-server.exe (запуск/остановка/healthcheck)
- **src/ingestion/** — парсеры PDF, DOCX, PPTX, изображений, аудио/видео (WhisperX, OpenCV)
- **frontend/src/components/** — MainApp, ChatArea, Sidebar, DocumentViewer, SettingsModal, NotebookSelector

---

## Документация

Подробная документация расположена в `docs/`:

### Пайплайны
- [Пайплайн ингеста](docs/pipelines/ingestion.md) — как документы попадают в систему
- [Пайплайн RAG-поиска](docs/pipelines/rag-search.md) — гибридный поиск с RRF и реранкингом
- [Пайплайн GGUF-серверов](docs/pipelines/gguf-servers.md) — управление локальными моделями
- [Пайплайн чата](docs/pipelines/chat.md) — генерация ответов с цитированием

### API и функции
- [Backend: routers](docs/areas/backend-routers.md) — все API-эндпоинты
- [Backend: core](docs/areas/backend-core.md) — main.py, config.py
- [Backend: GGUF](docs/areas/backend-gguf.md) — модуль управления серверами
- [Backend: ingestion](docs/areas/backend-ingestion.md) — парсеры файлов
- [Backend: RAG](docs/areas/backend-rag.md) — пайплайн поиска
- [Frontend: components](docs/areas/frontend-components.md) — React-компоненты
- [Frontend: lib](docs/areas/frontend-lib.md) — утилиты

### Для агентов
- [AGENTS.md](docs/AGENTS.md) — справочник для AI-агентов

---

## Конфигурация

Все настройки в `.env` в корне проекта:

```ini
# Backend
HOST=0.0.0.0
PORT=8000

# LLM (OpenAI-compatible)
LM_STUDIO_URL=http://localhost:1234/v1
LLM_DEFAULT_MODEL=gpt-4o

# Пути поиска .gguf файлов (разделитель ";")
GGUF_SEARCH_DIRS=F:/llm;C:/test/models

# Лимиты
UPLOAD_MAX_SIZE_MB=500
```

Полный список параметров — в `config.py`.

---

## Запуск тестов

```bash
# Backend (123 теста)
python -m pytest tests/

# Frontend (8 тестов)
cd frontend && npx vitest run
```

---

## Git-стратегия

Проект использует pre-commit хуки:
- **ruff** — форматирование + линтинг
- **pytest** — все backend-тесты
- **vitest** — все frontend-тесты

```bash
# Установить pre-commit
pre-commit install

# Запустить вручную
pre-commit run --all-files
```

---

## Файловая структура

```
C:\DocuMind/
├── bin/                   # llama-server.exe + DLL
├── docs/                  # Документация
│   ├── areas/             # API/функции (генерируется агентами)
│   ├── pipelines/         # Описание пайплайнов
│   └── AGENTS.md          # Справочник для агентов
├── frontend/              # React (Vite + Tailwind)
│   └── src/components/    # Основные компоненты UI
├── logs/                  # server.log
├── models/                # GGUF-модели (.gguf)
├── notebooks/             # Данные блокнотов
│   └── {nb_id}/
│       ├── chroma_db/     # Векторная БД
│       ├── bm25/          # BM25 индекс
│       ├── data/          # Загруженные файлы
│       └── images/        # Извлечённые изображения
├── routers/               # API-эндпоинты
├── src/
│   ├── rag/               # RAG-пайплайн
│   ├── gguf/              # GGUF-server manager
│   └── ingestion/         # Парсеры файлов
├── tests/                 # Pytest тесты
├── static/                # Статика (для standalone)
├── config.py              # Настройки из .env
├── main.py                # Точка входа FastAPI
├── setup.bat / .ps1       # Установка
├── run-merged.bat / .ps1  # Запуск (одно окно)
└── create_shortcuts.bat   # Ярлыки на рабочий стол
```

---

## Будущие улучшения

### Мультимодальный embedding

Текущий пайплайн эмбеджит производные данные (транскрипт WhisperX, описание Vision) текстовыми моделями. В планах — переход на **мульти модальный embedding** напрямую из оригинального контента:

- **Видео** — Jina v5 Omni эмбеджит видео целиком (вместо кадров + описаний)
- **Аудио** — прямой embedding аудио-дорожки (без промежуточного WhisperX-транскрипта для поиска)
- **Изображения** — embedding пикселей, а не OCR-текста

Это повысит точность поиска по визуальному и аудиальному контенту — теряется меньше информации на каждом шаге.

---

## Лицензия

MIT.
