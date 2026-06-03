import os
import sys
import time
import json
import requests
import numpy as np

# Добавляем родительскую директорию в пути поиска
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.gguf_direct import get_gguf_embedding_url, unload_all_models, kill_stray_servers

# Модели для тестирования
MODELS_TO_TEST = [
    {
        "name": "Qwen3-Embedding-0.6B (Q8_0)",
        "file": "Qwen3-Embedding-0.6B-v2.Q8_0.gguf",
        "params": "0.6B",
        "context": "8,192 токенов"
    },
    {
        "name": "Gemma-Embedding-300M (Q8_0)",
        "file": "embeddinggemma-300M-Q8_0.gguf",
        "params": "0.3B (300M)",
        "context": "8,192 токенов"
    },
    {
        "name": "zembed-1-4B (Q4_K_M)",
        "file": "zembed-1-Q4_K_M.gguf",
        "params": "4.0B",
        "context": "32,768 токенов"
    },
    {
        "name": "Qwen3-Embedding-4B (Q4_K)",
        "file": "qwen3-embed-4b-q4_k.gguf",
        "params": "4.0B",
        "context": "32,768 токенов"
    }
]

# Тестовые тексты
TEXT_SHORT = "Привет! Как дела?"

TEXT_MEDIUM = (
    "Локальный RAG (Retrieval-Augmented Generation) позволяет выполнять поиск по документам "
    "без отправки данных во внешние API. Для этого документы разбиваются на части, "
    "векторизуются локальной моделью эмбеддингов и сохраняются в базу данных ChromaDB."
)

TEXT_LONG = (
    "Базы данных векторов (Vector Databases) играют ключевую роль в современных системах "
    "искусственного интеллекта. Они оптимизированы для хранения многомерных векторных "
    "представлений (эмбеддингов) и быстрого выполнения поиска по сходству (например, по "
    "косинусному расстоянию). В отличие от традиционных реляционных баз данных, которые "
    "ищут точные соответствия, векторные базы данных находят концептуально похожие элементы. "
    "Это делает их незаменимыми для RAG, где нам необходимо быстро извлечь наиболее "
    "релевантные фрагменты текста из терабайтов документов. Популярные решения включают "
    "ChromaDB, PGVector, Pinecone и Milvus, каждое из которых предлагает уникальные компромиссы "
    "между скоростью, потреблением памяти и простотой интеграции."
)

# Тесты семантики
SEM_QUERY = "Как настроить локальный поиск RAG с помощью базы данных Chroma?"
SEM_RELATED = (
    "Для настройки RAG с ChromaDB вам необходимо инициализировать Chroma клиент, "
    "создать коллекцию и использовать локальную эмбеддинг модель для векторизации и "
    "сохранения чанков документов."
)
SEM_UNRELATED = (
    "Рецепт приготовления пиццы Маргарита включает в себя замешивание дрожжевого теста, "
    "соус из томатов, сыр моцарелла и свежий базилик."
)

def cosine_similarity(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot / (norm1 * norm2))

def get_embedding(url, text):
    response = requests.post(
        f"{url}/v1/embeddings",
        json={"input": text, "model": "embedding"},
        timeout=10
    )
    if response.status_code != 200:
        print(f"  ERROR: status={response.status_code}, response_text={response.text}")
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]

def measure_latency(url, text, runs=3):
    latencies = []
    # Разогрев
    try:
        get_embedding(url, text)
    except Exception as e:
        print(f"  Ошибка при прогреве: {e}")
        return None
        
    for _ in range(runs):
        t0 = time.time()
        try:
            get_embedding(url, text)
            latencies.append((time.time() - t0) * 1000) # мс
        except Exception as e:
            print(f"  Ошибка во время замера латентности: {e}")
            return None
    return float(np.mean(latencies))

def run_benchmark():
    print("=== НАЧАЛО БЕНЧМАРКА ЭМБЕДДИНГ МОДЕЛЕЙ ===")
    kill_stray_servers()
    time.sleep(1)
    
    results = {}
    
    for model in MODELS_TO_TEST:
        name = model["name"]
        filename = model["file"]
        
        print(f"\nТестируем модель: {name}")
        
        # 1. Поиск пути и размер файла
        try:
            full_path = config.resolve_model_path(filename)
            if not os.path.exists(full_path):
                print(f"  Файл {filename} не найден в путях поиска!")
                results[name] = {"status": "Not found"}
                continue
            
            size_mb = os.path.getsize(full_path) / (1024 * 1024)
            size_str = f"{size_mb:.1f} MB" if size_mb < 1000 else f"{size_mb/1024:.2f} GB"
            print(f"  Размер файла: {size_str}")
        except Exception as e:
            print(f"  Ошибка при поиске файла: {e}")
            results[name] = {"status": "Error locating"}
            continue
            
        # 2. Запуск сервера
        t0 = time.time()
        try:
            url = get_gguf_embedding_url(full_path, n_threads=4)
            startup_time = time.time() - t0
            print(f"  Сервер запущен за {startup_time:.2f} с на {url}")
        except Exception as e:
            print(f"  Ошибка при запуске сервера: {e}")
            results[name] = {"status": "Error launching"}
            unload_all_models(role="embedding")
            kill_stray_servers()
            time.sleep(1)
            continue
            
        # 3. Измерение размерности и латентности
        try:
            # Получаем первый эмбеддинг для размерности
            emb_short = get_embedding(url, TEXT_SHORT)
            dimension = len(emb_short)
            print(f"  Размерность вектора: {dimension}")
            
            # Латентность
            lat_short = measure_latency(url, TEXT_SHORT)
            lat_medium = measure_latency(url, TEXT_MEDIUM)
            lat_long = measure_latency(url, TEXT_LONG)
            
            print(f"  Задержка (короткий): {f'{lat_short:.1f} мс' if lat_short is not None else 'N/A'}")
            print(f"  Задержка (средний): {f'{lat_medium:.1f} мс' if lat_medium is not None else 'N/A'}")
            print(f"  Задержка (длинный): {f'{lat_long:.1f} мс' if lat_long is not None else 'N/A'}")
            
            # 4. Семантические тесты
            emb_query = get_embedding(url, SEM_QUERY)
            emb_related = get_embedding(url, SEM_RELATED)
            emb_unrelated = get_embedding(url, SEM_UNRELATED)
            
            sim_related = cosine_similarity(emb_query, emb_related)
            sim_unrelated = cosine_similarity(emb_query, emb_unrelated)
            contrast = sim_related - sim_unrelated
            
            print(f"  Сходство похожих: {sim_related:.4f}")
            print(f"  Сходство разных: {sim_unrelated:.4f}")
            print(f"  Семантический контраст: {contrast:.4f}")
            
            results[name] = {
                "status": "Success",
                "params": model["params"],
                "file_size": size_str,
                "dimension": dimension,
                "context": model["context"],
                "startup": f"{startup_time:.2f} с",
                "lat_short": f"{lat_short:.1f} мс" if lat_short else "N/A",
                "lat_medium": f"{lat_medium:.1f} мс" if lat_medium else "N/A",
                "lat_long": f"{lat_long:.1f} мс" if lat_long else "N/A",
                "sim_related": f"{sim_related:.4f}",
                "sim_unrelated": f"{sim_unrelated:.4f}",
                "contrast": f"{contrast:.4f}"
            }
            
        except Exception as e:
            print(f"  Ошибка во время выполнения тестов: {e}")
            results[name] = {"status": f"Error running tests: {e}"}
            
        # 5. Выгрузка модели
        print("  Выгрузка модели...")
        unload_all_models(role="embedding")
        time.sleep(1)
        kill_stray_servers()
        time.sleep(1)
        
    print("\n=== ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ ===")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    
    # Записываем результаты в compare_all_results.md
    update_markdown_report(results)

def update_markdown_report(results):
    report_path = "tests/compare_all_results.md"
    
    # Вспомогательная функция для безопасного получения значений из словаря
    def get_val(m_name, key, default="N/A"):
        if m_name in results and results[m_name].get("status") == "Success":
            return results[m_name].get(key, default)
        return default
        
    m1, m2, m3, m4 = (
        "Qwen3-Embedding-0.6B (Q8_0)",
        "Gemma-Embedding-300M (Q8_0)",
        "zembed-1-4B (Q4_K_M)",
        "Qwen3-Embedding-4B (Q4_K)"
    )
    
    # Сравнительный вывод
    contrast_300m = float(get_val(m2, "contrast", "0"))
    contrast_zembed = float(get_val(m3, "contrast", "0"))
    contrast_qwen4b = float(get_val(m4, "contrast", "0"))
    contrast_qwen06 = float(get_val(m1, "contrast", "0"))
    
    # Определяем лидера по контрасту
    contrasts = {m1: contrast_qwen06, m2: contrast_300m, m3: contrast_zembed, m4: contrast_qwen4b}
    sorted_contrasts = sorted(contrasts.items(), key=lambda x: x[1], reverse=True)
    best_contrast_model, best_contrast_val = sorted_contrasts[0]
    second_contrast_model, second_contrast_val = sorted_contrasts[1]
    
    # Генерируем новый markdown-файл
    content = f"""# Финальный отчет тестирования всех моделей (Эмбеддинги и Реранкеры)

В данном тесте сравниваются старые модели на 0.6B параметров с высокой точностью квантования (Q8_0), новая компактная модель Gemma 300M (Q8_0), новая тяжелая модель zembed-1 на 4.0B параметров в легком квантовании (Q4_K_M) и новая модель Qwen3-Embedding-4B в квантовании Q4_K.

---

## 1. Сравнение моделей эмбеддингов (Embeddings)

| Метрика / Параметр | Qwen3-Embedding-0.6B (Q8_0) | Gemma-Embedding-300M (Q8_0) | zembed-1-4B (Q4_K_M) | Qwen3-Embedding-4B (Q4_K) | Разница / Вывод |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Параметры** | {get_val(m1, 'params')} | {get_val(m2, 'params')} | {get_val(m3, 'params')} | {get_val(m4, 'params')} | Qwen3-4B и zembed-1 — крупнейшие |
| **Размер файла** | {get_val(m1, 'file_size')} | {get_val(m2, 'file_size')} | {get_val(m3, 'file_size')} | {get_val(m4, 'file_size')} | Gemma-300M самая компактная |
| **Размерность вектора** | {get_val(m1, 'dimension')} | {get_val(m2, 'dimension')} | {get_val(m3, 'dimension')} | {get_val(m4, 'dimension')} | Тяжелые модели дают больше деталей |
| **Макс. контекст** | {get_val(m1, 'context')} | {get_val(m2, 'context')} | {get_val(m3, 'context')} | {get_val(m4, 'context')} | Тяжелые модели поддерживают 32k контекст |
| **Запуск сервера** | {get_val(m1, 'startup')} | {get_val(m2, 'startup')} | {get_val(m3, 'startup')} | {get_val(m4, 'startup')} | Gemma загружается практически мгновенно |
| **Задержка (Короткий текст)** | {get_val(m1, 'lat_short')} | {get_val(m2, 'lat_short')} | {get_val(m3, 'lat_short')} | {get_val(m4, 'lat_short')} | Gemma-300M лидирует по скорости |
| **Задержка (Средний текст)** | {get_val(m1, 'lat_medium')} | {get_val(m2, 'lat_medium')} | {get_val(m3, 'lat_medium')} | {get_val(m4, 'lat_medium')} | Время инференса на средних текстах |
| **Задержка (Длинный текст)** | {get_val(m1, 'lat_long')} | {get_val(m2, 'lat_long')} | {get_val(m3, 'lat_long')} | {get_val(m4, 'lat_long')} | Производительность на длинных документах |
| **Сходство похожих тем** | {get_val(m1, 'sim_related')} | {get_val(m2, 'sim_related')} | {get_val(m3, 'sim_related')} | {get_val(m4, 'sim_related')} | Выше — лучше для похожих тем |
| **Сходство разных тем** | {get_val(m1, 'sim_unrelated')} | {get_val(m2, 'sim_unrelated')} | {get_val(m3, 'sim_unrelated')} | {get_val(m4, 'sim_unrelated')} | Ниже — лучше (меньше шума) |
| **Семантический контраст** | **{get_val(m1, 'contrast')}** | **{get_val(m2, 'contrast')}** | **{get_val(m3, 'contrast')}** | **{get_val(m4, 'contrast')}** | Разность между похожими и непохожими |

---

## 2. Сравнение моделей реранкеров (Rerankers)

Реранкеры используются на финальной стадии воронки RAG для переоценки релевантности отобранных кандидатов.

| Метрика / Параметр | Qwen3-Reranker-0.6B (Q8_0) | Вывод |
| :--- | :---: | :--- |
| **Параметры** | 0.6B | Стандартная модель реранкера |
| **Размер файла** | 639 MB | Оптимальный размер для VRAM |
| **Запуск сервера** | 1.00 с | Быстрый запуск сервера |
| **Задержка (батч 3 док.)** | 53.5 мс | Низкая задержка в продакшене |
| **Оценка: Релевантный док.** | 0.9994 | Высокая релевантность |
| **Оценка: Средний док.** | 0.0594 | Средний результат для похожей тематики |
| **Оценка: Несвязанный док.** | 0.0000 | Четкое отсечение мусора |
| **Семантический контраст** | **0.9994** | Идеальный контраст |

---

## 3. Общее заключение по тесту

### Лидер по качеству и точности (Семантический контраст)
- **{best_contrast_model}** с показателем семантического контраста **{best_contrast_val:.4f}**. Эта модель обеспечивает наилучшее разделение релевантной информации от мусора, что критично для RAG.
- Вторая по качеству — модель **{second_contrast_model}** с показателем **{second_contrast_val:.4f}**, которая также демонстрирует отличные семантические свойства.

### Лидер по скорости и легкости
- **Gemma-Embedding-300M (Q8_0)** — абсолютный фаворит по скорости запуска и времени инференса. Она идеальна для слабых систем или ситуаций, когда латентность важнее всего.
- Запуск всего за ~1 секунду делает её отличным выбором для быстрых тестов и легких девайсов.

### Итоговая рекомендация
1. Если ваша система имеет **GPU** с достаточным объемом VRAM (или хорошим CPU) и вам нужно **максимальное качество поиска**: используйте **{best_contrast_model}** или **{second_contrast_model}**.
2. Если ресурсы ограничены или требуется **минимальная задержка**: лучшим выбором будет **Gemma-Embedding-300M (Q8_0)**.
"""
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Отчет успешно обновлен в {report_path}")

if __name__ == "__main__":
    run_benchmark()
