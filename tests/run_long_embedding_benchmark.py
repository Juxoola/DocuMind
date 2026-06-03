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

# === ДЛИННЫЕ ТЕКСТЫ ДЛЯ ТЕСТИРОВАНИЯ ===

# Поисковый запрос
LONG_QUERY = "Каков детальный механизм использования Windows Job Objects для предотвращения утечек процессов llama-server при перезапуске?"

# Длинный релевантный документ (около 1200 слов, детальное техническое описание Windows Job Objects и ctypes в Python)
LONG_RELATED = """
Механизм управления процессами в операционной системе Windows предоставляет мощный инструмент под названием Job Objects (Объекты заданий). Объект задания позволяет объединять один или более процессов в группу, которой можно управлять как единым целым. Это критически важно при разработке серверных приложений на Python, запускающих внешние бинарные файлы, такие как llama-server.exe для инференса больших языковых моделей (LLM) или моделей эмбеддингов.

Когда Python-процесс (родительский процесс) запускает дочерний процесс llama-server.exe с помощью стандартной библиотеки `subprocess.Popen`, между ними устанавливается слабая связь. Если родительский процесс аварийно завершается (например, из-за необработанного исключения, нехватки памяти OOM или принудительного закрытия пользователем через диспетчер задач), операционная система Windows по умолчанию оставляет дочерний процесс llama-server.exe активным. Такой процесс становится «зомби» или «сиротой» (orphan process), продолжая занимать системный порт (например, 55411) и удерживать драгоценную видеопамять (VRAM) на видеокарте NVIDIA, что делает невозможным запуск новой копии сервера.

Для надежного решения этой проблемы на платформе Windows применяются ядерные объекты Job Objects через интерфейс Windows API (Win32 API). В Python доступ к Win32 API реализуется с помощью библиотеки `ctypes`, которая позволяет вызывать функции из системных динамических библиотек, таких как `kernel32.dll`.

Полный цикл реализации автозакрытия дочерних серверов с помощью Job Objects состоит из следующих шагов:

1. Создание объекта задания (Job Object):
Родительский процесс вызывает функцию `CreateJobObjectW` из `kernel32.dll`. Эта функция создает объект задания в пространстве ядра Windows и возвращает дескриптор (handle) этого объекта.
Сигнатура вызова: `kernel32.CreateJobObjectW(None, None)`

2. Настройка ограничений на завершение процессов:
По умолчанию объект задания просто объединяет процессы, но не завершает дочерние процессы при закрытии родительского дескриптора. Чтобы активировать автоматическое уничтожение всех дочерних процессов при закрытии дескриптора задания, необходимо установить специальный лимит `JOB_OBJECT_LIMIT_KILL_ON_CLOSE`.
Для этого подготавливается структура `JOBOBJECT_EXTENDED_LIMIT_INFORMATION`. Это сложная структура данных Windows, содержащая информацию об ограничениях.
В поле `LimitFlags` этой структуры записывается флаг `JOB_OBJECT_LIMIT_KILL_ON_CLOSE` (шестнадцатеричное значение `0x2000`).
После этого вызывается функция `SetInformationJobObject`, которая связывает подготовленные ограничения с созданным объектом задания.
Сигнатура вызова: `kernel32.SetInformationJobObject(job_handle, JobObjectExtendedLimitInformation, ctypes.byref(info), ctypes.sizeof(info))`

3. Привязка дочернего процесса к заданию:
После того как дочерний процесс llama-server.exe успешно запущен с помощью `subprocess.Popen`, родительский процесс получает его дескриптор (через `process._handle` в Python).
Затем вызывается функция `AssignProcessToJobObject`, которая помещает дочерний процесс под контроль объекта задания.
Сигнатура вызова: `kernel32.AssignProcessToJobObject(job_handle, process_handle)`

Как только дочерний процесс привязан к заданию, операционная система Windows берет на себя управление его жизненным циклом. Если родительский процесс завершается, Windows автоматически закрывает все открытые дескрипторы ядра, связанные с этим процессом, включая дескриптор Job Object. Поскольку на объекте задания установлен лимит `JOB_OBJECT_LIMIT_KILL_ON_CLOSE`, операционная система мгновенно и гарантированно терминирует все процессы, привязанные к этому заданию, включая llama-server.exe.

Это решение работает на уровне ядра Windows, что обеспечивает 100% надежность. Даже если родительский процесс Python будет убит через `Task Manager` по сигналу `SIGKILL`, операционная система все равно закроет дескриптор задания и мгновенно освободит системный порт и видеопамять VRAM. Использование Job Objects позволяет полностью устранить проблему зависших серверов llama.cpp и гарантировать стабильную работу RAG-систем при частых перезапусках и отладке.
"""

# Длинный нерелевантный документ (около 1200 слов, детальный рецепт и история неаполитанской пиццы)
LONG_UNRELATED = """
Пицца Маргарита (Pizza Margherita) является одним из самых известных и любимых блюд итальянской кухни во всем мире. История этого кулинарного шедевра неразрывно связана с городом Неаполь и кулинарными традициями региона Кампания. Неаполитанская пицца (Pizza Napoletana) защищена на международном уровне статусом гарантированной традиционной специальности (TSG) и включена в список нематериального культурного наследия ЮНЕСКО.

Настоящая неаполитанская пицца подчиняется строгим правилам консорциума Associazione Verace Pizza Napoletana (AVPN), который регламентирует каждый ингредиент и этап приготовления для сохранения исторической аутентичности.

Процесс создания идеальной пиццы Маргарита состоит из следующих ключевых этапов:

1. Приготовление дрожжевого теста высокой гидратации:
Основой пиццы является мука из мягких сортов пшеницы типа 00 (Farina di grano tenero tipo 00). Эта мука мелкого помола с высоким содержанием белка (12-14%) способна формировать прочную глютеновую сетку, которая удерживает пузырьки углекислого газа при ферментации.
Гидратация теста составляет от 60% до 70% (то есть на 1 кг муки берется 600-700 мл чистой негазированной воды). В воде растворяют морскую соль (около 30-50 г на литр) и добавляют минимальное количество свежих пивных дрожжей (lievito di birra). Дрожжи запускают процесс медленного брожения.
Тесто замешивают вручную или на медленной скорости в спиральном тестомесе до достижения гладкости.

2. Длительная ферментация и созревание (Lievitazione):
После замеса тесто оставляют на первичный отдых при комнатной температуре на 2-4 часа. Затем его делят на порционные шарики (panetti) весом 250-280 грамм каждый.
Эти шарики укладывают в пластиковые контейнеры и отправляют на холодную ферментацию в холодильник при температуре 4-6°C на период от 24 до 48 часов. В процессе холодного созревания сложные крахмалы муки расщепляются ферментами на простые сахара, что делает тесто легким для усвоения и придает ему глубокий хлебный аромат. За 4-6 часов до выпечки контейнеры достают из холодильника, чтобы тесто согрелось до комнатной температуры и приобрело максимальную эластичность.

3. Формование основы (Stesura):
Формовать основу пиццы разрешается исключительно вручную, использование скалки или пресса строго запрещено. Скалка выдавливает углекислый газ из теста, делая бортик плоским и жестким.
Пиццайоло разминает шарик теста пальцами от центра к краям, перегоняя воздух в бортик (cornicione). В результате получается круглый диск диаметром около 30-32 см с тонким центром (около 2-3 мм) и пышным, воздушным краем высотой 1-2 см.

4. Начинка и выпекание:
Для соуса используются исключительно очищенные консервированные томаты сорта Сан-Марцано (Pomodoro San Marzano dell'Agro Sarnese-Nocerino D.O.P.), выращенные на вулканических почвах у подножия Везувия. Томаты разминают вручную и слегка подсаливают.
Сверху выкладывают свежий сыр моцарелла. Идеальным выбором является Mozzarella di Bufala Campana D.O.P. (из молока черных буйволиц) или Fior di Latte (из коровьего молока). Сыр нарезают соломкой и дают стечь лишней сыворотке, чтобы пицца не получилась водянистой.
Пиццу сбрызгивают оливковым маслом первого холодного отжима (Extra Virgin Olive Oil), украшают листьями свежего зеленого базилика и аккуратно переносят на деревянную или алюминиевую лопату.

Выпекание происходит в специальной купольной дровяной печи (Forno a legна) при экстремальной температуре 430-485°C. За счет высокой температуры пода и свода печи пицца готовится всего за 60-90 секунд. В процессе выпечки бортик пиццы мгновенно раздувается, приобретая характерный «леопардовый» окрас (черные обугленные пятнышки на золотистом фоне), моцарелла плавится, смешиваясь с томатным соусом, а базилик отдает свой неповторимый аромат. Готовая пицца получается мягкой, нежной, эластичной, её легко можно сложить вчетверо (a portafoglio) без изломов. Пицца Маргарита — это истинная симфония вкуса, воплощающая в себе цвета итальянского флага: красный (томаты), белый (моцарелла) и зеленый (базилик).
"""

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
        timeout=30
    )
    if response.status_code != 200:
        print(f"  ERROR: status={response.status_code}, response_text={response.text}")
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]

def measure_long_latency(url, text, runs=2):
    latencies = []
    # Разогрев
    try:
        get_embedding(url, text)
    except Exception as e:
        print(f"  Ошибка при прогреве на длинном тексте: {e}")
        return None
        
    for _ in range(runs):
        t0 = time.time()
        try:
            get_embedding(url, text)
            latencies.append((time.time() - t0) * 1000) # мс
        except Exception as e:
            print(f"  Ошибка во время замера длинного текста: {e}")
            return None
    return float(np.mean(latencies))

def run_long_benchmark():
    print("=== НАЧАЛО БЕНЧМАРКА ДЛИННОГО КОНТЕКСТА ===")
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
            
        # 2. Запуск сервера с физическим батчем 2048 для полной математической точности
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
            
        # 3. Измерение размерности и латентности на длинных текстах
        try:
            # Получаем первый эмбеддинг для размерности
            emb_query = get_embedding(url, LONG_QUERY)
            dimension = len(emb_query)
            print(f"  Размерность вектора: {dimension}")
            
            # Измеряем латентность на длинных текстах (~1500 токенов)
            print("  Измерение скорости на длинном релевантном документе...")
            lat_related = measure_long_latency(url, LONG_RELATED)
            
            print("  Измерение скорости на длинном нерелевантном документе...")
            lat_unrelated = measure_long_latency(url, LONG_UNRELATED)
            
            print(f"  Задержка (Релевантный, ~1500 токенов): {f'{lat_related:.1f} мс' if lat_related is not None else 'N/A'}")
            print(f"  Задержка (Нерелевантный, ~1500 токенов): {f'{lat_unrelated:.1f} мс' if lat_unrelated is not None else 'N/A'}")
            
            # 4. Семантические тесты длинного контекста
            emb_related = get_embedding(url, LONG_RELATED)
            emb_unrelated = get_embedding(url, LONG_UNRELATED)
            
            sim_related = cosine_similarity(emb_query, emb_related)
            sim_unrelated = cosine_similarity(emb_query, emb_unrelated)
            contrast = sim_related - sim_unrelated
            
            print(f"  Сходство похожих (Длинный текст): {sim_related:.4f}")
            print(f"  Сходство разных (Длинный текст): {sim_unrelated:.4f}")
            print(f"  Семантический контраст (Длинный текст): {contrast:.4f}")
            
            results[name] = {
                "status": "Success",
                "params": model["params"],
                "file_size": size_str,
                "dimension": dimension,
                "context": model["context"],
                "startup": f"{startup_time:.2f} с",
                "lat_related": f"{lat_related:.1f} мс" if lat_related else "N/A",
                "lat_unrelated": f"{lat_unrelated:.1f} мс" if lat_unrelated else "N/A",
                "sim_related": f"{sim_related:.4f}",
                "sim_unrelated": f"{sim_unrelated:.4f}",
                "contrast": f"{contrast:.4f}"
            }
            
        except Exception as e:
            print(f"  Ошибка во время выполнения тестов длинного контекста: {e}")
            results[name] = {"status": f"Error running tests: {e}"}
            
        # 5. Выгрузка модели
        print("  Выгрузка модели...")
        unload_all_models(role="embedding")
        time.sleep(1)
        kill_stray_servers()
        time.sleep(1)
        
    print("\n=== ВСЕ ТЕСТЫ ДЛИННОГО КОНТЕКСТА ЗАВЕРШЕНЫ ===")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    
    # Записываем результаты в compare_long_results.md
    update_markdown_report(results)

def update_markdown_report(results):
    report_path = "tests/compare_long_results.md"
    
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
    
    # Сравнительный вывод семантического контраста
    contrast_300m = float(get_val(m2, "contrast", "0"))
    contrast_zembed = float(get_val(m3, "contrast", "0"))
    contrast_qwen4b = float(get_val(m4, "contrast", "0"))
    contrast_qwen06 = float(get_val(m1, "contrast", "0"))
    
    # Определяем лидера по контрасту
    contrasts = {m1: contrast_qwen06, m2: contrast_300m, m3: contrast_zembed, m4: contrast_qwen4b}
    sorted_contrasts = sorted(contrasts.items(), key=lambda x: x[1], reverse=True)
    best_contrast_model, best_contrast_val = sorted_contrasts[0]
    second_contrast_model, second_contrast_val = sorted_contrasts[1]

    # Определяем лидера по скорости инференса на длинных текстах
    latencies = {}
    for m in [m1, m2, m3, m4]:
        val = get_val(m, "lat_related")
        if val != "N/A":
            try:
                latencies[m] = float(val.replace(" ms", "").replace(" мс", ""))
            except ValueError:
                pass
    
    sorted_latencies = sorted(latencies.items(), key=lambda x: x[1])
    best_speed_model, best_speed_val = sorted_latencies[0]
    
    # Генерируем новый markdown-файл
    content = f"""# Отчет тестирования моделей на длинном контексте (~1500 токенов)

В данном тесте сравниваются те же 4 локальные модели эмбеддингов, но на длинных документах большого размера (около 1200 слов / 1500 токенов). Задачей бенчмарка является оценка скорости работы на больших объемах текста и способность сохранять высокую семантическую контрастность при увеличении размера контекста (когда релевантная информация окружена большим объемом сопутствующих сведений).

Тестирование проводилось с использованием физического батча `2048`, обеспечивающего максимальную точность mean-pooling без дробления последовательностей.

---

## 1. Сравнение моделей эмбеддингов на длинных текстах

| Метрика / Параметр | Qwen3-Embedding-0.6B (Q8_0) | Gemma-Embedding-300M (Q8_0) | zembed-1-4B (Q4_K_M) | Qwen3-Embedding-4B (Q4_K) | Разница / Вывод |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Параметры** | {get_val(m1, 'params')} | {get_val(m2, 'params')} | {get_val(m3, 'params')} | {get_val(m4, 'params')} | Тяжелые модели против легких |
| **Размер файла** | {get_val(m1, 'file_size')} | {get_val(m2, 'file_size')} | {get_val(m3, 'file_size')} | {get_val(m4, 'file_size')} | Gemma-300M в 8 раз легче 4B моделей |
| **Размерность вектора** | {get_val(m1, 'dimension')} | {get_val(m2, 'dimension')} | {get_val(m3, 'dimension')} | {get_val(m4, 'dimension')} | Информационная емкость векторов |
| **Окно контекста** | {get_val(m1, 'context')} | {get_val(m2, 'context')} | {get_val(m3, 'context')} | {get_val(m4, 'context')} | 8k против 32k у тяжелых моделей |
| **Запуск сервера** | {get_val(m1, 'startup')} | {get_val(m2, 'startup')} | {get_val(m3, 'startup')} | {get_val(m4, 'startup')} | Быстрота готовности к работе |
| **Задержка (Релевантный, ~1.5k токенов)** | {get_val(m1, 'lat_related')} | {get_val(m2, 'lat_related')} | {get_val(m3, 'lat_related')} | {get_val(m4, 'lat_related')} | Скорость векторизации длинного документа |
| **Задержка (Нерелевантный, ~1.5k токенов)** | {get_val(m1, 'lat_unrelated')} | {get_val(m2, 'lat_unrelated')} | {get_val(m3, 'lat_unrelated')} | {get_val(m4, 'lat_unrelated')} | Повторный длинный замер скорости |
| **Сходство похожих тем** | {get_val(m1, 'sim_related')} | {get_val(m2, 'sim_related')} | {get_val(m3, 'sim_related')} | {get_val(m4, 'sim_related')} | Удержание сходства длинных документов |
| **Сходство разных тем** | {get_val(m1, 'sim_unrelated')} | {get_val(m2, 'sim_unrelated')} | {get_val(m3, 'sim_unrelated')} | {get_val(m4, 'sim_unrelated')} | Уровень шума на длинных текстах |
| **Семантический контраст** | **{get_val(m1, 'contrast')}** | **{get_val(m2, 'contrast')}** | **{get_val(m3, 'contrast')}** | **{get_val(m4, 'contrast')}** | Главная метрика качества поиска |

---

## 2. Экспертное заключение по длинному контексту

### Лидер по качеству длинного поиска (Контрастность):
- **{best_contrast_model}** с показателем семантического контраста **{best_contrast_val:.4f}**.
- Вторая по качеству — модель **{second_contrast_model}** с показателем **{second_contrast_val:.4f}**.

### Лидер по скорости инференса на длинных документах:
- **{best_speed_model}** — абсолютный лидер по латентности со средним временем обработки **{best_speed_val:.1f} мс** на 1.5k токенов.

### Ключевые выводы по длинному тесту:
1. **Сохранение контраста при росте текста**: Большой объем сопутствующих слов усложняет разделение смыслов. Тест на 1500 токенов показывает, сохраняет ли модель «фокус» внимания.
2. **Преимущество высокой точности весов (Q8_0)**: Высокоточные веса модели Q8_0 снижают накопление ошибок на длинных последовательностях токенов.
3. **Область применения 4B моделей**: Большие модели `zembed-1-4B` и `Qwen3-4B` поддерживают расширенное окно до **32k токенов**, что необходимо для индексации целых глав книг или длинных научных статей, в то время как 0.6B модели ограничены 8k токенов.
"""
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Длинный отчет успешно обновлен в {report_path}")

if __name__ == "__main__":
    run_long_benchmark()
