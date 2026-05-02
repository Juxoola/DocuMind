import os
import sys
import base64
from llama_cpp import Llama
import config

def get_image_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def test_vision(image_path, gguf_path, mmproj_path):
    print(f"--- Тестирование Vision ---")
    print(f"Модель: {gguf_path}")
    print(f"Проектор: {mmproj_path}")
    print(f"Картинка: {image_path}")
    
    if not os.path.exists(gguf_path) or not os.path.exists(mmproj_path):
        print("Ошибка: Файлы модели не найдены!")
        return

    print("Загрузка модели...")
    llm = Llama(
        model_path=gguf_path,
        chat_format="chatml",
        clip_model_path=mmproj_path,
        n_ctx=4096,
        n_gpu_layers=-1,
        verbose=False
    )

    prompt = """Проанализируй это изображение (кадр из видео или слайд презентации) и составь его подробное описание на русском языке для системы поиска.

1. ТЕКСТ: Выпиши весь видимый текст, заголовки и важные подписи.
2. ГРАФИКА: Опиши схемы, таблицы или графики, если они есть.
3. ВИЗУАЛ: Опиши ключевые объекты, людей или обстановку.
4. СМЫСЛ: Кратко сформулируй основную тему этого кадра.

Пиши объективно и только по делу."""

    print("Кодирование изображения...")
    base64_data = get_image_base64(image_path)
    
    print("Генерация описания...")
    response = llm.create_chat_completion(
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_data}"}}
            ]
        }],
        temperature=0.2,
        max_tokens=500
    )

    result = response["choices"][0]["message"]["content"]
    print("\n[РЕЗУЛЬТАТ]:")
    print("-" * 30)
    print(result)
    print("-" * 30)

if __name__ == "__main__":
    # Берем последнюю использованную модель
    last = config.load_last_model()
    g = last.get("gguf")
    m = last.get("mmproj")
    
    if not g or not m:
        print("Ошибка: Нет сохраненной конфигурации модели. Сначала запустите ингестию через UI.")
        sys.exit(1)
        
    # Ищем тестовую картинку в первой попавшейся папке images ноутбуков
    test_img = None
    if os.path.exists(config.NOTEBOOKS_DIR):
        for nb_id in os.listdir(config.NOTEBOOKS_DIR):
            img_dir = os.path.join(config.NOTEBOOKS_DIR, nb_id, "images")
            if os.path.exists(img_dir):
                files = os.listdir(img_dir)
                if files:
                    test_img = os.path.join(img_dir, files[0])
                    break
    
    if not test_img:
        print("Ошибка: Не найдено ни одного изображения для теста в папках ноутбуков.")
    else:
        test_vision(test_img, config.resolve_model_path(g), config.resolve_model_path(m))
