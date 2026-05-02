import os
import base64
from llama_cpp import Llama
import config

def get_image_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def run_single_test(llm, prompt, image_b64, name):
    print(f"\n>>> ТЕСТ: {name}")
    try:
        # Пробуем разные варианты структуры content
        # Вариант А: Изображение ПЕРЕД текстом
        res = llm.create_chat_completion(
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    {"type": "text", "text": prompt}
                ]
            }],
            temperature=0.1,
            max_tokens=100
        )
        print(f"      [Результат (Img-Text)]: {res['choices'][0]['message']['content'].strip()[:200]}")
        
        # Вариант Б: Текст ПЕРЕД изображением
        res = llm.create_chat_completion(
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ]
            }],
            temperature=0.1,
            max_tokens=100
        )
        print(f"      [Результат (Text-Img)]: {res['choices'][0]['message']['content'].strip()[:200]}")
    except Exception as e:
        print(f"      [ОШИБКА]: {e}")

def main():
    last = config.load_last_model()
    g_path = config.resolve_model_path(last.get("gguf"))
    m_path = config.resolve_model_path(last.get("mmproj"))
    
    # Берем тестовую картинку (вебинар из ваших логов)
    test_img = "C:/test/notebooks/04f63ba3/images/video_frame_161d4224.jpg"
    if not os.path.exists(test_img):
        print(f"Картинка {test_img} не найдена. Тест невозможен.")
        return
    
    img_b64 = get_image_base64(test_img)
    prompt = "Опиши это изображение одним предложением на русском языке."

    formats = ["qwen", "chatml", None]
    
    for fmt in formats:
        print(f"\n" + "="*50)
        print(f"ЗАГРУЗКА МОДЕЛИ С FORMAT: {fmt}")
        print("="*50)
        try:
            llm = Llama(
                model_path=g_path,
                clip_model_path=m_path,
                chat_format=fmt,
                n_ctx=4096,
                n_gpu_layers=-1,
                verbose=False
            )
            run_single_test(llm, prompt, img_b64, f"Format={fmt}")
            del llm # Очистка для следующего теста
        except Exception as e:
            print(f"Не удалось загрузить модель с форматом {fmt}: {e}")

if __name__ == "__main__":
    main()
