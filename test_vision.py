import os
import base64
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler
import config

def get_image_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def main():
    last = config.load_last_model()
    g_path = config.resolve_model_path(last.get("gguf"))
    m_path = config.resolve_model_path(last.get("mmproj"))
    
    # Тот самый сложный кадр со схемой
    test_img = "C:/test/notebooks/04f63ba3/images/video_frame_0a90284f.jpg"
    if not os.path.exists(test_img):
        print(f"Картинка {test_img} не найдена.")
        return
    
    img_b64 = get_image_base64(test_img)
    prompt = "Проанализируй это изображение (кадр вебинара) и составь его подробное описание на русском языке."

    print(f"\n" + "="*50)
    print(f"ТЕСТ: Llava15ChatHandler")
    print("="*50)
    
    try:
        chat_handler = Llava15ChatHandler(clip_model_path=m_path)
        llm = Llama(
            model_path=g_path,
            chat_handler=chat_handler,
            n_ctx=8192,
            n_gpu_layers=-1,
            verbose=False
        )
        
        res = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are a helpful assistant that describes images in Russian."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                    ]
                }
            ],
            temperature=0.2,
            max_tokens=500
        )
        
        result = res['choices'][0]['message']['content']
        print("\n[РЕЗУЛЬТАТ]:")
        print("-" * 30)
        print(result)
        print("-" * 30)
        
        with open("test_handler_result.txt", "w", encoding="utf-8") as f:
            f.write(result)
            
    except Exception as e:
        print(f"Ошибка: {e}")

if __name__ == "__main__":
    main()
