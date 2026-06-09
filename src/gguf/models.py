"""Определение семейства модели по имени файла.
"""

# Файл: models.py — определяет семейство модели по имени GGUF-файла.
# Используется для выбора тегов разметки thinking (think/channel)
# и других особенностей обработки в stream_gguf_chat.

def detect_model_family(gguf_path: str) -> str:

    name = gguf_path.lower()
    # Проверка по ключевым словам в имени файла:
    # Qwen/QwQ → "qwen"; Gemma (4 vs 3) → "gemma4"/"gemma3";
    # DeepSeek/R1 → "deepseek"; Llama → "llama"; иначе "generic".
    if any(x in name for x in ["qwen", "qwq"]):
        return "qwen"
    if "gemma" in name:
        if any(
            x in name
            for x in ["gemma-4", "gemma4", "gemma_4", "-4b", "-4e", "e4b"]
        ):
            return "gemma4"
        return "gemma3"
    if any(x in name for x in ["deepseek", "-r1", "_r1"]):
        return "deepseek"
    if "llama" in name:
        return "llama"
    return "generic"
