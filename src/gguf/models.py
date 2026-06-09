"""Определение семейства модели по имени файла.
"""


def detect_model_family(gguf_path: str) -> str:
    """Определяет семейство модели по имени файла.

    Returns: qwen | gemma3 | gemma4 | deepseek | llama | generic
    """
    name = gguf_path.lower()
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
