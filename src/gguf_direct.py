import os
import re
import llama_cpp
from typing import Dict, Optional, Tuple
import config

# Отключаем лишние логи llama-cpp
try:
    def dummy_log_callback(level, message, user_data):
        pass
    llama_cpp.llama_log_set(dummy_log_callback, None)
except: pass

# Глобальный кэш загруженных моделей
_model_cache: Dict[str, llama_cpp.Llama] = {}


# ── Определение семейства модели ─────────────────────────────────────────────

def detect_model_family(gguf_path: str) -> str:
    """
    Определяет семейство модели по имени файла.
    Возвращает: 'qwen' | 'gemma4' | 'gemma3' | 'deepseek' | 'llama' | 'generic'
    """
    name = os.path.basename(gguf_path).lower()
    if any(x in name for x in ["qwen", "qwq"]):
        return "qwen"
    if "gemma" in name:
        # Gemma 4 вышла в апреле 2025 — новый формат токенов
        if any(x in name for x in ["gemma-4", "gemma4", "gemma_4", "-4b", "-4e", "e4b"]):
            return "gemma4"
        return "gemma3"  # Gemma 1/2/3 — старый формат
    if any(x in name for x in ["deepseek", "-r1", "_r1"]):
        return "deepseek"
    if "llama" in name:
        return "llama"
    return "generic"


# ── Шаблоны форматов моделей ──────────────────────────────────────────────────

# Стоп-токены для каждого семейства
STOP_TOKENS = {
    "qwen":     ["<|im_end|>", "<|endoftext|>"],
    "gemma4":   ["<turn|>", "<eos>"],        # Gemma 4: новый формат
    "gemma3":   ["<end_of_turn>", "<eos>"],  # Gemma 1/2/3
    "deepseek": ["<|im_end|>", "</s>"],
    "llama":    ["<|eot_id|>", "<|end_of_text|>"],
    "generic":  ["<|im_end|>", "</s>", "<|endoftext|>"],
}


def build_thinking_prompt(messages: list, enable_thinking: bool, model_family: str) -> str:
    """
    Универсальный сборщик промпта с поддержкой thinking mode.

    Стратегия:
      - Qwen / DeepSeek / generic (ChatML): prefill '<think>\\n' включает думание.
      - Gemma 4: новый формат <|turn|>/<turn|>, thinking через <|think|>.
      - Gemma 1/2/3: старый <start_of_turn>/<end_of_turn>, thinking не нативный.
      - Llama 3: llama-3 шаблон.

    Args:
        messages: список dict с ключами 'role' и 'content'
        enable_thinking: включить ли think-блок
        model_family: результат detect_model_family()

    Returns:
        Готовый строковый промпт для передачи в llm(prompt=...)
    """
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    # Поддержка мультимодального content (list of dicts)
    user_msg = next((m for m in messages if m["role"] == "user"), None)
    if user_msg:
        content = user_msg["content"]
        if isinstance(content, list):
            # Берём только текстовые части — изображения передаются отдельно
            user = " ".join(p["text"] for p in content if p.get("type") == "text")
        else:
            user = content
    else:
        user = ""

    if model_family in ("qwen", "deepseek", "generic"):
        # ChatML формат
        sys_block = f"<|im_start|>system\n{system}<|im_end|>\n" if system else ""
        think_suffix = "<think>\n" if enable_thinking else "<think>\n\n</think>\n"
        return (
            f"{sys_block}"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n{think_suffix}"
        )

    elif model_family == "gemma4":
        # Gemma 4 (2025): формат <|turn>role\ncontent<turn|>\n
        # Thinking включается через <|think|> в системном блоке.
        # Модель выводит рассуждения в формате: <channel|>...<|channel>
        if enable_thinking:
            # <|think|> идёт в начало системного блока
            sys_content = f"<|think|>{system}" if system else "<|think|>"
            sys_block = f"<|turn>system\n{sys_content}<turn|>\n"
        else:
            sys_block = f"<|turn>system\n{system}<turn|>\n" if system else ""
        return (
            f"{sys_block}"
            f"<|turn>user\n{user}<turn|>\n"
            f"<|turn>model\n"
        )

    elif model_family == "gemma3":
        # Gemma 1/2/3: старый формат <start_of_turn>/<end_of_turn>
        if system:
            return (
                f"<start_of_turn>user\n{system}\n\n{user}<end_of_turn>\n"
                f"<start_of_turn>model\n"
            )
        return f"<start_of_turn>user\n{user}<end_of_turn>\n<start_of_turn>model\n"

    elif model_family == "llama":
        # Llama 3 Instruct формат
        sys_block = f"<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>\n" if system else ""
        return (
            f"<|begin_of_text|>{sys_block}"
            f"<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>\n"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    # Fallback — ChatML
    sys_block = f"<|im_start|>system\n{system}<|im_end|>\n" if system else ""
    think_suffix = "<think>\n" if enable_thinking else "<think>\n\n</think>\n"
    return (
        f"{sys_block}"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n{think_suffix}"
    )


def parse_thinking_response(text: str) -> Tuple[str, str]:
    """
    Разбирает ответ модели на части: мышление и финальный ответ.

    Returns:
        (thinking_content, final_answer)
        thinking_content будет пустой строкой, если think-блока не было.
    """
    match = re.search(r"<think>(.*?)</think>(.*)", text, re.DOTALL)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return "", text.strip()


def get_stop_tokens(model_family: str) -> list:
    """Возвращает список стоп-токенов для данного семейства моделей."""
    return STOP_TOKENS.get(model_family, STOP_TOKENS["generic"])


# ── Загрузка модели ───────────────────────────────────────────────────────────

def get_gguf_llm(
    gguf_path: str,
    mmproj_path: str = None,
    temperature: float = 0.3,
    ctx_size: int = 8192,
    gpu_layers: int = -1,
    n_threads: int = None,
    n_batch: int = 2048,
    flash_attn: bool = False,
    max_tokens: int = 2048,
    type_k: int = 2,
    type_v: int = 2,
    enable_thinking: bool = True,
) -> llama_cpp.Llama:
    """
    Загружает GGUF модель через прямой API llama-cpp-python.
    Кэширует по пути модели (thinking не влияет на загрузку — он управляется промптом).
    """
    gguf_path = os.path.normpath(gguf_path)
    cache_key = gguf_path  # thinking не требует перезагрузки модели

    if cache_key in _model_cache:
        family = detect_model_family(gguf_path)
        print(f"[GGUF Direct] Кэш: {os.path.basename(gguf_path)} | семейство: {family} | thinking: {enable_thinking}")
        return _model_cache[cache_key]

    from src.rag_pipeline import unload_rag_models
    unload_rag_models()

    if not os.path.exists(gguf_path):
        raise FileNotFoundError(f"GGUF модель не найдена: {gguf_path}")

    family = detect_model_family(gguf_path)
    print(f"[GGUF Direct] Загрузка: {os.path.basename(gguf_path)} | семейство: {family} | thinking: {enable_thinking}")

    llama_kwargs = {
        "model_path": gguf_path,
        "n_ctx": ctx_size,
        "n_gpu_layers": gpu_layers,
        "n_threads": n_threads if n_threads is not None else config.GGUF_THREADS,
        "n_batch": n_batch if n_batch is not None else 2048,
        "flash_attn": flash_attn,
        "type_k": type_k,
        "type_v": type_v,
        "verbose": False,
    }

    if mmproj_path and os.path.exists(mmproj_path):
        print(f"[GGUF Direct] mmproj найден: {os.path.basename(mmproj_path)}")

    llm = llama_cpp.Llama(**llama_kwargs)
    _model_cache[cache_key] = llm
    return llm


# ── Высокоуровневый стриминг ──────────────────────────────────────────────────

def stream_gguf_chat(
    llm: llama_cpp.Llama,
    messages: list,
    enable_thinking: bool,
    max_tokens: int,
    temperature: float,
    repeat_penalty: float,
    top_p: float,
    min_p: float,
):
    """
    Генератор дельт текста.

    Использует build_thinking_prompt для построения правильного промпта
    (семейство определяется автоматически по llm.model_path),
    затем вызывает llm(prompt=..., stream=True).

    Интерфейс в main.py не знает ни о семействах, ни о шаблонах — всё здесь.
    """
    model_path = getattr(llm, "model_path", "") or ""
    family = detect_model_family(model_path)
    prompt = build_thinking_prompt(messages, enable_thinking, family)
    stop_tokens = get_stop_tokens(family)

    print(f"[stream_gguf_chat] семейство={family} | thinking={enable_thinking}")

    for chunk in llm(
        prompt=prompt,
        stream=True,
        max_tokens=max_tokens,
        temperature=temperature,
        repeat_penalty=repeat_penalty,
        top_p=top_p,
        min_p=min_p,
        stop=stop_tokens,
    ):
        delta = chunk["choices"][0].get("text", "")
        if delta:
            yield delta


# ── Управление кэшем ──────────────────────────────────────────────────────────

def unload_all_models():
    """Полная очистка всех загруженных моделей из кэша и VRAM."""
    global _model_cache
    for path, model in _model_cache.items():
        print(f"[GGUF Direct] Выгрузка: {os.path.basename(path)}")
        del model
    _model_cache = {}
    import gc
    gc.collect()


def get_loaded_models():
    """Возвращает список путей к загруженным моделям."""
    return list(_model_cache.keys())
