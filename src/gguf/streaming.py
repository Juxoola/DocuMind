"""Асинхронный стриминг чат-ответов через llama-server API."""

# Файл: streaming.py — асинхронный генератор для получения ответов
# от llama-server через SSE (Server-Sent Events). Разбирает поток
# data: событий, извлекает reasoning_content и content, оборачивает
# мышление в теги <think>...</think> (или <|channel|> для gemma4).

import json
import logging

logger = logging.getLogger(__name__)


async def stream_gguf_chat(
    llm_url: str,
    messages: list,
    enable_thinking: bool,
    max_tokens: int,
    temperature: float,
    repeat_penalty: float,
    top_p: float,
    min_p: float,
    model_family: str = "generic",
):

    import httpx

    # Формируем запрос к /v1/chat/completions со stream=True.
    # Для gemma4 используются теги <|channel|>, для остальных — <think>.
    payload = {
        "messages": messages,
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "repeat_penalty": repeat_penalty,
        "top_p": top_p,
        "min_p": min_p,
    }

    OPEN_TAG, CLOSE_TAG = (
        ("<|channel|>", "<channel|>") if model_family == "gemma4" else ("<think>", "</think>")
    )

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", f"{llm_url}/v1/chat/completions", json=payload) as r:
                r.raise_for_status()
                is_thinking = False

                # Парсинг SSE-потока: строки вида "data: {...}".
                # reasoning_content → оборачиваем в OPEN_TAG/CLOSE_TAG,
                # content — выдаём как есть.
                async for line in r.aiter_lines():
                    if line:
                        line_str = line
                        if line_str.startswith("data: "):
                            if line_str == "data: [DONE]":
                                break
                            try:
                                data = json.loads(line_str[6:])
                                delta = data["choices"][0]["delta"]

                                reasoning = delta.get("reasoning_content", "")
                                if reasoning:
                                    if not is_thinking:
                                        yield OPEN_TAG
                                        is_thinking = True
                                    yield reasoning
                                    continue

                                content = delta.get("content", "")
                                if content:
                                    if is_thinking:
                                        yield CLOSE_TAG
                                        is_thinking = False
                                    yield content
                            except Exception as e:
                                logger.debug(f"Ошибка парсинга SSE: {e}")
                                continue

                if is_thinking:
                    yield CLOSE_TAG

    except Exception as e:
        logger.error(f"[GGUF Stream] Ошибка: {e}")
        yield f"Ошибка связи с сервером: {e}"
