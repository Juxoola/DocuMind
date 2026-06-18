"""Асинхронный стриминг чат-ответов через llama-server API."""

import json
import logging

logger = logging.getLogger(__name__)

_stream_client: "httpx.AsyncClient | None" = None


async def get_stream_client() -> "httpx.AsyncClient":
    global _stream_client
    if _stream_client is None or _stream_client.is_closed:
        import httpx

        _stream_client = httpx.AsyncClient(timeout=60.0)
    return _stream_client


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
        client = await get_stream_client()
        async with client.stream("POST", f"{llm_url}/v1/chat/completions", json=payload) as r:
            r.raise_for_status()
            is_thinking = False

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
