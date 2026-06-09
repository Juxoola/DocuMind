"""
Роутер: чат-эндпоинт (SSE-стриминг).
"""

import json
import logging
import time
import traceback

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import config

from .shared import _http_session

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    query: str
    allowed_files: list[str]
    max_tokens: int = 2048
    notebook_id: str
    thinking_mode: bool = False
    llm_url: str | None = None
    llm_api_key: str | None = config.LLM_DEFAULT_API_KEY
    llm_model: str | None = config.LLM_DEFAULT_MODEL
    image_base64: str | None = None
    answer_mode: str | None = "concise"
    gguf_kv_quant: int | None = 2
    repeat_penalty: float | None = 1.1
    top_p: float | None = 0.95
    min_p: float | None = 0.05
    use_gguf: str | None = None
    gguf_model_path: str | None = None
    gguf_mmproj_path: str | None = None
    gguf_temperature: float | None = 0.7
    gguf_ctx_size: int | None = 32768
    gguf_gpu_layers: int | None = -1
    gguf_threads: int | None = 8
    gguf_batch_size: int | None = 512
    gguf_ubatch_size: int | None = 256
    gguf_flash_attn: str | None = "true"
    thinking_budget: int | None = 1024
    context_strategy: str | None = "sliding"
    mtp_enabled: bool | None = False


@router.post("/api/chat")
async def chat(request: ChatRequest):
    import asyncio

    global_start_time = time.time()
    logger.debug(
        f"DEBUG: Запрос чата. Стратегия контекста: {request.context_strategy}, "
        f"Лимит токенов: {request.max_tokens}"
    )

    if not request.allowed_files:

        async def no_files():
            yield f"data: {json.dumps({'type': 'sources', 'sources': []})}\n\n"
            yield f"data: {json.dumps({'type': 'chunk', 'text': 'Пожалуйста, выберите хотя бы один источник.'})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(no_files(), media_type="text/event-stream")

    # 1. RAG
    from src.rag_pipeline import build_file_context, retrieve_nodes

    query_for_rag = request.query
    nodes = []
    sources = []
    context = ""
    skip_initial_rag = (
        request.image_base64 and request.use_gguf == "true" and request.gguf_model_path
    )

    if query_for_rag.strip() and not skip_initial_rag:
        logger.debug(f"DEBUG: Запуск RAG поиска для: {query_for_rag[:50]}...")
        nodes = await asyncio.to_thread(
            retrieve_nodes, query_for_rag, request.notebook_id, request.allowed_files
        )
        sources, context = await asyncio.to_thread(build_file_context, nodes, request.notebook_id)
        logger.debug(f"DEBUG: RAG нашёл {len(nodes)} фрагментов.")

    # 2. LLM
    active_llm = None
    use_direct_gguf = False
    if request.use_gguf == "true" and request.gguf_model_path:
        use_direct_gguf = True
        from src.gguf_direct import get_gguf_llm

        try:
            active_llm = await asyncio.to_thread(
                get_gguf_llm,
                gguf_path=request.gguf_model_path,
                mmproj_path=request.gguf_mmproj_path or None,
                temperature=request.gguf_temperature,
                ctx_size=request.gguf_ctx_size,
                gpu_layers=request.gguf_gpu_layers,
                n_threads=request.gguf_threads,
                n_batch=request.gguf_batch_size,
                n_ubatch=request.gguf_ubatch_size,
                flash_attn=(request.gguf_flash_attn == "true"),
                max_tokens=request.max_tokens,
                type_k=request.gguf_kv_quant,
                type_v=request.gguf_kv_quant,
                enable_thinking=request.thinking_mode,
                thinking_budget=request.thinking_budget,
                mtp_enabled=request.mtp_enabled,
            )
        except Exception as e:
            error_msg = f"Ошибка загрузки LLM: {e!s}"

            async def error_gen():
                yield f"data: {json.dumps({'type': 'error', 'text': error_msg}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(error_gen(), media_type="text/event-stream")
    elif request.llm_url:
        from llama_index.llms.openai import OpenAI

        active_llm = OpenAI(
            api_base=request.llm_url,
            api_key=request.llm_api_key or config.LLM_DEFAULT_API_KEY,
            model=request.llm_model or config.LLM_DEFAULT_MODEL,
            temperature=0.1,
            max_tokens=request.max_tokens,
        )
    else:
        active_llm = None

    # 3. Генерация ответа
    async def generate():
        nonlocal query_for_rag, sources, context
        token_count = 0
        try:
            if request.image_base64 and use_direct_gguf:
                vision_messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{request.image_base64}"
                                },
                            },
                            {
                                "type": "text",
                                "text": "Сделай точный OCR всего текста на изображении.",
                            },
                        ],
                    }
                ]
                try:
                    v_payload = {"messages": vision_messages, "stream": False, "max_tokens": 1024}
                    r_vision = await asyncio.to_thread(
                        _http_session.post,
                        f"{active_llm}/v1/chat/completions",
                        json=v_payload,
                        timeout=60,
                    )
                    extracted = r_vision.json()["choices"][0]["message"]["content"].strip()
                    if request.query.strip():
                        search_query = f"{request.query.strip()} {extracted}"
                        query_for_rag = (
                            f"{request.query.strip()}\n\nТекст на картинке:\n{extracted}"
                        )
                    else:
                        search_query = extracted
                        query_for_rag = f"Пожалуйста, подробно ответь на вопросы или выполни задания с изображения.\n{extracted}"
                    nodes = await asyncio.to_thread(
                        retrieve_nodes, search_query, request.notebook_id, request.allowed_files
                    )
                    sources, context = await asyncio.to_thread(
                        build_file_context, nodes, request.notebook_id
                    )
                except Exception as ve:
                    logger.error(f"DEBUG: Ошибка OCR: {ve}")

            if not query_for_rag or not query_for_rag.strip():
                query_for_rag = request.query or "Опиши содержимое"

            yield f"data: {json.dumps({'type': 'sources', 'sources': sources}, ensure_ascii=False)}\n\n"

            if use_direct_gguf:
                from src.gguf_direct import detect_model_family, stream_gguf_chat

                sys_prompt = (
                    config.get_system_prompt(request.answer_mode)
                    + f"\n\nДоступные источники:\n{context}"
                )
                if request.image_base64:
                    user_content = [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{request.image_base64}"},
                        },
                        {"type": "text", "text": query_for_rag},
                    ]
                else:
                    user_content = query_for_rag
                messages_for_chat = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_content},
                ]
                model_family = detect_model_family(request.gguf_model_path)
                OPEN_TAG, CLOSE_TAG = (
                    ("<|channel|>", "<channel|>")
                    if model_family == "gemma4"
                    else ("<think>", "</think>")
                )
                phase = "think_detect"
                buf = ""
                async_gen = stream_gguf_chat(
                    llm_url=active_llm,
                    messages=messages_for_chat,
                    enable_thinking=request.thinking_mode,
                    max_tokens=request.max_tokens,
                    temperature=request.gguf_temperature,
                    repeat_penalty=request.repeat_penalty,
                    top_p=request.top_p,
                    min_p=request.min_p,
                    model_family=model_family,
                )
                async for delta in async_gen:
                    if not delta:
                        continue
                    token_count += 1
                    buf += delta
                    if phase == "think_detect":
                        if OPEN_TAG in buf:
                            if request.thinking_mode:
                                buf = buf[buf.index(OPEN_TAG) + len(OPEN_TAG) :]
                                yield f"data: {json.dumps({'type': 'thinking_start'}, ensure_ascii=False)}\n\n"
                                phase = "thinking"
                            else:
                                buf = buf[buf.index(OPEN_TAG) + len(OPEN_TAG) :]
                                phase = "thinking_ignore"
                        elif len(buf) > 10:
                            phase = "answer"
                            yield f"data: {json.dumps({'type': 'chunk', 'text': buf}, ensure_ascii=False)}\n\n"
                            buf = ""
                    if phase == "thinking_ignore":
                        if CLOSE_TAG in buf:
                            _, _, rest = buf.partition(CLOSE_TAG)
                            buf = rest.lstrip("\n")
                            phase = "answer"
                        else:
                            if len(buf) > len(CLOSE_TAG):
                                buf = buf[-len(CLOSE_TAG) :]
                            continue
                    if phase == "thinking":
                        if CLOSE_TAG in buf:
                            think_part, _, rest = buf.partition(CLOSE_TAG)
                            if think_part:
                                yield f"data: {json.dumps({'type': 'thinking_chunk', 'text': think_part}, ensure_ascii=False)}\n\n"
                            yield f"data: {json.dumps({'type': 'thinking_done'}, ensure_ascii=False)}\n\n"
                            phase = "answer"
                            buf = rest.lstrip("\n")
                            if buf:
                                yield f"data: {json.dumps({'type': 'chunk', 'text': buf}, ensure_ascii=False)}\n\n"
                                buf = ""
                        else:
                            safe = buf[: -len(CLOSE_TAG)] if len(buf) > len(CLOSE_TAG) else ""
                            if safe:
                                yield f"data: {json.dumps({'type': 'thinking_chunk', 'text': safe}, ensure_ascii=False)}\n\n"
                                buf = buf[len(safe) :]
                    elif phase == "answer":
                        yield f"data: {json.dumps({'type': 'chunk', 'text': buf}, ensure_ascii=False)}\n\n"
                        buf = ""
                if buf and phase == "thinking":
                    yield f"data: {json.dumps({'type': 'thinking_chunk', 'text': buf}, ensure_ascii=False)}\n\n"
                    yield f"data: {json.dumps({'type': 'thinking_done'}, ensure_ascii=False)}\n\n"
                elif buf and phase == "answer":
                    yield f"data: {json.dumps({'type': 'chunk', 'text': buf}, ensure_ascii=False)}\n\n"
            else:
                from src.rag_pipeline import make_prompt

                prompt = make_prompt(
                    request.query,
                    context,
                    thinking_mode=request.thinking_mode,
                    max_tokens=request.max_tokens,
                    answer_mode=request.answer_mode,
                )
                for chunk in active_llm.stream_complete(prompt):
                    if chunk.delta:
                        token_count += 1
                        yield f"data: {json.dumps({'type': 'chunk', 'text': chunk.delta}, ensure_ascii=False)}\n\n"

            elapsed = time.time() - global_start_time
            yield f"data: {json.dumps({'type': 'stats', 'elapsed_sec': round(elapsed, 2), 'total_tokens': token_count, 'tokens_per_sec': round(token_count / elapsed, 1) if elapsed > 0 else 0})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'text': str(e)}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            if use_direct_gguf and active_llm:
                try:
                    _http_session.post(f"{active_llm}/slots/0/clear", timeout=1)
                except Exception:
                    pass

    return StreamingResponse(generate(), media_type="text/event-stream")
