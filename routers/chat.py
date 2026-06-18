"""Роутер: чат-эндпоинт (SSE-стриминг)."""

import asyncio
import logging
import time

import orjson
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import config
from src.rag.prompts import get_system_prompt
from src.rag.state import RAG_POOL

from .shared import get_async_http, safe_extract_llm_response

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
    history: list[dict] = Field(default_factory=list)
    context_strategy: str | None = "sliding"
    mtp_enabled: bool | None = False


@router.post("/api/chat")
async def chat(request: ChatRequest):

    global_start_time = time.time()
    logger.debug(
        f"DEBUG: Запрос чата. Стратегия контекста: {request.context_strategy}, "
        f"Лимит токенов: {request.max_tokens}"
    )

    if not request.allowed_files:

        async def no_files():
            yield f"data: {orjson.dumps({'type': 'sources', 'sources': []}).decode()}\n\n"
            yield f"data: {orjson.dumps({'type': 'chunk', 'text': 'Пожалуйста, выберите хотя бы один источник.'}).decode()}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(no_files(), media_type="text/event-stream")

    from src.rag.prompt import build_file_context
    from src.rag.retrieval import retrieve_nodes

    query_for_rag = request.query
    nodes = []
    sources = []
    context = ""
    loop = asyncio.get_running_loop()
    _use_gguf = request.use_gguf == "true"
    skip_initial_rag = request.image_base64 and _use_gguf and request.gguf_model_path

    if query_for_rag.strip() and not skip_initial_rag:
        logger.debug(f"DEBUG: Запуск RAG поиска для: {query_for_rag[:50]}...")
        nodes = await loop.run_in_executor(
            RAG_POOL, retrieve_nodes, query_for_rag, request.notebook_id, request.allowed_files
        )
        sources, context = await loop.run_in_executor(
            RAG_POOL, build_file_context, nodes, request.notebook_id
        )
        logger.debug(f"DEBUG: RAG нашёл {len(nodes)} фрагментов.")

    active_llm = None
    use_direct_gguf = False
    if _use_gguf and request.gguf_model_path:
        use_direct_gguf = True
        from src.gguf.server import get_gguf_llm

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
            error_msg = f"Ошибка загрузки LLM: {type(e).__name__}"

            async def error_gen():
                yield f"data: {orjson.dumps({'type': 'error', 'text': error_msg}).decode()}\n\n"
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
                    http = get_async_http()
                    r_vision = await http.post(
                        f"{active_llm}/v1/chat/completions",
                        json=v_payload,
                        timeout=60,
                    )
                    extracted = safe_extract_llm_response(r_vision.json()) or ""
                    if request.query.strip():
                        search_query = f"{request.query.strip()} {extracted}"
                        query_for_rag = (
                            f"{request.query.strip()}\n\nТекст на картинке:\n{extracted}"
                        )
                    else:
                        search_query = extracted
                        query_for_rag = f"Пожалуйста, подробно ответь на вопросы или выполни задания с изображения.\n{extracted}"
                    nodes = await loop.run_in_executor(
                        RAG_POOL,
                        retrieve_nodes,
                        search_query,
                        request.notebook_id,
                        request.allowed_files,
                    )
                    sources, context = await loop.run_in_executor(
                        RAG_POOL, build_file_context, nodes, request.notebook_id
                    )
                except Exception as ve:
                    logger.error(f"DEBUG: Ошибка OCR: {ve}")

            if not query_for_rag or not query_for_rag.strip():
                query_for_rag = request.query or "Опиши содержимое"

            yield f"data: {orjson.dumps({'type': 'sources', 'sources': sources}).decode()}\n\n"

            if use_direct_gguf:
                from src.gguf.models import detect_model_family
                from src.gguf.streaming import stream_gguf_chat

                sys_prompt = (
                    get_system_prompt(request.answer_mode) + f"\n\nДоступные источники:\n{context}"
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
                messages_for_chat = [{"role": "system", "content": sys_prompt}]
                for h_msg in request.history:
                    if h_msg.get("role") in ("user", "assistant"):
                        messages_for_chat.append(
                            {"role": h_msg["role"], "content": h_msg["content"]}
                        )
                messages_for_chat.append({"role": "user", "content": user_content})
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
                                yield f"data: {orjson.dumps({'type': 'thinking_start'}).decode()}\n\n"
                                phase = "thinking"
                            else:
                                buf = buf[buf.index(OPEN_TAG) + len(OPEN_TAG) :]
                                phase = "thinking_ignore"
                        elif len(buf) > 10:
                            phase = "answer"
                            yield f"data: {orjson.dumps({'type': 'chunk', 'text': buf}).decode()}\n\n"
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
                                yield f"data: {orjson.dumps({'type': 'thinking_chunk', 'text': think_part}).decode()}\n\n"
                            yield f"data: {orjson.dumps({'type': 'thinking_done'}).decode()}\n\n"
                            phase = "answer"
                            buf = rest.lstrip("\n")
                            if buf:
                                yield f"data: {orjson.dumps({'type': 'chunk', 'text': buf}).decode()}\n\n"
                                buf = ""
                        else:
                            safe = buf[: -len(CLOSE_TAG)] if len(buf) > len(CLOSE_TAG) else ""
                            if safe:
                                yield f"data: {orjson.dumps({'type': 'thinking_chunk', 'text': safe}).decode()}\n\n"
                                buf = buf[len(safe) :]
                    elif phase == "answer":
                        yield f"data: {orjson.dumps({'type': 'chunk', 'text': buf}).decode()}\n\n"
                        buf = ""
                if buf and phase == "thinking":
                    yield f"data: {orjson.dumps({'type': 'thinking_chunk', 'text': buf}).decode()}\n\n"
                    yield f"data: {orjson.dumps({'type': 'thinking_done'}).decode()}\n\n"
                elif buf and phase == "answer":
                    yield f"data: {orjson.dumps({'type': 'chunk', 'text': buf}).decode()}\n\n"
            else:
                sys_prompt = (
                    get_system_prompt(request.answer_mode) + f"\n\nДоступные источники:\n{context}"
                )
                chat_messages = [{"role": "system", "content": sys_prompt}]
                for h_msg in request.history:
                    if h_msg.get("role") in ("user", "assistant"):
                        chat_messages.append({"role": h_msg["role"], "content": h_msg["content"]})
                chat_messages.append({"role": "user", "content": request.query})
                if active_llm is None:
                    yield f"data: {orjson.dumps({'type': 'error', 'text': 'LLM не инициализирован. Настройте URL API-модели или загрузите GGUF-модель.'}).decode()}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                queue = asyncio.Queue()

                def _sync_producer():
                    try:
                        for chunk in active_llm.stream_chat(chat_messages):
                            if chunk.delta:
                                asyncio.run_coroutine_threadsafe(queue.put(chunk.delta), loop)
                    finally:
                        asyncio.run_coroutine_threadsafe(queue.put(None), loop)

                loop.run_in_executor(RAG_POOL, _sync_producer)

                while True:
                    delta = await queue.get()
                    if delta is None:
                        break
                    token_count += 1
                    yield f"data: {orjson.dumps({'type': 'chunk', 'text': delta}).decode()}\n\n"

            elapsed = time.time() - global_start_time
            yield f"data: {orjson.dumps({'type': 'stats', 'elapsed_sec': round(elapsed, 2), 'total_tokens': token_count, 'tokens_per_sec': round(token_count / elapsed, 1) if elapsed > 0 else 0}).decode()}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error("Ошибка при обработке чата", exc_info=True)
            error_text = "Внутренняя ошибка сервера. Попробуйте позже."
            yield f"data: {orjson.dumps({'type': 'error', 'text': error_text}).decode()}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            pass

    return StreamingResponse(generate(), media_type="text/event-stream")
