"""Vision: описание изображений через llama-server или LM Studio."""

import asyncio
import base64
import logging
import re

import config
from routers.shared import get_async_http, safe_extract_llm_response
from src.gguf.server import get_vision_server
from src.ingestion.utils import cleanup_gpu

logger = logging.getLogger(__name__)


def get_image_base64(image_path, max_dimension=1568):
    try:
        import io

        from PIL import Image

        with Image.open(image_path) as img:
            if max(img.size) > max_dimension:
                ratio = max_dimension / max(img.size)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img = img.resize(new_size, Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            return base64.b64encode(buf.getvalue()).decode("utf-8")
    except ImportError:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")


def make_vision_message(base64_data: str, text: str = "") -> list:
    msg = [{"type": "text", "text": text}] if text else []
    msg.append(
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_data}"},
        }
    )
    return msg


async def get_vision_url(llm_settings, progress_cb=None):
    if not llm_settings or not llm_settings.get("use_gguf_direct"):
        return None

    v_model = llm_settings.get("vision_model_path") or llm_settings.get("gguf_model_path")
    if not v_model:
        return None
    try:
        if progress_cb:
            progress_cb(60, "Запуск Vision-сервера (ленивая загрузка)...")
        await asyncio.to_thread(cleanup_gpu)

        v_mmproj = llm_settings.get("vision_mmproj_path") or llm_settings.get("gguf_mmproj_path")
        g_path = config.resolve_model_path(v_model)
        m_path = config.resolve_model_path(v_mmproj)
        v_ctx = int(llm_settings.get("vision_ctx_size") or config.GGUF_CTX_SIZE)
        v_gl = int(llm_settings.get("vision_gpu_layers") or -1)
        v_b = int(llm_settings.get("vision_batch_size") or 512)
        v_ub = int(llm_settings.get("vision_ubatch_size") or 256)
        v_fa = llm_settings.get("vision_flash_attn") == "true"
        v_kv = int(llm_settings.get("vision_kv_quant") or 2)
        v_conc = int(llm_settings.get("vision_concurrency") or config.VISION_CONCURRENCY)
        v_mtp = bool(llm_settings.get("vision_mtp_enabled", False))
        v_max_tokens = int(llm_settings.get("vision_max_tokens") or 4096)
        v_threads = int(llm_settings.get("vision_threads") or 0)

        return await get_vision_server(
            gguf_path=g_path,
            mmproj_path=m_path,
            ctx_size=4096,
            gpu_layers=v_gl,
            n_batch=v_b,
            n_ubatch=v_ub,
            flash_attn=v_fa,
            n_parallel=1,
            n_threads=v_threads or None,
            custom_args=[],
        )
    except Exception as e:
        logger.error(f"[Vision] Ошибка ленивого запуска: {e}")
        return None


def _clean_think_tags(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|.*?\|>", "", text)
    text = re.sub(r"<start_of_turn>|<end_of_turn>", "", text)
    return text.strip()


async def describe_image_with_lmstudio(
    image_path, llm_settings=None, existing_llm_url=None, cancel_check=None
):

    prompt = """Проведи детальный анализ изображения. 

1. ГРАФИКА: 
   - Если есть схемы, диаграммы или таблицы, опиши их структуру максимально подробно. 
   - Используй таблицу (Узлы, Связи, Пояснения) для сложных схем.
   - Описывай каждый элемент и связь только ОДИН раз. Не дублируй информацию.
   - Укажи назначение ключевых обозначений, если они присутствуют.

2. ТЕКСТ: 
   - Выполни OCR только того текста, который является текстовым блоком (заголовки, подписи к рисункам, абзацы, списки, вопросы). 
   - НЕ описывай элементы графики в этом разделе, если они уже были детально описаны в п.1.
   - Сохраняй исходную структуру текста.

3. КОНТЕКСТ: 
   - В 1-2 предложениях опиши общую суть и назначение страницы.

Пиши технически точно, лаконично, без лишних вводных фраз и пояснений процесса."""

    if existing_llm_url:
        for attempt in range(2):
            if cancel_check and cancel_check():
                logger.info("[Ingestion] Отмена: vision запрос пропущен")
                return "Изображение без описания."
            try:
                v_temp = float(llm_settings.get("vision_temperature") or config.VISION_TEMPERATURE)
                v_max = int(llm_settings.get("vision_max_tokens") or 4096)
                v_r_pen = float(
                    llm_settings.get("vision_repeat_penalty") or config.VISION_REPEAT_PENALTY
                )
                v_top_p = float(llm_settings.get("vision_top_p") or config.VISION_TOP_P)
                v_min_p = float(llm_settings.get("vision_min_p") or config.VISION_MIN_P)
                v_pres = float(llm_settings.get("vision_presence_penalty") or 0.0)
                v_freq = float(llm_settings.get("vision_frequency_penalty") or 0.0)

                payload = {
                    "messages": [
                        {
                            "role": "user",
                            "content": make_vision_message(get_image_base64(image_path), prompt),
                        }
                    ],
                    "temperature": v_temp,
                    "max_tokens": v_max,
                    "repeat_penalty": v_r_pen,
                    "top_p": v_top_p,
                    "min_p": v_min_p,
                    "presence_penalty": v_pres,
                    "frequency_penalty": v_freq,
                    "cache_prompt": False,
                    "slot_id": -1,
                }
                client = get_async_http()
                r = await client.post(
                    f"{existing_llm_url}/v1/chat/completions",
                    json=payload,
                    timeout=30,
                )
                del payload
                if cancel_check and cancel_check():
                    logger.info("[Ingestion] Отмена: vision запрос прерван")
                    return "Изображение без описания."
                if r.status_code == 200:
                    res = r.json()
                    # Освобождаем слот llama-server сразу после ответа
                    slot_id = res.get("slot", 0)
                    try:
                        await client.post(
                            f"{existing_llm_url}/slots/{slot_id}?action=erase",
                            timeout=5,
                        )
                    except Exception:
                        pass
                    if "choices" in res:
                        ans = safe_extract_llm_response(res) or "Ошибка извлечения ответа"
                        reason = res.get("choices", [{}])[0].get("finish_reason")
                        ans = _clean_think_tags(ans)
                        logger.info(
                            f"[Ingestion] Описание получено ({len(ans)} симв.). Причина завершения: {reason}"
                        )
                        return ans
                elif r.status_code == 500:
                    logger.info(f"[Ingestion] GGUF 500 (попытка {attempt + 1}). Повтор...")
                    await asyncio.sleep(2)
                    continue
                else:
                    logger.error(f"[Ingestion] Ошибка GGUF {r.status_code}: {r.text}")
            except Exception as e:
                logger.error(f"[Ingestion] Ошибка запроса (попытка {attempt + 1}): {e}")
                await asyncio.sleep(1)
        return "Ошибка анализа после всех попыток"

    api_url = (llm_settings.get("llm_url") if llm_settings else None) or config.LM_STUDIO_URL
    if cancel_check and cancel_check():
        return "Изображение без описания."
    api_key = (
        llm_settings.get("llm_api_key") if llm_settings else None
    ) or config.LLM_DEFAULT_API_KEY
    model_name = (
        llm_settings.get("llm_model") if llm_settings else None
    ) or config.LLM_DEFAULT_MODEL
    try:
        v_temp = float(llm_settings.get("vision_temperature") or 0.2)
        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{get_image_base64(image_path)}"
                            },
                        },
                    ],
                }
            ],
            "temperature": v_temp,
        }
        client = get_async_http()
        r = await client.post(
            f"{api_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
            timeout=30,
        )
        ans = safe_extract_llm_response(r.json()) or "Ошибка извлечения ответа"
        return _clean_think_tags(ans)
    except Exception as e:
        logger.warning(f"Ошибка резервного Vision через LM Studio: {e}")
        return "Изображение без описания."
