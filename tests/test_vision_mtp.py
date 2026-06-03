"""
Тест совместимости MTP (--spec-type draft-mtp) с vision-моделями.

Запускает llama-server в нескольких режимах и отправляет запрос с картинкой:
  1. Vision + MTP ON
  2. Vision + MTP OFF
  3. Text-only + MTP ON (контроль, что MTP вообще работает)
  4. Text-only + MTP OFF

Каждый сценарий изолирован: сервер запускается, опрашивается, гасится.
Выводит сводную таблицу: scenario | status | elapsed | tokens | note
"""
import os
import sys
import time
import json
import base64
import socket
import subprocess
import io

import requests
from PIL import Image, ImageDraw, ImageFont

GGUF_PATH = r"F:\llm\unsloth\Qwen3.5-4B-MTP-GGUF\Qwen3.5-4B-IQ4_XS.gguf"
MMPROJ_PATH = r"F:\llm\unsloth\Qwen3.5-4B-MTP-GGUF\mmproj-F32.gguf"
SERVER_EXE = r"C:\test\bin\llama-server.exe"
CTX_SIZE = 8192
GPU_LAYERS = -1
BATCH = 512
UBATCH = 256
TIMEOUT_READY = 90
TIMEOUT_REQUEST = 120


def make_test_image() -> bytes:
    img = Image.new("RGB", (256, 256), color=(40, 80, 160))
    d = ImageDraw.Draw(img)
    d.rectangle([40, 40, 216, 216], outline=(255, 255, 255), width=4)
    d.text((60, 110), "MTP TEST", fill=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def free_port() -> int:
    s = socket.socket()
    s.bind(("", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def wait_ready(port: int, timeout: int) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


def build_cmd(port: int, mtp: bool, vision: bool) -> list:
    cmd = [
        SERVER_EXE,
        "-m", GGUF_PATH,
        "--port", str(port),
        "-c", str(CTX_SIZE),
        "-ngl", str(GPU_LAYERS),
        "-b", str(BATCH),
        "-ub", str(UBATCH),
        "--parallel", "1",
        "--jinja",
        "-n", "256",
        "--flash-attn", "on",
    ]
    if vision:
        cmd += ["--mmproj", MMPROJ_PATH]
    if mtp:
        cmd += ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"]
    return cmd


def start_server(port: int, mtp: bool, vision: bool):
    cmd = build_cmd(port, mtp, vision)
    print(f"\n[CMD] {' '.join(cmd)}")
    creationflags = 0x08000000  # CREATE_NO_WINDOW
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, creationflags=creationflags)
    return p


def run_request(port: int, image_b64: str | None, prompt: str, scenario: str) -> dict:
    if image_b64 is not None:
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
        ]
    else:
        content = prompt
    payload = {
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.1,
        "max_tokens": 64,
    }
    t0 = time.time()
    try:
        r = requests.post(f"http://127.0.0.1:{port}/v1/chat/completions", json=payload, timeout=TIMEOUT_REQUEST)
        elapsed = time.time() - t0
        body = r.text
        if r.status_code == 200:
            data = r.json()
            msg = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            usage = data.get("usage", {})
            return {
                "scenario": scenario,
                "status": r.status_code,
                "ok": True,
                "elapsed": round(elapsed, 2),
                "tokens": usage.get("completion_tokens", "?"),
                "content": msg[:120].replace("\n", " "),
                "note": "",
            }
        return {
            "scenario": scenario,
            "status": r.status_code,
            "ok": False,
            "elapsed": round(elapsed, 2),
            "tokens": "-",
            "content": "",
            "note": body[:200].replace("\n", " "),
        }
    except Exception as e:
        elapsed = time.time() - t0
        return {
            "scenario": scenario,
            "status": "EXC",
            "ok": False,
            "elapsed": round(elapsed, 2),
            "tokens": "-",
            "content": "",
            "note": str(e)[:200],
        }


def kill_server(p: subprocess.Popen):
    try:
        p.terminate()
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()
    except Exception:
        pass


def scenario(name: str, mtp: bool, vision: bool, image_b64, prompt: str) -> dict:
    port = free_port()
    print(f"\n=== {name} (mtp={mtp}, port={port}) ===")
    p = start_server(port, mtp, vision)
    try:
        if not wait_ready(port, TIMEOUT_READY):
            kill_server(p)
            return {"scenario": name, "status": "TIMEOUT_READY", "ok": False, "elapsed": TIMEOUT_READY, "tokens": "-", "content": "", "note": "Server did not become ready"}
        print(f"  [ready, sending request]")
        return run_request(port, image_b64, prompt, name)
    finally:
        kill_server(p)
        time.sleep(2)


def main():
    if not os.path.exists(GGUF_PATH):
        print(f"GGUF not found: {GGUF_PATH}")
        sys.exit(1)
    if not os.path.exists(MMPROJ_PATH):
        print(f"mmproj not found: {MMPROJ_PATH}")
        sys.exit(1)
    if not os.path.exists(SERVER_EXE):
        print(f"llama-server.exe not found: {SERVER_EXE}")
        sys.exit(1)

    print(f"Model:    {GGUF_PATH}")
    print(f"mmproj:   {MMPROJ_PATH}")
    print(f"Server:   {SERVER_EXE}")
    print(f"version:  {subprocess.check_output([SERVER_EXE, '--version'], text=True).strip()}")

    img_bytes = make_test_image()
    img_b64 = base64.b64encode(img_bytes).decode("ascii")
    print(f"Image:    256x256 PNG, {len(img_bytes)} bytes (b64 {len(img_b64)})")

    vision_prompt = "Опиши одним предложением, что изображено на картинке. Ответь по-русски."
    text_prompt = "Ответь одним предложением: сколько будет 2+2?"

    results = []
    results.append(scenario("vision + MTP ON ", mtp=True, vision=True, image_b64=img_b64, prompt=vision_prompt))
    results.append(scenario("vision + MTP OFF", mtp=False, vision=True, image_b64=img_b64, prompt=vision_prompt))
    results.append(scenario("text  + MTP ON ", mtp=True, vision=False, image_b64=None, prompt=text_prompt))
    results.append(scenario("text  + MTP OFF", mtp=False, vision=False, image_b64=None, prompt=text_prompt))

    print("\n" + "=" * 110)
    print("СВОДКА")
    print("=" * 110)
    header = f"{'scenario':<20} {'ok':<4} {'status':<6} {'elapsed':<9} {'tokens':<8} {'note':<60}"
    print(header)
    print("-" * 110)
    for r in results:
        print(f"{r['scenario']:<20} {str(r['ok']):<4} {str(r['status']):<6} {str(r['elapsed'])+'s':<9} {str(r['tokens']):<8} {r['note'][:58]:<60}")
        if r.get("content"):
            print(f"  → {r['content']}")
    print("=" * 110)

    print("\nВЫВОД:")
    vision_mtp_on = results[0]
    vision_mtp_off = results[1]
    text_mtp_on = results[2]
    text_mtp_off = results[3]
    if text_mtp_on["ok"] and not vision_mtp_on["ok"] and vision_mtp_off["ok"]:
        print("  ✗ MTP работает для text-only, но ломает vision → подтверждено: MTP не дружит с vision")
    elif text_mtp_on["ok"] and vision_mtp_on["ok"] and vision_mtp_off["ok"]:
        diff = vision_mtp_on["elapsed"] - vision_mtp_off["elapsed"]
        print(f"  ✓ MTP работает и для text, и для vision (vision: {diff:+.2f}s с MTP)")
    elif not text_mtp_on["ok"]:
        print("  ✗ MTP не работает даже для text-only — возможно, проблема с самой моделью/сборкой")
    else:
        print(f"  ? Неожиданный результат. Проверь ответы выше.")


if __name__ == "__main__":
    main()
