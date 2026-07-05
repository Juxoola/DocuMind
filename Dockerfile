FROM python:3.11-slim

WORKDIR /app

# System deps: ffmpeg, build tools for llama.cpp, libreoffice
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libreoffice-core \
    libreoffice-writer \
    libreoffice-calc \
    build-essential \
    cmake \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Build llama.cpp from source (CPU-only, works everywhere)
RUN git clone --depth 1 https://github.com/ggerganov/llama.cpp.git /tmp/llama.cpp \
    && cd /tmp/llama.cpp \
    && cmake -B build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --config Release -j$(nproc) \
    && cp build/bin/llama-server /usr/local/bin/ \
    && rm -rf /tmp/llama.cpp

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY . .

# Env defaults
ENV HOST=0.0.0.0
ENV PORT=8000
ENV HF_HUB_OFFLINE=1
ENV LLAMA_CPP_BINARY=/usr/local/bin/llama-server

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import httpx; r=httpx.get('http://localhost:8000/health'); assert r.status_code==200"

CMD ["python", "main.py"]
