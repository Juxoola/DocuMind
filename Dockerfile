FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements-cuda.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# App
COPY . .

# Env defaults
ENV HOST=0.0.0.0
ENV PORT=8000
ENV HF_HUB_OFFLINE=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import httpx; r=httpx.get('http://localhost:8000/health'); assert r.status_code==200"

CMD ["python", "main.py"]
