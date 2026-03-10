# ── MediFlow Backend ─────────────────────────────────────────────────────────
# Single-stage Python image.
# The frontend is deployed separately on Vercel and is NOT built here.
#
# Notable choices:
#   - python:3.11-slim   →  avoids the ~1 GB overhead of the full image
#   - no torch/ST        →  embeddings use the NVIDIA API (pure HTTP); no local
#                           model weights needed, saves ~1.4 GB
#   - en_core_web_sm     →  12 MB spaCy model; sufficient for Presidio PII NER
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# ── System packages ───────────────────────────────────────────────────────────
# tesseract-ocr       → pytesseract (image OCR)
# tesseract-ocr-eng   → English language pack
# libgl1              → OpenCV / Pillow internal deps on slim base
# libglib2.0-0        → same
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-eng \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy requirements first so Docker can cache this layer independently of
# application code changes.
COPY requirements.txt .

# 1. Install all requirements (no torch/sentence-transformers — embeddings are
#    served by the NVIDIA API so no local model weights are needed).
RUN pip install --no-cache-dir -r requirements.txt

# 2. presidio-analyzer has en_core_web_lg as a pip dependency and installs it
#    automatically (~400 MB). Uninstall it and replace with en_core_web_sm
#    (~12 MB). presidio_service.py explicitly configures Presidio to use sm.
RUN pip uninstall -y en-core-web-lg || true
RUN python -m spacy download en_core_web_sm

# ── Application code ──────────────────────────────────────────────────────────
COPY main.py .
COPY src/  src/

# ── Runtime directories ───────────────────────────────────────────────────────
# Create writable dirs that the app writes to at runtime.
# knowledge_base/ is populated post-deployment via: python -m src.rag.ingest_global
RUN mkdir -p data/knowledge_base data/User logs

# ── Environment ───────────────────────────────────────────────────────────────
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# PORT is overridden at runtime by Render ($PORT) or docker-compose.
ENV PORT=8000
EXPOSE 8000

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')"

# ── Start ─────────────────────────────────────────────────────────────────────
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT}
