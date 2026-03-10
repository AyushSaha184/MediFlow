# ── MediFlow Backend ─────────────────────────────────────────────────────────
# Single-stage Python image.
# The frontend is deployed separately on Vercel and is NOT built here.
#
# Notable choices:
#   - python:3.11-slim  →  avoids the ~1 GB overhead of the full image
#   - CPU-only torch     →  embeddings are served by the NVIDIA API; there is
#                           no need for GPU-specific PyTorch wheels (~1.4 GB saved)
#   - en_core_web_lg    →  required by presidio-analyzer's default NLP engine
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

# 1. Install CPU-only PyTorch before the rest of requirements.txt so pip
#    does not pull the full CUDA wheel (~2 GB) as a transitive dependency of
#    sentence-transformers.
RUN pip install --no-cache-dir \
        torch==2.2.2 \
        --index-url https://download.pytorch.org/whl/cpu

# 2. Install all remaining requirements (torch is now pinned to the CPU build).
RUN pip install --no-cache-dir -r requirements.txt

# 3. Download the spaCy model used by presidio-analyzer's default NLP engine.
RUN python -m spacy download en_core_web_lg

# ── Application code ──────────────────────────────────────────────────────────
COPY main.py .
COPY src/  src/

# Copy any knowledge-base files that were committed to the repo.
# The data/User/ directory is intentionally excluded (runtime data, never baked in).
COPY data/knowledge_base/ data/knowledge_base/

# ── Runtime directories ───────────────────────────────────────────────────────
# Create writable dirs that the app writes to at runtime.
RUN mkdir -p data/User logs

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
