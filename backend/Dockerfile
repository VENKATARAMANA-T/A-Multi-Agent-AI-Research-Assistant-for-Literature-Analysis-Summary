FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/app/.cache/huggingface

# PyMuPDF and torch need these; libgl1/libglib are also what opencv (pulled in
# by the OCR engine) links against. No OCR binary is installed: RapidOCR ships
# its models as ONNX inside the wheel.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app ./app

# Pre-download the embedding model so the first upload is not slowed by a
# cold-start download (and so the container works without internet access).
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')" \
    || echo 'Model pre-download skipped; the app will fetch it at runtime.'

# Warm the OCR models too, so the first scanned upload does not pay for it.
RUN python -c "from rapidocr_onnxruntime import RapidOCR; RapidOCR()" \
    || echo 'OCR warm-up skipped; the engine will initialise at runtime.'

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/data \
    && chown -R app:app /app /home/app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=5 \
    CMD curl -fsS http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
