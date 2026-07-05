FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.hf_cache \
    OPS_EMBEDDING_MODEL=all-MiniLM-L6-v2

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY packages/common /app/packages/common
RUN pip install -e /app/packages/common

COPY services/ingestion /app/services/ingestion
RUN pip install -e /app/services/ingestion

COPY services/api-gateway/requirements.txt /app/services/api-gateway/requirements.txt
RUN pip install -r /app/services/api-gateway/requirements.txt

# Pre-download the embedding model into the image so runtime never hits the
# network. Model name must match OPS_EMBEDDING_MODEL above.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY packages /app/packages
COPY services/ingestion /app/services/ingestion
COPY services/api-gateway/api_app /app/api_app
COPY services/intelligence /app/services/intelligence
COPY services/rag /app/services/rag
COPY data-hub /app/data-hub

ENV PYTHONPATH=/app:/app/packages/common:/app/services/ingestion

EXPOSE 8000

CMD ["uvicorn", "api_app.main:app", "--host", "0.0.0.0", "--port", "8000"]