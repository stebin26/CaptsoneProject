FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.hf_cache \
    OPS_EMBEDDING_MODEL=all-MiniLM-L6-v2

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Editable local packages (rarely change)
COPY packages/common /app/packages/common
RUN --mount=type=cache,target=/root/.cache/pip pip install -e /app/packages/common

COPY services/ingestion /app/services/ingestion
RUN --mount=type=cache,target=/root/.cache/pip pip install -e /app/services/ingestion

# Heavy, stable deps (torch/sentence-transformers etc.) — own layer so adding a
# small app dep never re-triggers this multi-hundred-MB install/download.
COPY services/api-gateway/requirements-base.txt /app/services/api-gateway/requirements-base.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -r /app/services/api-gateway/requirements-base.txt

# Pre-download embedding model (needs sentence-transformers from base above).
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Light app deps (fastapi, auth, langgraph…) — the layer that busts when you add
# a dependency. Fast because no torch here.
COPY services/api-gateway/requirements.txt /app/services/api-gateway/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -r /app/services/api-gateway/requirements.txt

COPY packages /app/packages
COPY services/ingestion /app/services/ingestion
COPY services/api-gateway/api_app /app/api_app
COPY services/intelligence /app/services/intelligence
COPY services/rag /app/services/rag
COPY services/agent /app/services/agent
COPY services/ml /app/services/ml
COPY data-hub /app/data-hub

ENV PYTHONPATH=/app:/app/packages/common:/app/services/ingestion

EXPOSE 8000

CMD ["uvicorn", "api_app.main:app", "--host", "0.0.0.0", "--port", "8000"]