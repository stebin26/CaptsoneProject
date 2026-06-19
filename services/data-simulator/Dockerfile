FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY packages/common /app/packages/common
RUN pip install -e /app/packages/common

COPY services/data-simulator/requirements.txt /app/services/data-simulator/requirements.txt
RUN pip install -r /app/services/data-simulator/requirements.txt

COPY packages /app/packages
COPY services/data-simulator/app /app/app

ENV PYTHONPATH=/app:/app/packages/common

CMD ["python", "-m", "app.cli", "seed"]