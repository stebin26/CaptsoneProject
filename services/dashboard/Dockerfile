FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY services/dashboard/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY services/dashboard/app /app/app

ENV PYTHONPATH=/app

EXPOSE 8050

# Agent questions take 60-200s on CPU. Gunicorn kills any worker that has not
# finished a request within --timeout, so this ceiling MUST sit above the
# client timeout in api_client.py (5, 240) — otherwise the worker dies before
# the answer arrives and the UI hangs on "Investigating..." forever.
CMD ["gunicorn", "app.main:server", \
     "--bind", "0.0.0.0:8050", \
     "--workers", "2", \
     "--timeout", "300", \
     "--graceful-timeout", "300"]