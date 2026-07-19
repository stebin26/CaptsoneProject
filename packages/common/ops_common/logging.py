"""Structured logging configuration shared by every service.

Emits JSON in deployed environments (machine-parseable, one object per line)
and a readable plain format during development. Both formatters surface the
extra fields passed via ``logger.info(..., extra={...})``, so contextual detail
like dataset ids and paths stays attached to the message instead of being
formatted into it.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

_CONFIGURED = False

_RESERVED_ATTRS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
    "taskName",
}


class JsonFormatter(logging.Formatter):
    """Render log records as single-line JSON objects.

    Includes timestamp, level, logger name, and message, plus any exception or
    stack trace and every non-reserved extra field attached to the record.
    """
    def format(self, record: logging.LogRecord) -> str:
        """Format a log record.

        Args:
            record: The log record to render.

        Returns:
            The rendered log line.
        """
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)

        for key, value in record.__dict__.items():
            if key not in _RESERVED_ATTRS and not key.startswith("_"):
                payload[key] = _coerce(value)

        return json.dumps(payload, default=str, ensure_ascii=False)


class PlainFormatter(logging.Formatter):
    """Render log records as readable text for local development.

    Appends any extra fields as ``key=value`` pairs after the message so context is
    not lost in the human-facing format.
    """
    def __init__(self) -> None:
        """Configure the timestamped development log format."""
        super().__init__(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record.

        Args:
            record: The log record to render.

        Returns:
            The rendered log line.
        """
        base = super().format(record)
        extras = {
            k: v
            for k, v in record.__dict__.items()
            if k not in _RESERVED_ATTRS and not k.startswith("_")
        }
        if extras:
            rendered = " ".join(f"{k}={v}" for k, v in extras.items())
            base = f"{base} | {rendered}"
        return base


def _coerce(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, (list, tuple, dict)):
        return value
    return str(value)


def configure_logging(level: str | None = None, json_output: bool | None = None) -> None:
    """Configure root logging for the current process.

    Replaces any existing handlers with a single stdout handler, chooses JSON or
    plain output based on the environment, and quiets noisy third-party loggers.
    Safe to call more than once.

    Args:
        level: Log level override; defaults to the configured level.
        json_output: Force JSON on or off; defaults to JSON outside development.
    """
    global _CONFIGURED

    from ops_common.config import settings

    resolved_level = (level or settings.log_level).upper()
    use_json = json_output if json_output is not None else (settings.environment != "development")

    root = logging.getLogger()
    root.setLevel(resolved_level)

    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JsonFormatter() if use_json else PlainFormatter())
    handler.setLevel(resolved_level)
    root.addHandler(handler)

    for noisy in ("uvicorn.access", "sqlalchemy.engine.Engine", "duckdb"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, configuring logging on first use.

    Args:
        name: Logger name, conventionally the calling module's ``__name__``.

    Returns:
        The configured logger.
    """
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)
