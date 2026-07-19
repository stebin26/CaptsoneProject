"""Logging configuration for the dashboard service.

The dashboard image does not ship ``ops_common``, so it configures the standard
library directly. The format deliberately mirrors the one the other services
emit, and extra fields passed via ``logger.warning(..., extra={...})`` are
appended as ``key=value`` pairs so context survives into the container log.
"""

from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False

_RESERVED_ATTRS = frozenset(
    logging.LogRecord("", 0, "", 0, "", None, None).__dict__
) | {"message", "asctime", "taskName"}


class _ContextFormatter(logging.Formatter):
    """Append any extra fields to the message as ``key=value`` pairs."""

    def format(self, record: logging.LogRecord) -> str:
        """Format a record, appending its non-reserved extra fields.

        Args:
            record: The record to render.

        Returns:
            The rendered log line.
        """
        base = super().format(record)
        extras = [
            f"{key}={value}"
            for key, value in record.__dict__.items()
            if key not in _RESERVED_ATTRS and not key.startswith("_")
        ]
        return f"{base} | {' '.join(extras)}" if extras else base


def configure_logging() -> None:
    """Configure the root logger once for the dashboard process.

    Safe to call more than once: repeat calls return immediately rather than
    stacking duplicate handlers, which would print every line twice under
    gunicorn's multiple workers.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    level_name = os.environ.get("OPS_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, None)
    if not isinstance(level, int):
        # An unknown level must not stop the dashboard from starting; the
        # fallback is logged once the handler below exists.
        level = logging.INFO
        level_name = f"{level_name} (unrecognised, using INFO)"

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        _ContextFormatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    _CONFIGURED = True
    logging.getLogger(__name__).info("Dashboard logging configured at %s", level_name)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, configuring logging on first use.

    Args:
        name: Logger name, normally the calling module's ``__name__``.

    Returns:
        The named logger.
    """
    configure_logging()
    return logging.getLogger(name)
