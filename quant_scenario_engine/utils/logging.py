"""Structured logging utilities with JSON output."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from typing import Any, Dict, Optional

DEFAULT_FIELDS = {"run_id", "component", "symbol", "config_index", "duration_ms"}


class JSONFormatter(logging.Formatter):
    """JSON formatter adding common contextual fields when present."""

    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        payload: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "message": record.getMessage(),
        }
        for field in DEFAULT_FIELDS:
            if hasattr(record, field):
                payload[field] = getattr(record, field)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging(run_id: Optional[str] = None, component: Optional[str] = None, level: int = logging.INFO) -> None:
    """Configure root logger with structured JSON output.

    Embeds run_id/component defaults so downstream loggers inherit context without
    requiring every call to pass `extra`.
    """

    class ContextFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
            if run_id and not hasattr(record, "run_id"):
                record.run_id = run_id
            if component and not hasattr(record, "component"):
                record.component = component
            return True

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    handler.addFilter(ContextFilter())
    root.addHandler(handler)


def get_logger(name: str, run_id: Optional[str] = None, component: Optional[str] = None) -> logging.Logger:
    """Convenience helper to fetch a logger with optional context defaults."""

    logger = logging.getLogger(name)
    if run_id or component:
        f = logging.Filter()

        def _filter(record: logging.LogRecord) -> bool:  # type: ignore[override]
            if run_id and not hasattr(record, "run_id"):
                record.run_id = run_id
            if component and not hasattr(record, "component"):
                record.component = component
            return True

        f.filter = _filter  # type: ignore[assignment]
        logger.addFilter(f)
    return logger

