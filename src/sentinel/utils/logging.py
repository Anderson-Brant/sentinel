"""Rich-aware logger setup. Respects SENTINEL_LOG_LEVEL."""

from __future__ import annotations

import logging

from rich.logging import RichHandler

from sentinel.config import load_secrets

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        level = load_secrets().log_level.upper()
        logging.basicConfig(
            level=level,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(rich_tracebacks=True, show_path=False, markup=True)],
        )
        _CONFIGURED = True
    return logging.getLogger(name)
