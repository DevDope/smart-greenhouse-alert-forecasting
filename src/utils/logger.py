"""Logging helpers."""

from __future__ import annotations

import logging
import os


def configure_logging() -> None:
    level_name = os.getenv("GREENHOUSE_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level_name, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)
