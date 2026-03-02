"""Minimal stderr logger factory."""

from __future__ import annotations

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Return a logger configured to write to stderr."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(
        logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s')
    )

    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger
