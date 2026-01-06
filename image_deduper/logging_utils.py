from __future__ import annotations

import logging
from typing import Optional


def configure_logging(level: int = logging.INFO, logger_name: str = "image_deduper") -> logging.Logger:
    """Configures and returns an application logger.

    Args:
        level: Logging level.
        logger_name: Logger name.

    Returns:
        Configured logger.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger
