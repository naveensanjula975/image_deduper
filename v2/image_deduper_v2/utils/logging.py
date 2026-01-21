"""Logging configuration utilities.

This module provides consistent logging setup across the package.
"""
from __future__ import annotations

import logging
import sys
from typing import TextIO


def configure_logging(
    level: int = logging.INFO,
    format_string: str | None = None,
    stream: TextIO | None = None,
    log_file: str | None = None,
) -> logging.Logger:
    """Configure and return the package logger.
    
    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO).
        format_string: Custom format string for log messages.
        stream: Output stream (defaults to stderr).
        log_file: Optional file path to also log to.
        
    Returns:
        Configured logger instance.
        
    Example:
        >>> logger = configure_logging(level=logging.DEBUG)
        >>> logger.info("Pipeline started")
    """
    if format_string is None:
        format_string = (
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        )

    # Get the package logger
    logger = logging.getLogger("image_deduper_v2")
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    console_handler = logging.StreamHandler(stream or sys.stderr)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a logger instance.
    
    Args:
        name: Optional name suffix for the logger.
        
    Returns:
        Logger instance.
        
    Example:
        >>> logger = get_logger("scanner")
        >>> logger.debug("Scanning directory...")
    """
    if name:
        return logging.getLogger(f"image_deduper_v2.{name}")
    return logging.getLogger("image_deduper_v2")


class LoggerMixin:
    """Mixin class that provides a logger property.
    
    Classes that inherit from this mixin will have access to a
    logger instance via the `logger` property.
    
    Example:
        >>> class MyProcessor(LoggerMixin):
        ...     def process(self):
        ...         self.logger.info("Processing...")
    """

    @property
    def logger(self) -> logging.Logger:
        """Return a logger named after this class.
        
        Returns:
            Logger instance for this class.
        """
        return logging.getLogger(
            f"image_deduper_v2.{self.__class__.__name__}"
        )
