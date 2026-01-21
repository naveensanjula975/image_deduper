"""Image Deduper V2 - Production-grade image deduplication pipeline.

This package provides a comprehensive solution for:
- Scanning directories for images
- Computing perceptual and cryptographic hashes
- Deduplicating by exact match, perceptual similarity, and GPS proximity
- Scoring and ranking images by quality metrics
- Selecting top-N images and exporting to a target directory

Example:
    >>> from image_deduper_v2 import ImageDeduper, Settings
    >>> settings = Settings(
    ...     input_paths=["./photos"],
    ...     output_dir="./selected",
    ...     target_count=200,
    ... )
    >>> deduper = ImageDeduper(settings)
    >>> results = deduper.run()
"""
from __future__ import annotations

__version__ = "2.0.0"
__author__ = "Avolution Tech"

from image_deduper_v2.config import Settings
from image_deduper_v2.exceptions import (
    ImageDeduperError,
    AnalysisError,
    ExportError,
    ConfigurationError,
)
from image_deduper_v2.models import ImageRecord, ImageMetrics, DeduplicationResult
from image_deduper_v2.pipeline import ImageDeduper
from image_deduper_v2.protocols import ScoringStrategy, DeduplicationStrategy

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Core classes
    "Settings",
    "ImageDeduper",
    "ImageRecord",
    "ImageMetrics",
    "DeduplicationResult",
    # Protocols
    "ScoringStrategy",
    "DeduplicationStrategy",
    # Exceptions
    "ImageDeduperError",
    "AnalysisError",
    "ExportError",
    "ConfigurationError",
]
