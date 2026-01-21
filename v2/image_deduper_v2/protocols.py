"""Protocol definitions for extensible components.

This module defines abstract interfaces (protocols) that allow for
dependency injection and custom implementations of key components.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from image_deduper_v2.models import ImageRecord, ImageMetrics


class ScoringStrategy(ABC):
    """Abstract base class for image quality scoring strategies.
    
    Implement this protocol to create custom scoring algorithms that
    prioritize different aspects of image quality.
    
    Example:
        >>> class ColorfulnessScorer(ScoringStrategy):
        ...     def compute_score(self, metrics: ImageMetrics) -> float:
        ...         return metrics.colorfulness * metrics.resolution
    """

    @abstractmethod
    def compute_score(self, metrics: "ImageMetrics") -> float:
        """Compute a quality score for an image based on its metrics.
        
        Args:
            metrics: Computed metrics for the image.
            
        Returns:
            A float score where higher values indicate better quality.
            Scores should typically be non-negative.
        """
        ...

    @property
    def name(self) -> str:
        """Return the name of this scoring strategy."""
        return self.__class__.__name__


class DeduplicationStrategy(ABC):
    """Abstract base class for deduplication strategies.
    
    Implement this protocol to create custom deduplication algorithms
    based on different similarity metrics.
    """

    @abstractmethod
    def find_duplicates(
        self,
        records: Sequence["ImageRecord"],
    ) -> list[list[int]]:
        """Find groups of duplicate images.
        
        Args:
            records: Sequence of image records to analyze.
            
        Returns:
            List of clusters, where each cluster is a list of indices
            into the input records sequence. Each cluster contains
            indices of images considered duplicates.
        """
        ...

    @property
    def name(self) -> str:
        """Return the name of this deduplication strategy."""
        return self.__class__.__name__


class ProgressCallback(ABC):
    """Abstract base class for progress reporting.
    
    Implement this protocol to receive progress updates during
    the deduplication pipeline execution.
    """

    @abstractmethod
    def on_stage_start(self, stage: str, total: int) -> None:
        """Called when a pipeline stage begins.
        
        Args:
            stage: Name of the stage (e.g., "scanning", "analyzing").
            total: Total number of items to process in this stage.
        """
        ...

    @abstractmethod
    def on_progress(self, stage: str, current: int, total: int) -> None:
        """Called to report progress within a stage.
        
        Args:
            stage: Name of the current stage.
            current: Number of items processed so far.
            total: Total number of items to process.
        """
        ...

    @abstractmethod
    def on_stage_complete(self, stage: str, message: str) -> None:
        """Called when a pipeline stage completes.
        
        Args:
            stage: Name of the completed stage.
            message: Summary message about the stage completion.
        """
        ...


class ConsoleProgressCallback(ProgressCallback):
    """Simple console-based progress reporter."""

    def on_stage_start(self, stage: str, total: int) -> None:
        """Print stage start message to console."""
        print(f"[{stage}] Starting... ({total} items)")

    def on_progress(self, stage: str, current: int, total: int) -> None:
        """Print progress update to console."""
        percent = (current / total * 100) if total > 0 else 0
        print(f"\r[{stage}] {current}/{total} ({percent:.1f}%)", end="", flush=True)

    def on_stage_complete(self, stage: str, message: str) -> None:
        """Print stage completion message to console."""
        print(f"\n[{stage}] Complete: {message}")
