"""Quality scoring module.

This module provides strategies for scoring image quality
based on various computed metrics.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

from image_deduper_v2.protocols import ScoringStrategy

if TYPE_CHECKING:
    from image_deduper_v2.config import ScoringWeights
    from image_deduper_v2.models import ImageMetrics


class DefaultScorer(ScoringStrategy):
    """Default scoring strategy using weighted metrics.
    
    Combines sharpness, colorfulness, resolution, and GPS presence
    into a single quality score.
    
    Attributes:
        weights: ScoringWeights configuration.
    """

    def __init__(self, weights: "ScoringWeights | None" = None) -> None:
        """Initialize the scorer.
        
        Args:
            weights: Optional ScoringWeights configuration.
                    If None, uses default weights.
        """
        from image_deduper_v2.config import ScoringWeights
        
        self._weights = weights or ScoringWeights()

    @property
    def weights(self) -> "ScoringWeights":
        """Return the scoring weights."""
        return self._weights

    def compute_score(self, metrics: "ImageMetrics") -> float:
        """Compute weighted quality score.
        
        Args:
            metrics: Image quality metrics.
            
        Returns:
            Combined quality score (higher = better).
        """
        # Normalize sharpness (typical range 0-10000, cap at 5000)
        norm_sharpness = min(metrics.sharpness / 5000.0, 1.0)
        
        # Normalize colorfulness (typical range 0-100, cap at 100)
        norm_colorfulness = min(metrics.colorfulness / 100.0, 1.0)
        
        # Normalize resolution using logarithmic scale
        # 1MP = 6.0, 10MP = 7.0, 100MP = 8.0
        log_resolution = math.log10(max(1, metrics.resolution))
        norm_resolution = min((log_resolution - 5.0) / 3.0, 1.0)  # 0 at 100K, 1 at 100MP
        norm_resolution = max(0.0, norm_resolution)
        
        # Compute weighted sum
        score = (
            norm_sharpness * self._weights.sharpness
            + norm_colorfulness * self._weights.colorfulness
            + norm_resolution * self._weights.resolution
        )
        
        return score


class SharpnessScorer(ScoringStrategy):
    """Scoring strategy that prioritizes sharpness.
    
    Useful when image clarity is the primary concern.
    """

    def compute_score(self, metrics: "ImageMetrics") -> float:
        """Compute score based primarily on sharpness.
        
        Args:
            metrics: Image quality metrics.
            
        Returns:
            Sharpness-weighted quality score.
        """
        norm_sharpness = min(metrics.sharpness / 5000.0, 2.0)
        norm_resolution = math.log10(max(1, metrics.resolution)) / 7.0
        
        return norm_sharpness * 0.8 + norm_resolution * 0.2


class ResolutionScorer(ScoringStrategy):
    """Scoring strategy that prioritizes resolution.
    
    Useful when image size is the primary concern.
    """

    def compute_score(self, metrics: "ImageMetrics") -> float:
        """Compute score based primarily on resolution.
        
        Args:
            metrics: Image quality metrics.
            
        Returns:
            Resolution-weighted quality score.
        """
        # Use megapixels directly
        megapixels = metrics.resolution / 1_000_000
        norm_sharpness = min(metrics.sharpness / 5000.0, 1.0)
        
        return megapixels * 0.8 + norm_sharpness * 0.2


class ColorfulnessScorer(ScoringStrategy):
    """Scoring strategy that prioritizes colorful images.
    
    Useful for selecting vibrant, colorful photos.
    """

    def compute_score(self, metrics: "ImageMetrics") -> float:
        """Compute score based primarily on colorfulness.
        
        Args:
            metrics: Image quality metrics.
            
        Returns:
            Colorfulness-weighted quality score.
        """
        norm_colorfulness = min(metrics.colorfulness / 50.0, 2.0)
        norm_sharpness = min(metrics.sharpness / 5000.0, 1.0)
        norm_resolution = math.log10(max(1, metrics.resolution)) / 7.0
        
        return (
            norm_colorfulness * 0.5
            + norm_sharpness * 0.3
            + norm_resolution * 0.2
        )


class BalancedScorer(ScoringStrategy):
    """Scoring strategy with equal weights for all metrics.
    
    Provides a balanced approach when no single metric
    should dominate.
    """

    def compute_score(self, metrics: "ImageMetrics") -> float:
        """Compute equally-weighted quality score.
        
        Args:
            metrics: Image quality metrics.
            
        Returns:
            Balanced quality score.
        """
        norm_sharpness = min(metrics.sharpness / 5000.0, 1.0)
        norm_colorfulness = min(metrics.colorfulness / 100.0, 1.0)
        log_resolution = math.log10(max(1, metrics.resolution))
        norm_resolution = min((log_resolution - 5.0) / 3.0, 1.0)
        norm_resolution = max(0.0, norm_resolution)
        
        return (norm_sharpness + norm_colorfulness + norm_resolution) / 3.0


def get_scorer(name: str, weights: "ScoringWeights | None" = None) -> ScoringStrategy:
    """Get a scoring strategy by name.
    
    Args:
        name: Strategy name ('default', 'sharpness', 'resolution', 
              'colorfulness', 'balanced').
        weights: Optional weights for default scorer.
        
    Returns:
        Scoring strategy instance.
        
    Raises:
        ValueError: If the name is not recognized.
        
    Example:
        >>> scorer = get_scorer("sharpness")
        >>> score = scorer.compute_score(metrics)
    """
    scorers = {
        "default": lambda: DefaultScorer(weights),
        "sharpness": SharpnessScorer,
        "resolution": ResolutionScorer,
        "colorfulness": ColorfulnessScorer,
        "balanced": BalancedScorer,
    }
    
    if name.lower() not in scorers:
        raise ValueError(
            f"Unknown scorer: {name}. "
            f"Available: {', '.join(scorers.keys())}"
        )
    
    return scorers[name.lower()]()
