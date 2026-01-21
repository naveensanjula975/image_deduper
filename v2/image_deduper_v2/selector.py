"""Image selection module.

This module handles selecting the top-N images based on
quality scores and configurable criteria.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List

from image_deduper_v2.models import ImageRecord
from image_deduper_v2.protocols import ScoringStrategy
from image_deduper_v2.scorer import DefaultScorer
from image_deduper_v2.utils.logging import get_logger

if TYPE_CHECKING:
    from image_deduper_v2.config import Settings


class ImageSelector:
    """Selects the best images based on quality scores.
    
    Ranks images using a scoring strategy and selects the
    top N images for export.
    
    Attributes:
        target_count: Number of images to select.
        scorer: Scoring strategy to use.
        prefer_gps: Whether to prefer images with GPS data.
    """

    def __init__(
        self,
        target_count: int = 200,
        scorer: ScoringStrategy | None = None,
        prefer_gps: bool = True,
    ) -> None:
        """Initialize the selector.
        
        Args:
            target_count: Number of images to select.
            scorer: Optional custom scoring strategy.
            prefer_gps: Whether to boost images with GPS data.
        """
        self._target_count = target_count
        self._scorer = scorer or DefaultScorer()
        self._prefer_gps = prefer_gps
        self._logger = get_logger("selector")

    @property
    def target_count(self) -> int:
        """Return the target selection count."""
        return self._target_count

    @property
    def scorer(self) -> ScoringStrategy:
        """Return the scoring strategy."""
        return self._scorer

    @classmethod
    def from_settings(
        cls,
        settings: "Settings",
        scorer: ScoringStrategy | None = None,
    ) -> "ImageSelector":
        """Create selector from settings.
        
        Args:
            settings: Application settings.
            scorer: Optional custom scoring strategy.
            
        Returns:
            Configured ImageSelector instance.
        """
        if scorer is None:
            scorer = DefaultScorer(settings.scoring_weights)
        
        return cls(
            target_count=settings.target_count,
            scorer=scorer,
            prefer_gps=settings.prefer_gps_images,
        )

    def select(self, records: List[ImageRecord]) -> List[ImageRecord]:
        """Select the top-N images by quality score.
        
        Args:
            records: List of candidate image records.
            
        Returns:
            List of selected records, sorted by quality (best first).
        """
        if not records:
            return []

        if len(records) <= self._target_count:
            self._logger.info(
                f"Only {len(records)} images available, "
                f"returning all (target was {self._target_count})"
            )
            return self._sort_by_quality(records)

        # Score and rank all images
        scored = self._score_records(records)
        
        # Select top N
        selected = scored[: self._target_count]
        
        self._logger.info(
            f"Selected top {len(selected)} images from {len(records)} candidates"
        )
        
        return selected

    def select_with_scores(
        self,
        records: List[ImageRecord],
    ) -> List[tuple[ImageRecord, float]]:
        """Select images and return with their scores.
        
        Args:
            records: List of candidate image records.
            
        Returns:
            List of (record, score) tuples, sorted by score descending.
        """
        if not records:
            return []

        # Score all images
        scored_pairs = [
            (record, self._compute_selection_score(record))
            for record in records
        ]
        
        # Sort by score descending
        scored_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Select top N
        selected = scored_pairs[: self._target_count]
        
        return selected

    def _score_records(self, records: List[ImageRecord]) -> List[ImageRecord]:
        """Score and sort records by quality.
        
        Args:
            records: List of image records.
            
        Returns:
            Records sorted by quality score descending.
        """
        return sorted(
            records,
            key=lambda r: self._compute_selection_score(r),
            reverse=True,
        )

    def _sort_by_quality(self, records: List[ImageRecord]) -> List[ImageRecord]:
        """Sort records by quality without limiting count.
        
        Args:
            records: List of image records.
            
        Returns:
            Records sorted by quality score descending.
        """
        return sorted(
            records,
            key=lambda r: self._compute_selection_score(r),
            reverse=True,
        )

    def _compute_selection_score(self, record: ImageRecord) -> float:
        """Compute final selection score for a record.
        
        Combines the scorer's quality score with GPS preference.
        
        Args:
            record: Image record to score.
            
        Returns:
            Combined selection score.
        """
        base_score = self._scorer.compute_score(record.metrics)
        
        # Add GPS bonus if preferred and present
        gps_bonus = 0.0
        if self._prefer_gps and record.has_gps:
            gps_bonus = 0.1  # Small bonus for GPS presence
        
        return base_score + gps_bonus


class DiversitySelector(ImageSelector):
    """Selector that promotes diversity in the selection.
    
    Attempts to select images that are spread across different
    locations and capture times, not just the highest quality.
    """

    def __init__(
        self,
        target_count: int = 200,
        scorer: ScoringStrategy | None = None,
        prefer_gps: bool = True,
        diversity_weight: float = 0.2,
    ) -> None:
        """Initialize the diversity selector.
        
        Args:
            target_count: Number of images to select.
            scorer: Optional custom scoring strategy.
            prefer_gps: Whether to prefer images with GPS data.
            diversity_weight: Weight for diversity vs quality (0-1).
        """
        super().__init__(target_count, scorer, prefer_gps)
        self._diversity_weight = diversity_weight

    def select(self, records: List[ImageRecord]) -> List[ImageRecord]:
        """Select images with diversity considerations.
        
        Uses a greedy algorithm that balances quality with
        diversity from already-selected images.
        
        Args:
            records: List of candidate image records.
            
        Returns:
            List of selected records.
        """
        if not records:
            return []

        if len(records) <= self._target_count:
            return self._sort_by_quality(records)

        # Initial scoring
        scored = [
            (r, self._compute_selection_score(r))
            for r in records
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Greedy selection with diversity
        selected: List[ImageRecord] = []
        remaining = list(scored)

        while len(selected) < self._target_count and remaining:
            if not selected:
                # First selection: pick highest quality
                best = remaining.pop(0)
                selected.append(best[0])
                continue

            # Find best considering diversity
            best_idx = 0
            best_combined = 0.0

            for i, (record, quality) in enumerate(remaining):
                diversity = self._compute_diversity(record, selected)
                combined = (
                    quality * (1 - self._diversity_weight)
                    + diversity * self._diversity_weight
                )
                if combined > best_combined:
                    best_combined = combined
                    best_idx = i

            selected.append(remaining[best_idx][0])
            remaining.pop(best_idx)

        return selected

    def _compute_diversity(
        self,
        candidate: ImageRecord,
        selected: List[ImageRecord],
    ) -> float:
        """Compute diversity score relative to selected images.
        
        Args:
            candidate: Candidate image record.
            selected: Already selected records.
            
        Returns:
            Diversity score (higher = more different from selected).
        """
        if not selected:
            return 1.0

        # Consider GPS distance if available
        if candidate.has_gps:
            gps_selected = [r for r in selected if r.has_gps]
            if gps_selected:
                # Use minimum distance to any selected image
                min_dist = min(
                    candidate.gps.distance_to(r.gps)  # type: ignore
                    for r in gps_selected
                )
                # Normalize: 1km+ = max diversity
                gps_diversity = min(min_dist / 1000.0, 1.0)
                return gps_diversity

        # Fallback: use hash distance
        min_hash_dist = min(
            (candidate.phash ^ r.phash).bit_count()
            for r in selected
        )
        # Normalize: 32+ bits different = max diversity
        return min(min_hash_dist / 32.0, 1.0)
