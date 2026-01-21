"""Main pipeline orchestrator.

This module provides the primary ImageDeduper class that
coordinates all stages of the deduplication pipeline.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional

from image_deduper_v2.analyzer import ImageAnalyzer
from image_deduper_v2.config import Settings
from image_deduper_v2.deduplicator import (
    CompositeDeduplicator,
    ExactHashDeduplicator,
    GPSProximityDeduplicator,
    PerceptualHashDeduplicator,
)
from image_deduper_v2.exporter import ImageExporter
from image_deduper_v2.models import DeduplicationResult, ImageRecord
from image_deduper_v2.protocols import ProgressCallback, ScoringStrategy
from image_deduper_v2.reporter import ReportGenerator, print_summary
from image_deduper_v2.scanner import ImageScanner
from image_deduper_v2.selector import ImageSelector
from image_deduper_v2.utils.logging import configure_logging, get_logger


class ImageDeduper:
    """Main orchestrator for the image deduplication pipeline.
    
    Coordinates file scanning, image analysis, deduplication,
    selection, and export into a cohesive workflow.
    
    Attributes:
        settings: Application settings.
        scoring_strategy: Optional custom scoring strategy.
        progress_callback: Optional progress callback.
    
    Example:
        >>> settings = Settings(
        ...     input_paths=["./photos"],
        ...     output_dir="./selected",
        ...     target_count=200,
        ... )
        >>> deduper = ImageDeduper(settings)
        >>> result = deduper.run()
        >>> print(f"Exported {result.exported_count} images")
    """

    def __init__(
        self,
        settings: Settings,
        scoring_strategy: ScoringStrategy | None = None,
        progress_callback: ProgressCallback | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the deduper pipeline.
        
        Args:
            settings: Application settings.
            scoring_strategy: Optional custom scoring strategy.
            progress_callback: Optional progress callback.
            logger: Optional logger instance.
        """
        self._settings = settings
        self._scoring_strategy = scoring_strategy
        self._progress_callback = progress_callback
        self._logger = logger or get_logger()
        
        # Initialize components
        self._scanner = ImageScanner(
            input_paths=settings.input_paths,
            allowed_extensions=settings.allowed_extensions_set,
            recursive=settings.recursive,
            min_size_bytes=0,
            max_size_bytes=settings.max_file_size_bytes,
        )
        
        self._analyzer = ImageAnalyzer(
            hash_size=settings.phash_size,
            extract_metadata=settings.enable_metadata_extraction,
        )
        
        self._selector = ImageSelector.from_settings(
            settings,
            scorer=scoring_strategy,
        )
        
        self._exporter = ImageExporter(
            output_dir=settings.output_dir,
            use_copy=settings.copy_mode,
            overwrite=settings.overwrite_output,
        )

    @property
    def settings(self) -> Settings:
        """Return the settings."""
        return self._settings

    @property
    def logger(self) -> logging.Logger:
        """Return the logger."""
        return self._logger

    def run(self) -> DeduplicationResult:
        """Run the complete deduplication pipeline.
        
        Executes all stages: scan -> analyze -> dedupe -> select -> export.
        
        Returns:
            DeduplicationResult with statistics and exported paths.
        """
        start_time = time.time()
        result = DeduplicationResult()

        try:
            # Stage 1: Scan for images
            self._logger.info("Stage 1/5: Scanning for images...")
            files = self._scan()
            result.total_scanned = len(files)
            self._logger.info(f"Found {len(files)} image files")

            if not files:
                self._logger.warning("No image files found, exiting")
                result.duration_seconds = time.time() - start_time
                return result

            # Stage 2: Analyze images
            self._logger.info("Stage 2/5: Analyzing images...")
            records = self._analyze(files)
            result.total_analyzed = len(records)
            self._logger.info(f"Successfully analyzed {len(records)} images")

            if not records:
                self._logger.warning("No images analyzed successfully, exiting")
                result.duration_seconds = time.time() - start_time
                return result

            # Stage 3: Deduplicate
            self._logger.info("Stage 3/5: Removing duplicates...")
            records, removed_counts = self._deduplicate(records)
            result.exact_duplicates_removed = removed_counts.get("ExactHash", 0)
            result.near_duplicates_removed = removed_counts.get("PerceptualHash", 0)
            result.gps_duplicates_removed = removed_counts.get("GPSProximity", 0)
            self._logger.info(f"{len(records)} unique images remaining")

            # Stage 4: Select top N
            self._logger.info("Stage 4/5: Selecting best images...")
            selected = self._select(records)
            result.selected_count = len(selected)
            self._logger.info(f"Selected {len(selected)} images for export")

            # Stage 5: Export
            self._logger.info("Stage 5/5: Exporting images...")
            exported = self._export(selected)
            result.exported_count = len(exported)
            result.exported_paths = exported
            self._logger.info(f"Exported {len(exported)} images to {self._settings.output_dir}")

            # Generate report if requested
            if self._settings.report_path:
                self._generate_report(result, selected)

            # Print summary
            print_summary(result, selected)

        except Exception as e:
            self._logger.exception(f"Pipeline failed: {e}")
            raise

        finally:
            result.duration_seconds = time.time() - start_time

        return result

    def _scan(self) -> List[Path]:
        """Scan for image files.
        
        Returns:
            List of discovered file paths.
        """
        if self._progress_callback:
            self._progress_callback.on_stage_start("scanning", 0)

        files = self._scanner.scan_to_list()

        if self._progress_callback:
            self._progress_callback.on_stage_complete(
                "scanning", f"Found {len(files)} files"
            )

        return files

    def _analyze(self, files: List[Path]) -> List[ImageRecord]:
        """Analyze image files.
        
        Args:
            files: List of file paths to analyze.
            
        Returns:
            List of successfully analyzed image records.
        """
        if self._progress_callback:
            self._progress_callback.on_stage_start("analyzing", len(files))

        records = self._analyzer.analyze_many(
            files,
            max_workers=self._settings.workers,
        )

        # Report GPS statistics
        gps_count = sum(1 for r in records if r.has_gps)
        self._logger.info(f"  {gps_count} images have GPS data")

        if self._progress_callback:
            self._progress_callback.on_stage_complete(
                "analyzing", f"Analyzed {len(records)} images"
            )

        return records

    def _deduplicate(
        self,
        records: List[ImageRecord],
    ) -> tuple[List[ImageRecord], dict[str, int]]:
        """Apply deduplication strategies.
        
        Args:
            records: List of image records.
            
        Returns:
            Tuple of (deduplicated records, removal counts by strategy).
        """
        if self._progress_callback:
            self._progress_callback.on_stage_start("deduplicating", len(records))

        # Build strategy list
        strategies = [
            ExactHashDeduplicator(),
            PerceptualHashDeduplicator(
                threshold=self._settings.hamming_threshold,
                total_bits=self._settings.phash_size ** 2,
            ),
        ]

        if self._settings.enable_gps_filter:
            strategies.append(
                GPSProximityDeduplicator(
                    distance_threshold=self._settings.gps_distance_threshold
                )
            )

        # Apply strategies
        deduplicator = CompositeDeduplicator(
            strategies=strategies,
            prefer_gps=self._settings.prefer_gps_images,
        )
        
        result, removed_counts = deduplicator.deduplicate(records)

        if self._progress_callback:
            total_removed = sum(removed_counts.values())
            self._progress_callback.on_stage_complete(
                "deduplicating", f"Removed {total_removed} duplicates"
            )

        return result, removed_counts

    def _select(self, records: List[ImageRecord]) -> List[ImageRecord]:
        """Select top-N images.
        
        Args:
            records: List of candidate image records.
            
        Returns:
            List of selected records.
        """
        if self._progress_callback:
            self._progress_callback.on_stage_start("selecting", len(records))

        selected = self._selector.select(records)

        if self._progress_callback:
            self._progress_callback.on_stage_complete(
                "selecting", f"Selected {len(selected)} images"
            )

        return selected

    def _export(self, records: List[ImageRecord]) -> List[Path]:
        """Export selected images.
        
        Args:
            records: List of image records to export.
            
        Returns:
            List of exported file paths.
        """
        if self._progress_callback:
            self._progress_callback.on_stage_start("exporting", len(records))

        exported = self._exporter.export(records)

        if self._progress_callback:
            self._progress_callback.on_stage_complete(
                "exporting", f"Exported {len(exported)} files"
            )

        return exported

    def _generate_report(
        self,
        result: DeduplicationResult,
        selected: List[ImageRecord],
    ) -> None:
        """Generate and save report.
        
        Args:
            result: Deduplication result.
            selected: Selected image records.
        """
        if not self._settings.report_path:
            return

        reporter = ReportGenerator(self._settings.report_path)
        reporter.save_json_report(
            result,
            selected,
            settings_dict=self._settings.to_dict(),
        )
