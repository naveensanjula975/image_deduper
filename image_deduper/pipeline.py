from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from image_deduper.config import AppSettings
from image_deduper.dedupe import cluster_near_duplicates, cluster_by_gps_location
from image_deduper.image_ops import analyze_image, iter_image_files
from image_deduper.io_ops import export_selected, prepare_output_dir
from image_deduper.models import ImageRecord


class ImageDeduper:
    """Deduplicates images (exact + near-duplicate + GPS) and selects a target count."""

    def __init__(self, settings: AppSettings, logger: Optional[logging.Logger] = None) -> None:
        """Initializes the deduper.

        Args:
            settings: Application settings.
            logger: Optional logger.
        """
        self._settings = settings
        self._logger = logger or logging.getLogger("image_deduper")

    @property
    def settings(self) -> AppSettings:
        """Returns current settings."""
        return self._settings

    @property
    def logger(self) -> logging.Logger:
        """Returns the logger."""
        return self._logger

    def _build_records(self) -> List[ImageRecord]:
        """Builds image records for all discovered images.

        Returns:
            List of ImageRecord objects.
        """
        files = list(
            iter_image_files(
                input_paths=self.settings.input_paths,
                allowed_exts=self.settings.allowed_extensions_set,
                recursive=self.settings.recursive,
            )
        )
        self.logger.info(f"Discovered {len(files)} image files")

        records: List[ImageRecord] = []
        with ThreadPoolExecutor(max_workers=self.settings.workers) as ex:
            futures = {
                ex.submit(
                    analyze_image,
                    p,
                    self.settings.phash_size,
                    self.logger,
                    self.settings.enable_metadata_extraction,
                ): p
                for p in files
            }
            for fut in as_completed(futures):
                path = futures[fut]
                try:
                    analysis = fut.result()
                except Exception as e:
                    self.logger.warning(f"Analysis failed unexpectedly: {path} ({e})")
                    continue
                if analysis is None:
                    continue
                try:
                    stat = path.stat()
                    records.append(
                        ImageRecord(
                            path=path,
                            file_size=int(stat.st_size),
                            width=analysis.width,
                            height=analysis.height,
                            sha256=analysis.sha256,
                            phash=analysis.phash,
                            quality=float(analysis.quality),
                            latitude=analysis.latitude,
                            longitude=analysis.longitude,
                            capture_time=analysis.capture_time,
                        )
                    )
                except OSError as e:
                    self.logger.warning(f"Failed to stat file: {path} ({e})")

        gps_count = sum(1 for r in records if r.has_gps)
        self.logger.info(f"Analyzed {len(records)} images successfully ({gps_count} with GPS data)")
        return records

    def _dedupe_exact(self, records: List[ImageRecord]) -> List[ImageRecord]:
        """Deduplicates exact duplicates using SHA-256, keeping the best per hash.

        Args:
            records: Image records.

        Returns:
            Deduplicated records.
        """
        by_sha: Dict[str, List[ImageRecord]] = {}
        for r in records:
            by_sha.setdefault(r.sha256, []).append(r)

        kept: List[ImageRecord] = []
        removed = 0
        for group in by_sha.values():
            group_sorted = sorted(
                group,
                key=lambda x: (
                    x.has_gps if self.settings.prefer_gps_images else False,
                    x.quality,
                    x.pixels,
                    -x.file_size,
                ),
                reverse=True,
            )
            kept.append(group_sorted[0])
            removed += max(0, len(group_sorted) - 1)
        self.logger.info(f"Exact dedupe removed {removed} files")
        return kept

    def _dedupe_near(self, records: List[ImageRecord]) -> List[ImageRecord]:
        """Deduplicates near-duplicates using perceptual hash clustering.

        Args:
            records: Image records.

        Returns:
            Near-deduplicated records.
        """
        if not records:
            return []
        phashes = [r.phash for r in records]
        total_bits = self.settings.phash_size * self.settings.phash_size
        clusters = cluster_near_duplicates(
            phashes=phashes,
            threshold=self.settings.hamming_threshold,
            total_bits=total_bits,
        )
        selected: List[ImageRecord] = []
        removed = 0
        for cluster in clusters:
            group = [records[i] for i in cluster]
            group_sorted = sorted(
                group,
                key=lambda x: (
                    x.has_gps if self.settings.prefer_gps_images else False,
                    x.quality,
                    x.pixels,
                    -x.file_size,
                ),
                reverse=True,
            )
            selected.append(group_sorted[0])
            removed += max(0, len(group_sorted) - 1)
        self.logger.info(f"Near-duplicate clustering removed {removed} files")
        return selected

    def _dedupe_by_gps(self, records: List[ImageRecord]) -> List[ImageRecord]:
        """Deduplicates images taken at the same GPS location.

        Args:
            records: Image records.

        Returns:
            GPS-deduplicated records.
        """
        if not records:
            return []

        coordinates = [(r.latitude, r.longitude) for r in records]
        clusters = cluster_by_gps_location(
            coordinates=coordinates,
            distance_threshold=self.settings.gps_distance_threshold,
        )

        selected: List[ImageRecord] = []
        removed = 0
        for cluster in clusters:
            group = [records[i] for i in cluster]
            group_sorted = sorted(
                group,
                key=lambda x: (x.quality, x.pixels, -x.file_size),
                reverse=True,
            )
            selected.append(group_sorted[0])
            removed += max(0, len(group_sorted) - 1)

        self.logger.info(f"GPS location clustering removed {removed} files")
        return selected

    def _select_top_n(self, records: List[ImageRecord], n: int) -> List[ImageRecord]:
        """Selects top-N images by quality and resolution.

        Args:
            records: Candidate records.
            n: Target count.

        Returns:
            Selected records.
        """
        if n <= 0:
            return []
        records_sorted = sorted(
            records,
            key=lambda x: (
                x.has_gps if self.settings.prefer_gps_images else False,
                x.quality,
                x.pixels,
                -x.file_size,
            ),
            reverse=True,
        )
        return records_sorted[: min(n, len(records_sorted))]

    def run(self) -> List[Path]:
        """Runs the pipeline: analyze -> exact dedupe -> near dedupe -> GPS dedupe -> select -> export.

        Returns:
            Paths to exported images.
        """
        self.settings.ensure_output_dir()
        prepare_output_dir(self.settings.output_dir, self.settings.overwrite_output, self.logger)

        records = self._build_records()
        records = self._dedupe_exact(records)
        records = self._dedupe_near(records)

        if self.settings.enable_gps_filter:
            records = self._dedupe_by_gps(records)

        selected = self._select_top_n(records, self.settings.target_count)

        self.logger.info(f"Selected {len(selected)} unique images for export")
        exported = export_selected(
            selected_paths=[r.path for r in selected],
            output_dir=self.settings.output_dir,
            copy_mode=self.settings.copy_mode,
            logger=self.logger,
        )
        self.logger.info(f"Exported {len(exported)} images to {self.settings.output_dir}")
        return exported
