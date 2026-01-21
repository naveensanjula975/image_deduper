"""Report generation module.

This module handles creating JSON and summary reports
for the deduplication pipeline results.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from image_deduper_v2.models import DeduplicationResult, ImageRecord
from image_deduper_v2.utils.logging import get_logger


class ReportGenerator:
    """Generates reports for deduplication results.
    
    Supports JSON export and summary statistics.
    
    Attributes:
        output_path: Path to write the report.
    """

    def __init__(self, output_path: Path | None = None) -> None:
        """Initialize the report generator.
        
        Args:
            output_path: Optional path for report output.
        """
        self._output_path = output_path
        self._logger = get_logger("reporter")

    @property
    def output_path(self) -> Path | None:
        """Return the output path."""
        return self._output_path

    def generate_json_report(
        self,
        result: DeduplicationResult,
        selected_records: List[ImageRecord],
        settings_dict: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Generate a comprehensive JSON report.
        
        Args:
            result: Deduplication result object.
            selected_records: List of selected image records.
            settings_dict: Optional settings dictionary to include.
            
        Returns:
            Dictionary containing the full report.
        """
        report = {
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_scanned": result.total_scanned,
                "total_analyzed": result.total_analyzed,
                "total_duplicates_removed": result.total_duplicates_removed,
                "exact_duplicates": result.exact_duplicates_removed,
                "near_duplicates": result.near_duplicates_removed,
                "gps_duplicates": result.gps_duplicates_removed,
                "selected_count": result.selected_count,
                "exported_count": result.exported_count,
                "duration_seconds": round(result.duration_seconds, 2),
            },
            "selected_images": [
                self._record_to_dict(record)
                for record in selected_records
            ],
            "skipped_files": [
                {"path": path, "reason": reason}
                for path, reason in result.skipped_files
            ],
        }

        if settings_dict:
            report["settings"] = settings_dict

        return report

    def save_json_report(
        self,
        result: DeduplicationResult,
        selected_records: List[ImageRecord],
        output_path: Path | None = None,
        settings_dict: Dict[str, Any] | None = None,
    ) -> Path:
        """Generate and save a JSON report to file.
        
        Args:
            result: Deduplication result object.
            selected_records: List of selected image records.
            output_path: Optional override for output path.
            settings_dict: Optional settings to include.
            
        Returns:
            Path to the saved report file.
            
        Raises:
            ValueError: If no output path is configured.
        """
        path = output_path or self._output_path
        if path is None:
            raise ValueError("No output path configured for report")

        report = self.generate_json_report(result, selected_records, settings_dict)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        self._logger.info(f"Report saved to: {path}")
        return path

    def generate_summary(
        self,
        result: DeduplicationResult,
        selected_records: List[ImageRecord],
    ) -> str:
        """Generate a human-readable summary.
        
        Args:
            result: Deduplication result object.
            selected_records: List of selected image records.
            
        Returns:
            Formatted summary string.
        """
        lines = [
            "=" * 60,
            "IMAGE DEDUPLICATION SUMMARY",
            "=" * 60,
            "",
            f"Total files scanned:      {result.total_scanned:,}",
            f"Successfully analyzed:    {result.total_analyzed:,}",
            "",
            "Duplicates Removed:",
            f"  - Exact duplicates:     {result.exact_duplicates_removed:,}",
            f"  - Near duplicates:      {result.near_duplicates_removed:,}",
            f"  - GPS duplicates:       {result.gps_duplicates_removed:,}",
            f"  - Total removed:        {result.total_duplicates_removed:,}",
            "",
            f"Images selected:          {result.selected_count:,}",
            f"Images exported:          {result.exported_count:,}",
            f"Processing time:          {result.duration_seconds:.1f} seconds",
            "",
        ]

        if selected_records:
            # Quality statistics
            qualities = [r.metrics.quality_score for r in selected_records]
            avg_quality = sum(qualities) / len(qualities)
            
            resolutions = [r.metrics.resolution for r in selected_records]
            avg_mp = sum(resolutions) / len(resolutions) / 1_000_000
            
            gps_count = sum(1 for r in selected_records if r.has_gps)

            lines.extend([
                "Selected Image Statistics:",
                f"  - Average quality:      {avg_quality:.3f}",
                f"  - Average resolution:   {avg_mp:.1f} MP",
                f"  - Images with GPS:      {gps_count:,} ({gps_count/len(selected_records)*100:.1f}%)",
                "",
            ])

        if result.skipped_files:
            lines.append(f"Skipped files:            {len(result.skipped_files):,}")

        lines.extend([
            "=" * 60,
        ])

        return "\n".join(lines)

    def _record_to_dict(self, record: ImageRecord) -> Dict[str, Any]:
        """Convert an image record to a dictionary.
        
        Args:
            record: Image record to convert.
            
        Returns:
            Dictionary representation.
        """
        return {
            "path": str(record.path),
            "filename": record.original_filename,
            "sha256": record.sha256,
            "phash": hex(record.phash),
            "dimensions": {
                "width": record.metrics.width,
                "height": record.metrics.height,
                "megapixels": round(record.metrics.megapixels, 2),
            },
            "quality": {
                "score": round(record.metrics.quality_score, 4),
                "sharpness": round(record.metrics.sharpness, 2),
                "colorfulness": round(record.metrics.colorfulness, 2),
            },
            "file_size_bytes": record.metrics.file_size,
            "gps": {
                "latitude": record.gps.latitude,
                "longitude": record.gps.longitude,
            } if record.gps else None,
            "capture_time": record.capture_time.isoformat() if record.capture_time else None,
        }


def print_summary(
    result: DeduplicationResult,
    selected_records: List[ImageRecord],
) -> None:
    """Print a summary to the console.
    
    Args:
        result: Deduplication result object.
        selected_records: List of selected image records.
    """
    generator = ReportGenerator()
    print(generator.generate_summary(result, selected_records))
