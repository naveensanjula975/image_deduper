"""File export module.

This module handles copying or linking selected images
to the output directory.
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import List

from image_deduper_v2.exceptions import ExportError
from image_deduper_v2.models import ImageRecord
from image_deduper_v2.utils.logging import get_logger


def prepare_output_dir(
    output_dir: Path,
    overwrite: bool = False,
    logger: logging.Logger | None = None,
) -> None:
    """Prepare the output directory for export.
    
    Creates the directory if it doesn't exist. Optionally clears
    existing contents if overwrite is True.
    
    Args:
        output_dir: Path to the output directory.
        overwrite: Whether to clear existing contents.
        logger: Optional logger instance.
        
    Raises:
        ExportError: If directory cannot be prepared.
    """
    log = logger or get_logger("exporter")
    
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise ExportError(
            f"Cannot create output directory: {output_dir}",
            details=str(e),
        ) from e

    if overwrite:
        _clear_directory(output_dir, log)


def _clear_directory(path: Path, logger: logging.Logger) -> None:
    """Clear contents of a directory.
    
    Args:
        path: Directory to clear.
        logger: Logger instance.
    """
    if not path.exists() or not path.is_dir():
        return

    for child in path.iterdir():
        try:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        except OSError as e:
            logger.warning(f"Failed to remove: {child} ({e})")


def export_image(
    source: Path,
    destination: Path,
    use_copy: bool = True,
) -> bool:
    """Export a single image to the destination.
    
    Args:
        source: Source image path.
        destination: Destination path.
        use_copy: If True, copy; if False, try hardlink first.
        
    Returns:
        True if successful, False otherwise.
    """
    try:
        # Remove existing destination
        if destination.exists():
            destination.unlink()

        if use_copy:
            shutil.copy2(source, destination)
        else:
            # Try hardlink first
            try:
                os.link(source, destination)
            except OSError:
                # Fall back to copy
                shutil.copy2(source, destination)
        
        return True

    except OSError:
        return False


def export_selected(
    records: List[ImageRecord],
    output_dir: Path,
    use_copy: bool = True,
    filename_pattern: str = "{index:04d}{ext}",
    preserve_names: bool = False,
    logger: logging.Logger | None = None,
) -> List[Path]:
    """Export selected images to the output directory.
    
    Args:
        records: List of image records to export.
        output_dir: Output directory path.
        use_copy: If True, copy files; if False, try hardlinks.
        filename_pattern: Pattern for generating filenames.
                         Supports {index}, {ext}, {original} placeholders.
        preserve_names: If True, use original filenames.
        logger: Optional logger instance.
        
    Returns:
        List of successfully exported destination paths.
        
    Example:
        >>> exported = export_selected(
        ...     records,
        ...     Path("./output"),
        ...     filename_pattern="{index:04d}_{original}"
        ... )
    """
    log = logger or get_logger("exporter")
    
    exported: List[Path] = []
    used_names: set[str] = set()

    for idx, record in enumerate(records, start=1):
        try:
            # Generate destination filename
            if preserve_names:
                base_name = record.original_filename
                # Handle duplicates
                if base_name in used_names:
                    stem = record.path.stem
                    ext = record.extension
                    counter = 1
                    while f"{stem}_{counter}{ext}" in used_names:
                        counter += 1
                    base_name = f"{stem}_{counter}{ext}"
            else:
                base_name = filename_pattern.format(
                    index=idx,
                    ext=record.extension,
                    original=record.path.stem,
                )
            
            used_names.add(base_name)
            destination = output_dir / base_name

            if export_image(record.path, destination, use_copy):
                exported.append(destination)
            else:
                log.warning(f"Failed to export: {record.path}")

        except Exception as e:
            log.warning(f"Failed to export {record.path}: {e}")

    return exported


class ImageExporter:
    """Class for exporting images with consistent settings.
    
    Provides additional features like progress reporting and
    batch export with error handling.
    
    Attributes:
        output_dir: Output directory path.
        use_copy: Whether to copy files (vs hardlink).
        overwrite: Whether to overwrite existing output.
        filename_pattern: Pattern for generating filenames.
    """

    def __init__(
        self,
        output_dir: Path,
        use_copy: bool = True,
        overwrite: bool = False,
        filename_pattern: str = "{index:04d}{ext}",
        preserve_names: bool = False,
    ) -> None:
        """Initialize the exporter.
        
        Args:
            output_dir: Output directory path.
            use_copy: If True, copy; if False, try hardlinks.
            overwrite: Whether to clear existing output.
            filename_pattern: Pattern for filenames.
            preserve_names: Whether to preserve original names.
        """
        self._output_dir = output_dir
        self._use_copy = use_copy
        self._overwrite = overwrite
        self._filename_pattern = filename_pattern
        self._preserve_names = preserve_names
        self._logger = get_logger("exporter")

    @property
    def output_dir(self) -> Path:
        """Return the output directory."""
        return self._output_dir

    def prepare(self) -> None:
        """Prepare the output directory.
        
        Creates the directory and optionally clears contents.
        """
        prepare_output_dir(
            self._output_dir,
            overwrite=self._overwrite,
            logger=self._logger,
        )

    def export(self, records: List[ImageRecord]) -> List[Path]:
        """Export all records to the output directory.
        
        Args:
            records: List of image records to export.
            
        Returns:
            List of successfully exported paths.
        """
        self.prepare()
        
        return export_selected(
            records=records,
            output_dir=self._output_dir,
            use_copy=self._use_copy,
            filename_pattern=self._filename_pattern,
            preserve_names=self._preserve_names,
            logger=self._logger,
        )

    def export_single(
        self,
        record: ImageRecord,
        custom_name: str | None = None,
    ) -> Path | None:
        """Export a single image.
        
        Args:
            record: Image record to export.
            custom_name: Optional custom filename.
            
        Returns:
            Destination path if successful, None otherwise.
        """
        if custom_name:
            destination = self._output_dir / custom_name
        else:
            destination = self._output_dir / record.original_filename

        if export_image(record.path, destination, self._use_copy):
            return destination
        return None
