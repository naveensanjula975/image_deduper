"""File scanning and discovery module.

This module handles finding image files in input directories,
filtering by extension and file attributes.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator, Iterable, Set

from image_deduper_v2.utils.logging import get_logger


def iter_image_files(
    input_paths: Iterable[Path],
    allowed_extensions: Set[str],
    recursive: bool = True,
    min_size_bytes: int = 0,
    max_size_bytes: int = 0,
    logger: logging.Logger | None = None,
) -> Generator[Path, None, None]:
    """Yield image files from input paths.
    
    This is a generator function that yields files one at a time,
    minimizing memory usage for large directory trees.
    
    Args:
        input_paths: Files or directories to scan.
        allowed_extensions: Set of allowed file extensions (lowercase, with dot).
        recursive: Whether to scan directories recursively.
        min_size_bytes: Minimum file size in bytes (0 = no minimum).
        max_size_bytes: Maximum file size in bytes (0 = no maximum).
        logger: Optional logger for debug output.
        
    Yields:
        Path objects for matching image files.
        
    Example:
        >>> extensions = {".jpg", ".png"}
        >>> for path in iter_image_files([Path("./photos")], extensions):
        ...     print(path)
    """
    log = logger or get_logger("scanner")

    for input_path in input_paths:
        path = Path(input_path)
        
        if not path.exists():
            log.warning(f"Input path does not exist: {path}")
            continue

        if path.is_file():
            # Single file
            if _is_valid_image_file(
                path, allowed_extensions, min_size_bytes, max_size_bytes, log
            ):
                yield path
            continue

        if path.is_dir():
            # Directory
            yield from _scan_directory(
                path,
                allowed_extensions,
                recursive,
                min_size_bytes,
                max_size_bytes,
                log,
            )


def _is_valid_image_file(
    path: Path,
    allowed_extensions: Set[str],
    min_size_bytes: int,
    max_size_bytes: int,
    logger: logging.Logger,
) -> bool:
    """Check if a file is a valid image file.
    
    Args:
        path: Path to the file.
        allowed_extensions: Set of allowed extensions.
        min_size_bytes: Minimum file size (0 = no minimum).
        max_size_bytes: Maximum file size (0 = no maximum).
        logger: Logger instance.
        
    Returns:
        True if the file is valid, False otherwise.
    """
    # Check extension
    ext = path.suffix.lower()
    if ext not in allowed_extensions:
        return False

    # Check file size
    try:
        size = path.stat().st_size
    except OSError as e:
        logger.debug(f"Cannot stat file: {path} ({e})")
        return False

    if min_size_bytes > 0 and size < min_size_bytes:
        logger.debug(f"File too small: {path} ({size} bytes)")
        return False

    if max_size_bytes > 0 and size > max_size_bytes:
        logger.debug(f"File too large: {path} ({size} bytes)")
        return False

    return True


def _scan_directory(
    directory: Path,
    allowed_extensions: Set[str],
    recursive: bool,
    min_size_bytes: int,
    max_size_bytes: int,
    logger: logging.Logger,
) -> Generator[Path, None, None]:
    """Scan a directory for image files.
    
    Args:
        directory: Directory to scan.
        allowed_extensions: Set of allowed extensions.
        recursive: Whether to scan recursively.
        min_size_bytes: Minimum file size.
        max_size_bytes: Maximum file size.
        logger: Logger instance.
        
    Yields:
        Path objects for matching image files.
    """
    try:
        if recursive:
            iterator = directory.rglob("*")
        else:
            iterator = directory.glob("*")

        for child in iterator:
            if not child.is_file():
                continue
            if _is_valid_image_file(
                child, allowed_extensions, min_size_bytes, max_size_bytes, logger
            ):
                yield child

    except PermissionError as e:
        logger.warning(f"Permission denied scanning directory: {directory} ({e})")
    except OSError as e:
        logger.warning(f"Error scanning directory: {directory} ({e})")


class ImageScanner:
    """Class-based scanner for image file discovery.
    
    Provides additional functionality like counting and filtering
    on top of the generator function.
    
    Attributes:
        input_paths: List of input paths to scan.
        allowed_extensions: Set of allowed file extensions.
        recursive: Whether to scan recursively.
        min_size_bytes: Minimum file size filter.
        max_size_bytes: Maximum file size filter.
    """

    def __init__(
        self,
        input_paths: list[Path],
        allowed_extensions: Set[str],
        recursive: bool = True,
        min_size_bytes: int = 0,
        max_size_bytes: int = 0,
    ) -> None:
        """Initialize the scanner.
        
        Args:
            input_paths: List of directories or files to scan.
            allowed_extensions: Set of allowed extensions (lowercase with dot).
            recursive: Whether to scan directories recursively.
            min_size_bytes: Minimum file size in bytes (0 = no minimum).
            max_size_bytes: Maximum file size in bytes (0 = no maximum).
        """
        self._input_paths = input_paths
        self._allowed_extensions = allowed_extensions
        self._recursive = recursive
        self._min_size_bytes = min_size_bytes
        self._max_size_bytes = max_size_bytes
        self._logger = get_logger("scanner")

    @property
    def input_paths(self) -> list[Path]:
        """Return the input paths."""
        return self._input_paths

    @property
    def allowed_extensions(self) -> Set[str]:
        """Return the allowed extensions set."""
        return self._allowed_extensions

    def scan(self) -> Generator[Path, None, None]:
        """Scan for image files.
        
        Yields:
            Path objects for matching image files.
        """
        return iter_image_files(
            input_paths=self._input_paths,
            allowed_extensions=self._allowed_extensions,
            recursive=self._recursive,
            min_size_bytes=self._min_size_bytes,
            max_size_bytes=self._max_size_bytes,
            logger=self._logger,
        )

    def scan_to_list(self) -> list[Path]:
        """Scan and collect all files into a list.
        
        Warning: This loads all paths into memory at once.
        
        Returns:
            List of all matching file paths.
        """
        return list(self.scan())

    def count(self) -> int:
        """Count the number of matching files without loading paths.
        
        Returns:
            Total number of matching files.
        """
        return sum(1 for _ in self.scan())
