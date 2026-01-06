from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Iterable, List, Tuple


def _safe_clear_dir(path: Path, logger: logging.Logger) -> None:
    """Clears a directory contents safely.

    Args:
        path: Directory to clear.
        logger: Logger instance.
    """
    if not path.exists():
        return
    if not path.is_dir():
        raise ValueError(f"Output path exists but is not a directory: {path}")
    for child in path.iterdir():
        try:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        except OSError as e:
            logger.warning(f"Failed to remove: {child} ({e})")


def prepare_output_dir(output_dir: Path, overwrite: bool, logger: logging.Logger) -> None:
    """Prepares the output directory.

    Args:
        output_dir: Output directory.
        overwrite: Whether to clear if not empty.
        logger: Logger instance.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        _safe_clear_dir(output_dir, logger)


def _try_hardlink(src: Path, dst: Path) -> bool:
    """Attempts to hardlink a file.

    Args:
        src: Source file.
        dst: Destination file.

    Returns:
        True if hardlink succeeded; False otherwise.
    """
    try:
        os.link(src, dst)
        return True
    except OSError:
        return False


def export_selected(
    selected_paths: List[Path],
    output_dir: Path,
    copy_mode: bool,
    logger: logging.Logger,
) -> List[Path]:
    """Exports selected images into the output directory.

    Args:
        selected_paths: Source file paths.
        output_dir: Output directory.
        copy_mode: If True, copy; if False, hardlink when possible, else copy.
        logger: Logger instance.

    Returns:
        List of destination paths created.
    """
    created: List[Path] = []
    for i, src in enumerate(selected_paths, start=1):
        ext = src.suffix.lower() if src.suffix else ".img"
        dst = output_dir / f"{i:04d}{ext}"
        try:
            if dst.exists():
                dst.unlink()
            if copy_mode:
                shutil.copy2(src, dst)
            else:
                if not _try_hardlink(src, dst):
                    shutil.copy2(src, dst)
            created.append(dst)
        except OSError as e:
            logger.warning(f"Failed to export {src} -> {dst} ({e})")
    return created
