from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable, Optional, Tuple

from PIL import Image, ImageOps, UnidentifiedImageError


@dataclass(frozen=True)
class ImageAnalysis:
    """Represents computed features for an image file."""

    width: int
    height: int
    sha256: str
    phash: int
    quality: float


def iter_image_files(
    input_paths: Iterable[Path],
    allowed_exts: set[str],
    recursive: bool,
) -> Generator[Path, None, None]:
    """Yields image file paths from inputs.

    Args:
        input_paths: Files or directories to scan.
        allowed_exts: Allowed file extensions.
        recursive: Whether to scan directories recursively.

    Yields:
        Image file paths.
    """
    for p in (Path(x) for x in input_paths):
        if p.is_file():
            if p.suffix.lower() in allowed_exts:
                yield p
            continue
        if not p.exists():
            continue
        if p.is_dir():
            if recursive:
                for child in p.rglob("*"):
                    if child.is_file() and child.suffix.lower() in allowed_exts:
                        yield child
            else:
                for child in p.glob("*"):
                    if child.is_file() and child.suffix.lower() in allowed_exts:
                        yield child


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Computes SHA-256 of a file.

    Args:
        path: File path.
        chunk_size: Read chunk size in bytes.

    Returns:
        Hex digest SHA-256.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_ahash(img: Image.Image, size: int = 8) -> int:
    """Computes average hash (aHash) for an image as an integer bitset.

    Args:
        img: PIL image.
        size: Hash grid size; bits = size*size.

    Returns:
        Integer representing the hash.
    """
    gray = ImageOps.grayscale(img).resize((size, size), Image.Resampling.LANCZOS)
    pixels = list(gray.getdata())
    avg = sum(pixels) / len(pixels) if pixels else 0.0
    bits = 0
    for i, px in enumerate(pixels):
        if px >= avg:
            bits |= 1 << i
    return bits


def _laplacian_variance_gray(img: Image.Image) -> float:
    """Approximates sharpness using a discrete Laplacian variance on a grayscale image.

    Args:
        img: PIL image.

    Returns:
        Variance of Laplacian response.
    """
    g = ImageOps.grayscale(img)
    w, h = g.size
    if w < 3 or h < 3:
        return 0.0
    p = g.load()
    values = []
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            center = p[x, y]
            lap = (
                (p[x - 1, y] + p[x + 1, y] + p[x, y - 1] + p[x, y + 1])
                - (4 * center)
            )
            values.append(float(lap))
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return var


def analyze_image(
    path: Path,
    phash_size: int,
    logger: logging.Logger,
) -> Optional[ImageAnalysis]:
    """Loads and analyzes an image, returning computed features.

    Args:
        path: Image file path.
        phash_size: Hash grid size.
        logger: Logger instance.

    Returns:
        ImageAnalysis if successful; otherwise None.
    """
    try:
        sha = sha256_file(path)
    except OSError as e:
        logger.warning(f"Failed to read file for hashing: {path} ({e})")
        return None

    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            width, height = im.size
            phash = compute_ahash(im, size=phash_size)
            base_w = min(512, max(64, width))
            base_h = max(1, int(height * (base_w / max(1, width))))
            thumb = im.resize((base_w, base_h), Image.Resampling.LANCZOS)
            sharp = _laplacian_variance_gray(thumb)
            pixels = float(width * height)
            quality = (sharp + 1.0) * (pixels ** 0.5)
            return ImageAnalysis(width=width, height=height, sha256=sha, phash=phash, quality=quality)
    except (UnidentifiedImageError, OSError, ValueError) as e:
        logger.warning(f"Failed to open/analyze image: {path} ({e})")
        return None
