from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable, Optional, Tuple

from PIL import Image, ImageOps, UnidentifiedImageError
from PIL.ExifTags import GPSTAGS, TAGS


@dataclass(frozen=True)
class ImageAnalysis:
    """Represents computed features for an image file."""

    width: int
    height: int
    sha256: str
    phash: int
    quality: float
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    capture_time: Optional[str] = None


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


def _convert_gps_to_degrees(value: Tuple) -> Optional[float]:
    """Converts GPS coordinates from EXIF format to decimal degrees.

    Args:
        value: GPS coordinate tuple from EXIF (degrees, minutes, seconds).

    Returns:
        Decimal degrees or None if conversion fails.
    """
    try:
        degrees = float(value[0])
        minutes = float(value[1])
        seconds = float(value[2])
        return degrees + (minutes / 60.0) + (seconds / 3600.0)
    except (TypeError, IndexError, ValueError, ZeroDivisionError):
        return None


def _extract_gps_from_exif(exif_data: dict) -> Tuple[Optional[float], Optional[float]]:
    """Extracts latitude and longitude from EXIF data.

    Args:
        exif_data: Raw EXIF data dictionary.

    Returns:
        Tuple of (latitude, longitude) or (None, None).
    """
    gps_info = None
    for key, value in exif_data.items():
        tag_name = TAGS.get(key, key)
        if tag_name == "GPSInfo":
            gps_info = {}
            for gps_key, gps_value in value.items():
                gps_tag_name = GPSTAGS.get(gps_key, gps_key)
                gps_info[gps_tag_name] = gps_value
            break

    if not gps_info:
        return None, None

    lat = gps_info.get("GPSLatitude")
    lat_ref = gps_info.get("GPSLatitudeRef")
    lon = gps_info.get("GPSLongitude")
    lon_ref = gps_info.get("GPSLongitudeRef")

    if not all([lat, lat_ref, lon, lon_ref]):
        return None, None

    latitude = _convert_gps_to_degrees(lat)
    longitude = _convert_gps_to_degrees(lon)

    if latitude is None or longitude is None:
        return None, None

    if lat_ref == "S":
        latitude = -latitude
    if lon_ref == "W":
        longitude = -longitude

    return latitude, longitude


def _extract_capture_time_from_exif(exif_data: dict) -> Optional[str]:
    """Extracts capture timestamp from EXIF data.

    Args:
        exif_data: Raw EXIF data dictionary.

    Returns:
        DateTime string if found, otherwise None.
    """
    for key, value in exif_data.items():
        tag_name = TAGS.get(key, key)
        if tag_name in ("DateTimeOriginal", "DateTime", "DateTimeDigitized"):
            return str(value)
    return None


def analyze_image(
    path: Path,
    phash_size: int,
    logger: logging.Logger,
    extract_metadata: bool = True,
) -> Optional[ImageAnalysis]:
    """Loads and analyzes an image, returning computed features.

    Args:
        path: Image file path.
        phash_size: Hash grid size.
        logger: Logger instance.
        extract_metadata: Whether to extract GPS and timestamp metadata.

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

            latitude = None
            longitude = None
            capture_time = None

            if extract_metadata:
                try:
                    exif_data = im._getexif()
                    if exif_data:
                        latitude, longitude = _extract_gps_from_exif(exif_data)
                        capture_time = _extract_capture_time_from_exif(exif_data)
                except (AttributeError, KeyError, TypeError) as e:
                    logger.debug(f"Failed to extract metadata from {path}: {e}")

            return ImageAnalysis(
                width=width,
                height=height,
                sha256=sha,
                phash=phash,
                quality=quality,
                latitude=latitude,
                longitude=longitude,
                capture_time=capture_time,
            )
    except (UnidentifiedImageError, OSError, ValueError) as e:
        logger.warning(f"Failed to open/analyze image: {path} ({e})")
        return None
