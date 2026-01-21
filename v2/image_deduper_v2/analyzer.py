"""Image analysis module.

This module handles loading images, computing quality metrics,
and extracting metadata from image files.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from PIL import Image, ImageOps, ImageStat, UnidentifiedImageError
from PIL.ExifTags import GPSTAGS, TAGS

from image_deduper_v2.exceptions import AnalysisError
from image_deduper_v2.hasher import compute_phash, sha256_file
from image_deduper_v2.models import GPSCoordinates, ImageMetrics, ImageRecord
from image_deduper_v2.utils.logging import get_logger


def analyze_image(
    path: Path,
    hash_size: int = 16,
    extract_metadata: bool = True,
    logger: logging.Logger | None = None,
) -> ImageRecord | None:
    """Analyze an image file and compute all metrics.
    
    This function loads an image, computes hashes and quality metrics,
    and optionally extracts GPS and timestamp metadata.
    
    Args:
        path: Path to the image file.
        hash_size: Size parameter for perceptual hash.
        extract_metadata: Whether to extract EXIF metadata.
        logger: Optional logger instance.
        
    Returns:
        ImageRecord with all computed data, or None if analysis fails.
        
    Example:
        >>> record = analyze_image(Path("photo.jpg"))
        >>> if record:
        ...     print(f"Quality: {record.metrics.quality_score}")
    """
    log = logger or get_logger("analyzer")

    # Compute file hash
    try:
        sha256 = sha256_file(path)
    except OSError as e:
        log.warning(f"Failed to read file for hashing: {path} ({e})")
        return None

    # Open and analyze image
    try:
        with Image.open(path) as img:
            # Handle EXIF orientation
            img = ImageOps.exif_transpose(img)
            
            # Get dimensions
            width, height = img.size
            
            # Compute perceptual hash
            phash = compute_phash(img, hash_size)
            
            # Compute quality metrics
            metrics = _compute_quality_metrics(img, path)
            
            # Extract metadata if requested
            gps = None
            capture_time = None
            if extract_metadata:
                gps = _extract_gps(img, log)
                capture_time = _extract_capture_time(img, log)

            return ImageRecord(
                path=path,
                sha256=sha256,
                phash=phash,
                metrics=metrics,
                gps=gps,
                capture_time=capture_time,
            )

    except UnidentifiedImageError as e:
        log.warning(f"Cannot identify image format: {path}")
        return None
    except OSError as e:
        log.warning(f"Failed to open image: {path} ({e})")
        return None
    except Exception as e:
        log.warning(f"Unexpected error analyzing image: {path} ({e})")
        return None


def _compute_quality_metrics(img: Image.Image, path: Path) -> ImageMetrics:
    """Compute quality metrics for an image.
    
    Args:
        img: PIL Image object.
        path: Path to the image file (for file size).
        
    Returns:
        ImageMetrics object with computed values.
    """
    width, height = img.size
    
    # Compute sharpness using Laplacian variance
    sharpness = _compute_sharpness(img)
    
    # Compute colorfulness
    colorfulness = _compute_colorfulness(img)
    
    # Get file size
    try:
        file_size = path.stat().st_size
    except OSError:
        file_size = 0

    # Get bit depth
    bit_depth = 8
    if img.mode == "I;16" or img.mode == "I;16L" or img.mode == "I;16B":
        bit_depth = 16
    elif img.mode == "F":
        bit_depth = 32

    return ImageMetrics(
        width=width,
        height=height,
        sharpness=sharpness,
        colorfulness=colorfulness,
        file_size=file_size,
        bit_depth=bit_depth,
    )


def _compute_sharpness(img: Image.Image) -> float:
    """Compute sharpness using Laplacian variance.
    
    This method approximates sharpness by computing the variance
    of a discrete Laplacian filter applied to the grayscale image.
    
    Args:
        img: PIL Image object.
        
    Returns:
        Variance of Laplacian (higher = sharper).
    """
    # Convert to grayscale
    gray = ImageOps.grayscale(img)
    
    # Resize for faster processing
    max_dim = 512
    w, h = gray.size
    if w > max_dim or h > max_dim:
        scale = max_dim / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        gray = gray.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    w, h = gray.size
    if w < 3 or h < 3:
        return 0.0

    # Access pixel data
    pixels = gray.load()
    
    # Compute Laplacian
    values = []
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            center = pixels[x, y]
            laplacian = (
                pixels[x - 1, y]
                + pixels[x + 1, y]
                + pixels[x, y - 1]
                + pixels[x, y + 1]
                - 4 * center
            )
            values.append(float(laplacian))

    if not values:
        return 0.0

    # Compute variance
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    
    return variance


def _compute_colorfulness(img: Image.Image) -> float:
    """Compute colorfulness metric.
    
    Uses the method from Hasler and Süsstrunk (2003):
    "Measuring colourfulness in natural images"
    
    Args:
        img: PIL Image object.
        
    Returns:
        Colorfulness score (higher = more colorful).
    """
    # Convert to RGB if necessary
    if img.mode != "RGB":
        try:
            img = img.convert("RGB")
        except Exception:
            return 0.0

    # Resize for faster processing
    max_dim = 256
    w, h = img.size
    if w > max_dim or h > max_dim:
        scale = max_dim / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Get RGB channels
    r, g, b = img.split()
    
    # Compute opponent color channels
    # rg = R - G, yb = 0.5(R + G) - B
    pixels_r = list(r.getdata())
    pixels_g = list(g.getdata())
    pixels_b = list(b.getdata())
    
    n = len(pixels_r)
    if n == 0:
        return 0.0

    rg = [pixels_r[i] - pixels_g[i] for i in range(n)]
    yb = [0.5 * (pixels_r[i] + pixels_g[i]) - pixels_b[i] for i in range(n)]

    # Compute mean and std of opponent channels
    mean_rg = sum(rg) / n
    mean_yb = sum(yb) / n
    std_rg = (sum((v - mean_rg) ** 2 for v in rg) / n) ** 0.5
    std_yb = (sum((v - mean_yb) ** 2 for v in yb) / n) ** 0.5

    # Compute colorfulness
    std_rgyb = (std_rg ** 2 + std_yb ** 2) ** 0.5
    mean_rgyb = (mean_rg ** 2 + mean_yb ** 2) ** 0.5
    
    colorfulness = std_rgyb + 0.3 * mean_rgyb
    
    return colorfulness


def _extract_gps(img: Image.Image, logger: logging.Logger) -> GPSCoordinates | None:
    """Extract GPS coordinates from image EXIF data.
    
    Args:
        img: PIL Image object.
        logger: Logger instance.
        
    Returns:
        GPSCoordinates if found, None otherwise.
    """
    try:
        exif_data = img._getexif()
        if not exif_data:
            return None

        # Find GPSInfo tag
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
            return None

        # Extract coordinates
        lat = gps_info.get("GPSLatitude")
        lat_ref = gps_info.get("GPSLatitudeRef")
        lon = gps_info.get("GPSLongitude")
        lon_ref = gps_info.get("GPSLongitudeRef")

        if not all([lat, lat_ref, lon, lon_ref]):
            return None

        latitude = _convert_gps_to_degrees(lat)
        longitude = _convert_gps_to_degrees(lon)

        if latitude is None or longitude is None:
            return None

        if lat_ref == "S":
            latitude = -latitude
        if lon_ref == "W":
            longitude = -longitude

        return GPSCoordinates(latitude=latitude, longitude=longitude)

    except (AttributeError, KeyError, TypeError, ValueError) as e:
        logger.debug(f"Failed to extract GPS: {e}")
        return None


def _extract_capture_time(img: Image.Image, logger: logging.Logger) -> datetime | None:
    """Extract capture timestamp from image EXIF data.
    
    Args:
        img: PIL Image object.
        logger: Logger instance.
        
    Returns:
        datetime if found, None otherwise.
    """
    try:
        exif_data = img._getexif()
        if not exif_data:
            return None

        # Look for date/time tags
        date_tags = ("DateTimeOriginal", "DateTime", "DateTimeDigitized")
        
        for key, value in exif_data.items():
            tag_name = TAGS.get(key, key)
            if tag_name in date_tags:
                # Parse EXIF date format: "YYYY:MM:DD HH:MM:SS"
                try:
                    return datetime.strptime(str(value), "%Y:%m:%d %H:%M:%S")
                except ValueError:
                    continue

        return None

    except (AttributeError, KeyError, TypeError) as e:
        logger.debug(f"Failed to extract capture time: {e}")
        return None


def _convert_gps_to_degrees(value: tuple) -> float | None:
    """Convert GPS coordinates from EXIF format to decimal degrees.
    
    Args:
        value: GPS coordinate tuple (degrees, minutes, seconds).
        
    Returns:
        Decimal degrees, or None if conversion fails.
    """
    try:
        degrees = float(value[0])
        minutes = float(value[1])
        seconds = float(value[2])
        return degrees + (minutes / 60.0) + (seconds / 3600.0)
    except (TypeError, IndexError, ValueError, ZeroDivisionError):
        return None


class ImageAnalyzer:
    """Class for analyzing images with consistent settings.
    
    Provides a reusable analyzer with configurable options.
    
    Attributes:
        hash_size: Size parameter for perceptual hash.
        extract_metadata: Whether to extract EXIF metadata.
    """

    def __init__(
        self,
        hash_size: int = 16,
        extract_metadata: bool = True,
    ) -> None:
        """Initialize the analyzer.
        
        Args:
            hash_size: Size parameter for perceptual hash.
            extract_metadata: Whether to extract EXIF metadata.
        """
        self._hash_size = hash_size
        self._extract_metadata = extract_metadata
        self._logger = get_logger("analyzer")

    @property
    def hash_size(self) -> int:
        """Return the hash size."""
        return self._hash_size

    @property
    def extract_metadata(self) -> bool:
        """Return whether metadata extraction is enabled."""
        return self._extract_metadata

    def analyze(self, path: Path) -> ImageRecord | None:
        """Analyze an image file.
        
        Args:
            path: Path to the image file.
            
        Returns:
            ImageRecord if successful, None otherwise.
        """
        return analyze_image(
            path=path,
            hash_size=self._hash_size,
            extract_metadata=self._extract_metadata,
            logger=self._logger,
        )

    def analyze_many(
        self,
        paths: list[Path],
        max_workers: int = 8,
    ) -> list[ImageRecord]:
        """Analyze multiple images in parallel.
        
        Args:
            paths: List of image paths.
            max_workers: Number of worker threads.
            
        Returns:
            List of successful ImageRecord objects.
        """
        from image_deduper_v2.utils.concurrency import run_in_executor
        
        records: list[ImageRecord] = []
        
        for path, result, error in run_in_executor(
            self.analyze, paths, max_workers=max_workers
        ):
            if error:
                self._logger.warning(f"Analysis failed: {path} ({error})")
            elif result is not None:
                records.append(result)

        return records
