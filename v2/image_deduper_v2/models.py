"""Data models for image records and analysis results.

This module defines the core data structures used throughout the
deduplication pipeline using frozen dataclasses for immutability.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class GPSCoordinates:
    """Represents GPS coordinates extracted from image EXIF data.
    
    Attributes:
        latitude: Latitude in decimal degrees (-90 to 90).
        longitude: Longitude in decimal degrees (-180 to 180).
    """
    
    latitude: float
    longitude: float

    def __post_init__(self) -> None:
        """Validate coordinate ranges."""
        if not -90 <= self.latitude <= 90:
            raise ValueError(f"Latitude must be between -90 and 90, got {self.latitude}")
        if not -180 <= self.longitude <= 180:
            raise ValueError(f"Longitude must be between -180 and 180, got {self.longitude}")

    @property
    def location_key(self) -> tuple[float, float]:
        """Return a rounded tuple for approximate location matching.
        
        Returns:
            Tuple of (latitude, longitude) rounded to 6 decimal places.
        """
        return (round(self.latitude, 6), round(self.longitude, 6))

    def distance_to(self, other: "GPSCoordinates") -> float:
        """Calculate distance to another coordinate using Haversine formula.
        
        Args:
            other: Another GPS coordinate.
            
        Returns:
            Distance in meters.
        """
        r = 6371000  # Earth's radius in meters
        lat1 = math.radians(self.latitude)
        lat2 = math.radians(other.latitude)
        delta_lat = math.radians(other.latitude - self.latitude)
        delta_lon = math.radians(other.longitude - self.longitude)

        a = (
            math.sin(delta_lat / 2) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return r * c


@dataclass(frozen=True, slots=True)
class ImageMetrics:
    """Computed quality metrics for an image.
    
    These metrics are used by scoring strategies to rank images.
    
    Attributes:
        width: Image width in pixels.
        height: Image height in pixels.
        sharpness: Sharpness score (Laplacian variance).
        colorfulness: Colorfulness metric.
        file_size: File size in bytes.
        bit_depth: Bits per channel.
    """
    
    width: int
    height: int
    sharpness: float
    colorfulness: float
    file_size: int
    bit_depth: int = 8

    @property
    def resolution(self) -> int:
        """Return total pixel count.
        
        Returns:
            Width multiplied by height.
        """
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        """Return aspect ratio (width / height).
        
        Returns:
            Aspect ratio as a float.
        """
        return self.width / max(1, self.height)

    @property
    def megapixels(self) -> float:
        """Return resolution in megapixels.
        
        Returns:
            Resolution divided by 1,000,000.
        """
        return self.resolution / 1_000_000

    @property
    def quality_score(self) -> float:
        """Compute a default quality score.
        
        This is a simple combined metric. Use ScoringStrategy for
        customized scoring.
        
        Returns:
            Combined quality score.
        """
        # Normalize sharpness (typical range 0-10000)
        norm_sharpness = min(self.sharpness / 1000.0, 10.0)
        # Normalize colorfulness (typical range 0-100)
        norm_colorfulness = min(self.colorfulness / 50.0, 2.0)
        # Normalize resolution (logarithmic)
        norm_resolution = math.log10(max(1, self.resolution)) / 7.0  # ~10MP = 1.0
        
        return (norm_sharpness * 0.5) + (norm_colorfulness * 0.2) + (norm_resolution * 0.3)


@dataclass(frozen=True, slots=True)
class ImageRecord:
    """Complete record for an analyzed image.
    
    This is the primary data structure passed through the pipeline.
    
    Attributes:
        path: Absolute path to the image file.
        sha256: SHA-256 cryptographic hash of file contents.
        phash: Perceptual hash as an integer bitset.
        metrics: Computed quality metrics.
        gps: GPS coordinates if available.
        capture_time: Original capture timestamp if available.
        original_filename: Original filename for reporting.
    """
    
    path: Path
    sha256: str
    phash: int
    metrics: ImageMetrics
    gps: GPSCoordinates | None = None
    capture_time: datetime | None = None
    original_filename: str = ""

    def __post_init__(self) -> None:
        """Set original filename if not provided."""
        if not self.original_filename:
            object.__setattr__(self, "original_filename", self.path.name)

    @property
    def has_gps(self) -> bool:
        """Return True if GPS coordinates are available.
        
        Returns:
            True if gps is not None.
        """
        return self.gps is not None

    @property
    def extension(self) -> str:
        """Return the file extension in lowercase.
        
        Returns:
            File extension including the leading dot.
        """
        return self.path.suffix.lower()

    @property
    def file_size(self) -> int:
        """Return the file size in bytes.
        
        Returns:
            File size from metrics.
        """
        return self.metrics.file_size

    def gps_distance_to(self, other: "ImageRecord") -> float | None:
        """Calculate GPS distance to another image.
        
        Args:
            other: Another ImageRecord.
            
        Returns:
            Distance in meters, or None if either lacks GPS.
        """
        if not self.has_gps or not other.has_gps:
            return None
        return self.gps.distance_to(other.gps)  # type: ignore

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.
        
        Returns:
            Dictionary representation suitable for JSON export.
        """
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "phash": hex(self.phash),
            "width": self.metrics.width,
            "height": self.metrics.height,
            "file_size": self.metrics.file_size,
            "sharpness": round(self.metrics.sharpness, 2),
            "colorfulness": round(self.metrics.colorfulness, 2),
            "quality_score": round(self.metrics.quality_score, 4),
            "gps": {
                "latitude": self.gps.latitude,
                "longitude": self.gps.longitude,
            } if self.gps else None,
            "capture_time": self.capture_time.isoformat() if self.capture_time else None,
        }


@dataclass
class DeduplicationResult:
    """Result of the deduplication pipeline.
    
    Attributes:
        total_scanned: Total number of files scanned.
        total_analyzed: Total number of files successfully analyzed.
        exact_duplicates_removed: Count of exact duplicates removed.
        near_duplicates_removed: Count of near-duplicates removed.
        gps_duplicates_removed: Count of GPS-based duplicates removed.
        selected_count: Number of images selected for export.
        exported_count: Number of images successfully exported.
        exported_paths: List of paths to exported files.
        skipped_files: List of files that were skipped with reasons.
        duration_seconds: Total processing time in seconds.
    """
    
    total_scanned: int = 0
    total_analyzed: int = 0
    exact_duplicates_removed: int = 0
    near_duplicates_removed: int = 0
    gps_duplicates_removed: int = 0
    selected_count: int = 0
    exported_count: int = 0
    exported_paths: list[Path] = field(default_factory=list)
    skipped_files: list[tuple[str, str]] = field(default_factory=list)
    duration_seconds: float = 0.0

    @property
    def total_duplicates_removed(self) -> int:
        """Return total number of duplicates removed.
        
        Returns:
            Sum of all duplicate categories.
        """
        return (
            self.exact_duplicates_removed
            + self.near_duplicates_removed
            + self.gps_duplicates_removed
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.
        
        Returns:
            Dictionary representation suitable for JSON export.
        """
        return {
            "total_scanned": self.total_scanned,
            "total_analyzed": self.total_analyzed,
            "exact_duplicates_removed": self.exact_duplicates_removed,
            "near_duplicates_removed": self.near_duplicates_removed,
            "gps_duplicates_removed": self.gps_duplicates_removed,
            "total_duplicates_removed": self.total_duplicates_removed,
            "selected_count": self.selected_count,
            "exported_count": self.exported_count,
            "exported_paths": [str(p) for p in self.exported_paths],
            "skipped_files": [
                {"path": path, "reason": reason} 
                for path, reason in self.skipped_files
            ],
            "duration_seconds": round(self.duration_seconds, 2),
        }
