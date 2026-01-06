from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass(frozen=True)
class ImageRecord:
    """Stores analysis results for an image."""

    path: Path
    file_size: int
    width: int
    height: int
    sha256: str
    phash: int
    quality: float
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    capture_time: Optional[str] = None

    @property
    def pixels(self) -> int:
        """Returns total pixels."""
        return self.width * self.height

    @property
    def ext(self) -> str:
        """Returns the file extension."""
        return self.path.suffix.lower()

    @property
    def has_gps(self) -> bool:
        """Returns True if GPS coordinates are available."""
        return self.latitude is not None and self.longitude is not None

    @property
    def gps_key(self) -> Optional[Tuple[float, float]]:
        """Returns a rounded GPS tuple for location-based grouping."""
        if not self.has_gps:
            return None
        return (round(self.latitude, 5), round(self.longitude, 5))

    def gps_distance_to(self, other: ImageRecord) -> Optional[float]:
        """Calculates distance in meters to another image's GPS location.

        Args:
            other: Another ImageRecord.

        Returns:
            Distance in meters or None if either lacks GPS.
        """
        if not self.has_gps or not other.has_gps:
            return None
        
        import math
        
        r = 6371000
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
