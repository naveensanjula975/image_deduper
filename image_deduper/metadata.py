from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image
from PIL.ExifTags import GPSTAGS, TAGS


@dataclass(frozen=True)
class GPSCoordinates:
    """Represents GPS coordinates extracted from image EXIF data."""

    latitude: float
    longitude: float

    @property
    def location_key(self) -> Tuple[float, float]:
        """Returns a tuple key for location comparison."""
        return (round(self.latitude, 6), round(self.longitude, 6))

    def distance_to(self, other: GPSCoordinates) -> float:
        """Calculates approximate distance in meters using Haversine formula.

        Args:
            other: Another GPS coordinate.

        Returns:
            Distance in meters.
        """
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


def _convert_to_degrees(value: Tuple) -> Optional[float]:
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


def _get_gps_data(exif_data: dict) -> Optional[dict]:
    """Extracts GPS info dictionary from EXIF data.

    Args:
        exif_data: Raw EXIF data dictionary.

    Returns:
        GPS info dictionary or None.
    """
    for key, value in exif_data.items():
        tag_name = TAGS.get(key, key)
        if tag_name == "GPSInfo":
            gps_info = {}
            for gps_key, gps_value in value.items():
                gps_tag_name = GPSTAGS.get(gps_key, gps_key)
                gps_info[gps_tag_name] = gps_value
            return gps_info
    return None


def extract_gps_coordinates(
    path: Path,
    logger: logging.Logger,
) -> Optional[GPSCoordinates]:
    """Extracts GPS coordinates from image EXIF metadata.

    Args:
        path: Path to the image file.
        logger: Logger instance.

    Returns:
        GPSCoordinates if found, otherwise None.
    """
    try:
        with Image.open(path) as img:
            exif_data = img._getexif()
            if not exif_data:
                return None

            gps_info = _get_gps_data(exif_data)
            if not gps_info:
                return None

            lat = gps_info.get("GPSLatitude")
            lat_ref = gps_info.get("GPSLatitudeRef")
            lon = gps_info.get("GPSLongitude")
            lon_ref = gps_info.get("GPSLongitudeRef")

            if not all([lat, lat_ref, lon, lon_ref]):
                return None

            latitude = _convert_to_degrees(lat)
            longitude = _convert_to_degrees(lon)

            if latitude is None or longitude is None:
                return None

            if lat_ref == "S":
                latitude = -latitude
            if lon_ref == "W":
                longitude = -longitude

            return GPSCoordinates(latitude=latitude, longitude=longitude)

    except (AttributeError, KeyError, TypeError, OSError) as e:
        logger.debug(f"Failed to extract GPS from {path}: {e}")
        return None


def extract_capture_time(
    path: Path,
    logger: logging.Logger,
) -> Optional[str]:
    """Extracts capture timestamp from image EXIF metadata.

    Args:
        path: Path to the image file.
        logger: Logger instance.

    Returns:
        DateTime string if found, otherwise None.
    """
    try:
        with Image.open(path) as img:
            exif_data = img._getexif()
            if not exif_data:
                return None

            for key, value in exif_data.items():
                tag_name = TAGS.get(key, key)
                if tag_name in ("DateTimeOriginal", "DateTime", "DateTimeDigitized"):
                    return str(value)
            return None

    except (AttributeError, KeyError, TypeError, OSError) as e:
        logger.debug(f"Failed to extract capture time from {path}: {e}")
        return None
