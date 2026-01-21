"""GPS and geospatial utility functions.

This module provides functions for GPS coordinate calculations
used in location-based image deduplication.
"""
from __future__ import annotations

import math
from typing import Sequence


def haversine_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """Calculate distance between two GPS coordinates using the Haversine formula.
    
    Args:
        lat1: Latitude of first point in decimal degrees.
        lon1: Longitude of first point in decimal degrees.
        lat2: Latitude of second point in decimal degrees.
        lon2: Longitude of second point in decimal degrees.
        
    Returns:
        Distance in meters between the two points.
        
    Example:
        >>> dist = haversine_distance(40.7128, -74.0060, 34.0522, -118.2437)
        >>> print(f"{dist / 1000:.0f} km")  # ~3944 km from NYC to LA
    """
    # Earth's radius in meters
    earth_radius = 6_371_000

    # Convert to radians
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    # Haversine formula
    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return earth_radius * c


def bearing_between(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """Calculate initial bearing between two GPS coordinates.
    
    Args:
        lat1: Latitude of first point in decimal degrees.
        lon1: Longitude of first point in decimal degrees.
        lat2: Latitude of second point in decimal degrees.
        lon2: Longitude of second point in decimal degrees.
        
    Returns:
        Initial bearing in degrees (0-360).
    """
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lon = math.radians(lon2 - lon1)

    x = math.sin(delta_lon) * math.cos(lat2_rad)
    y = (
        math.cos(lat1_rad) * math.sin(lat2_rad)
        - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon)
    )

    bearing = math.degrees(math.atan2(x, y))
    return (bearing + 360) % 360


def compute_centroid(
    coordinates: Sequence[tuple[float, float]],
) -> tuple[float, float] | None:
    """Compute the geographic centroid of a set of coordinates.
    
    Uses a simple averaging method which works well for areas
    that don't span too much of the globe.
    
    Args:
        coordinates: Sequence of (latitude, longitude) tuples.
        
    Returns:
        Tuple of (latitude, longitude) for the centroid,
        or None if the sequence is empty.
        
    Example:
        >>> coords = [(40.7128, -74.0060), (34.0522, -118.2437)]
        >>> centroid = compute_centroid(coords)
    """
    if not coordinates:
        return None

    n = len(coordinates)
    total_lat = sum(lat for lat, _ in coordinates)
    total_lon = sum(lon for _, lon in coordinates)

    return (total_lat / n, total_lon / n)


def bounding_box(
    coordinates: Sequence[tuple[float, float]],
) -> tuple[float, float, float, float] | None:
    """Compute the bounding box for a set of coordinates.
    
    Args:
        coordinates: Sequence of (latitude, longitude) tuples.
        
    Returns:
        Tuple of (min_lat, min_lon, max_lat, max_lon),
        or None if the sequence is empty.
    """
    if not coordinates:
        return None

    lats = [lat for lat, _ in coordinates]
    lons = [lon for _, lon in coordinates]

    return (min(lats), min(lons), max(lats), max(lons))


def meters_per_degree_latitude(latitude: float) -> float:
    """Calculate meters per degree of latitude at a given latitude.
    
    Latitude degrees are nearly constant regardless of location.
    
    Args:
        latitude: Latitude in decimal degrees.
        
    Returns:
        Approximate meters per degree of latitude.
    """
    # This is approximately constant at ~111,132 meters
    return 111_132.92


def meters_per_degree_longitude(latitude: float) -> float:
    """Calculate meters per degree of longitude at a given latitude.
    
    Longitude degrees shrink toward the poles.
    
    Args:
        latitude: Latitude in decimal degrees.
        
    Returns:
        Approximate meters per degree of longitude at the given latitude.
    """
    return 111_132.92 * math.cos(math.radians(latitude))
