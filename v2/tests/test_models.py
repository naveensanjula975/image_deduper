"""Tests for the models module."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from image_deduper_v2.models import (
    DeduplicationResult,
    GPSCoordinates,
    ImageMetrics,
    ImageRecord,
)


class TestGPSCoordinates:
    """Tests for GPSCoordinates dataclass."""

    def test_creation(self) -> None:
        """Test creating GPS coordinates."""
        gps = GPSCoordinates(latitude=40.7128, longitude=-74.0060)
        assert gps.latitude == 40.7128

    def test_distance_to_same_location(self) -> None:
        """Test distance to same location is zero."""
        gps1 = GPSCoordinates(latitude=40.7128, longitude=-74.0060)
        gps2 = GPSCoordinates(latitude=40.7128, longitude=-74.0060)
        assert gps1.distance_to(gps2) == pytest.approx(0.0, abs=0.01)

    def test_invalid_latitude(self) -> None:
        """Test that invalid latitude raises error."""
        with pytest.raises(ValueError):
            GPSCoordinates(latitude=91.0, longitude=0.0)


class TestImageMetrics:
    """Tests for ImageMetrics dataclass."""

    def test_resolution(self) -> None:
        """Test resolution property."""
        metrics = ImageMetrics(
            width=1920, height=1080,
            sharpness=100.0, colorfulness=50.0, file_size=1000,
        )
        assert metrics.resolution == 1920 * 1080

    def test_megapixels(self) -> None:
        """Test megapixels property."""
        metrics = ImageMetrics(
            width=4000, height=3000,
            sharpness=100.0, colorfulness=50.0, file_size=1000,
        )
        assert metrics.megapixels == 12.0


class TestImageRecord:
    """Tests for ImageRecord dataclass."""

    def test_has_gps_without_gps(self) -> None:
        """Test has_gps property without GPS data."""
        record = ImageRecord(
            path=Path("/test/image.jpg"),
            sha256="abc123", phash=0,
            metrics=ImageMetrics(100, 100, 100.0, 50.0, 1000),
        )
        assert record.has_gps is False

    def test_has_gps_with_gps(self) -> None:
        """Test has_gps property with GPS data."""
        record = ImageRecord(
            path=Path("/test/image.jpg"),
            sha256="abc123", phash=0,
            metrics=ImageMetrics(100, 100, 100.0, 50.0, 1000),
            gps=GPSCoordinates(40.7128, -74.0060),
        )
        assert record.has_gps is True


class TestDeduplicationResult:
    """Tests for DeduplicationResult dataclass."""

    def test_total_duplicates_removed(self) -> None:
        """Test total_duplicates_removed property."""
        result = DeduplicationResult(
            exact_duplicates_removed=10,
            near_duplicates_removed=5,
            gps_duplicates_removed=3,
        )
        assert result.total_duplicates_removed == 18
