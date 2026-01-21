"""Tests for the deduplicator module."""
from __future__ import annotations

from pathlib import Path

import pytest

from image_deduper_v2.deduplicator import (
    CompositeDeduplicator,
    ExactHashDeduplicator,
    GPSProximityDeduplicator,
    PerceptualHashDeduplicator,
    UnionFind,
    hamming_distance,
)
from image_deduper_v2.models import GPSCoordinates, ImageMetrics, ImageRecord


def create_mock_record(
    path: str = "/mock/image.jpg",
    sha256: str = "abc123",
    phash: int = 0,
    width: int = 100,
    height: int = 100,
    sharpness: float = 100.0,
    colorfulness: float = 50.0,
    gps: GPSCoordinates | None = None,
) -> ImageRecord:
    """Create a mock ImageRecord for testing."""
    return ImageRecord(
        path=Path(path),
        sha256=sha256,
        phash=phash,
        metrics=ImageMetrics(
            width=width,
            height=height,
            sharpness=sharpness,
            colorfulness=colorfulness,
            file_size=10000,
        ),
        gps=gps,
    )


class TestUnionFind:
    """Tests for UnionFind data structure."""

    def test_create(self) -> None:
        """Test creating UnionFind structure."""
        uf = UnionFind.create(5)
        
        assert len(uf.parent) == 5
        assert len(uf.rank) == 5
        assert all(uf.parent[i] == i for i in range(5))

    def test_find_self(self) -> None:
        """Test finding element in its own set."""
        uf = UnionFind.create(5)
        
        for i in range(5):
            assert uf.find(i) == i

    def test_union_and_find(self) -> None:
        """Test unioning sets and finding representatives."""
        uf = UnionFind.create(5)
        
        uf.union(0, 1)
        uf.union(2, 3)
        
        assert uf.find(0) == uf.find(1)
        assert uf.find(2) == uf.find(3)
        assert uf.find(0) != uf.find(2)

    def test_transitive_union(self) -> None:
        """Test that unions are transitive."""
        uf = UnionFind.create(5)
        
        uf.union(0, 1)
        uf.union(1, 2)
        
        assert uf.find(0) == uf.find(2)

    def test_get_clusters(self) -> None:
        """Test getting all clusters."""
        uf = UnionFind.create(6)
        
        uf.union(0, 1)
        uf.union(2, 3)
        uf.union(2, 4)
        # 5 is alone
        
        clusters = uf.get_clusters()
        
        assert len(clusters) == 3
        cluster_sizes = sorted(len(c) for c in clusters)
        assert cluster_sizes == [1, 2, 3]


class TestExactHashDeduplicator:
    """Tests for exact hash deduplication."""

    def test_no_duplicates(self) -> None:
        """Test with no duplicates."""
        records = [
            create_mock_record(path="/img1.jpg", sha256="hash1"),
            create_mock_record(path="/img2.jpg", sha256="hash2"),
            create_mock_record(path="/img3.jpg", sha256="hash3"),
        ]
        
        deduper = ExactHashDeduplicator()
        clusters = deduper.find_duplicates(records)
        
        assert len(clusters) == 3
        assert all(len(c) == 1 for c in clusters)

    def test_with_duplicates(self) -> None:
        """Test with duplicate files."""
        records = [
            create_mock_record(path="/img1.jpg", sha256="hash1"),
            create_mock_record(path="/img2.jpg", sha256="hash1"),  # Duplicate
            create_mock_record(path="/img3.jpg", sha256="hash2"),
        ]
        
        deduper = ExactHashDeduplicator()
        clusters = deduper.find_duplicates(records)
        
        assert len(clusters) == 2
        cluster_sizes = sorted(len(c) for c in clusters)
        assert cluster_sizes == [1, 2]

    def test_empty_input(self) -> None:
        """Test with empty input."""
        deduper = ExactHashDeduplicator()
        clusters = deduper.find_duplicates([])
        
        assert clusters == []


class TestPerceptualHashDeduplicator:
    """Tests for perceptual hash deduplication."""

    def test_similar_hashes(self) -> None:
        """Test grouping of similar hashes."""
        records = [
            create_mock_record(path="/img1.jpg", sha256="h1", phash=0b11110000),
            create_mock_record(path="/img2.jpg", sha256="h2", phash=0b11110001),  # 1 bit diff
            create_mock_record(path="/img3.jpg", sha256="h3", phash=0b00001111),  # Very different
        ]
        
        deduper = PerceptualHashDeduplicator(threshold=3, total_bits=8)
        clusters = deduper.find_duplicates(records)
        
        # First two should be clustered, third alone
        assert len(clusters) == 2

    def test_all_unique(self) -> None:
        """Test with all unique hashes."""
        records = [
            create_mock_record(path="/img1.jpg", sha256="h1", phash=0xFF00),
            create_mock_record(path="/img2.jpg", sha256="h2", phash=0x00FF),
            create_mock_record(path="/img3.jpg", sha256="h3", phash=0x0FF0),
        ]
        
        deduper = PerceptualHashDeduplicator(threshold=2, total_bits=16)
        clusters = deduper.find_duplicates(records)
        
        assert len(clusters) == 3

    def test_threshold_boundary(self) -> None:
        """Test threshold boundary condition."""
        # Hashes with exactly threshold difference
        records = [
            create_mock_record(path="/img1.jpg", sha256="h1", phash=0b00000000),
            create_mock_record(path="/img2.jpg", sha256="h2", phash=0b00000111),  # 3 bits diff
        ]
        
        deduper_include = PerceptualHashDeduplicator(threshold=3, total_bits=8)
        clusters_include = deduper_include.find_duplicates(records)
        assert len(clusters_include) == 1  # Should be grouped
        
        deduper_exclude = PerceptualHashDeduplicator(threshold=2, total_bits=8)
        clusters_exclude = deduper_exclude.find_duplicates(records)
        assert len(clusters_exclude) == 2  # Should be separate


class TestGPSProximityDeduplicator:
    """Tests for GPS-based deduplication."""

    def test_nearby_locations(self) -> None:
        """Test grouping of nearby GPS locations."""
        records = [
            create_mock_record(
                path="/img1.jpg", 
                sha256="h1",
                gps=GPSCoordinates(40.7128, -74.0060),  # NYC
            ),
            create_mock_record(
                path="/img2.jpg",
                sha256="h2", 
                gps=GPSCoordinates(40.7129, -74.0061),  # Very close to NYC
            ),
            create_mock_record(
                path="/img3.jpg",
                sha256="h3",
                gps=GPSCoordinates(34.0522, -118.2437),  # LA (far away)
            ),
        ]
        
        deduper = GPSProximityDeduplicator(distance_threshold=100)  # 100 meters
        clusters = deduper.find_duplicates(records)
        
        assert len(clusters) == 2  # NYC pair + LA

    def test_no_gps_data(self) -> None:
        """Test with images lacking GPS data."""
        records = [
            create_mock_record(path="/img1.jpg", sha256="h1"),
            create_mock_record(path="/img2.jpg", sha256="h2"),
        ]
        
        deduper = GPSProximityDeduplicator(distance_threshold=100)
        clusters = deduper.find_duplicates(records)
        
        # Each in its own cluster (no grouping without GPS)
        assert len(clusters) == 2

    def test_mixed_gps(self) -> None:
        """Test with some images having GPS, some not."""
        records = [
            create_mock_record(
                path="/img1.jpg",
                sha256="h1",
                gps=GPSCoordinates(40.7128, -74.0060),
            ),
            create_mock_record(path="/img2.jpg", sha256="h2"),  # No GPS
            create_mock_record(
                path="/img3.jpg",
                sha256="h3",
                gps=GPSCoordinates(40.7129, -74.0061),  # Close to img1
            ),
        ]
        
        deduper = GPSProximityDeduplicator(distance_threshold=100)
        clusters = deduper.find_duplicates(records)
        
        # img1 and img3 grouped, img2 alone
        cluster_sizes = sorted(len(c) for c in clusters)
        assert cluster_sizes == [1, 2]


class TestCompositeDeduplicator:
    """Tests for composite deduplication."""

    def test_multiple_strategies(self) -> None:
        """Test applying multiple deduplication strategies."""
        records = [
            create_mock_record(path="/img1.jpg", sha256="h1", phash=0, sharpness=100),
            create_mock_record(path="/img2.jpg", sha256="h1", phash=0, sharpness=200),  # Exact dup
            create_mock_record(path="/img3.jpg", sha256="h3", phash=1, sharpness=150),  # Near dup to img4
            create_mock_record(path="/img4.jpg", sha256="h4", phash=1, sharpness=50),
        ]
        
        deduper = CompositeDeduplicator(
            strategies=[
                ExactHashDeduplicator(),
                PerceptualHashDeduplicator(threshold=5, total_bits=8),
            ],
            prefer_gps=False,
        )
        
        result, counts = deduper.deduplicate(records)
        
        # Should keep best from each group
        assert len(result) == 2
        assert counts["ExactHash"] == 1
        assert counts["PerceptualHash"] == 1
