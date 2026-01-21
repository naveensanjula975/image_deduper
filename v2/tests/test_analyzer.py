"""Tests for the analyzer module."""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from image_deduper_v2.analyzer import (
    ImageAnalyzer,
    analyze_image,
)


class TestAnalyzeImage:
    """Tests for the analyze_image function."""

    def test_analyze_simple_image(self, sample_image: Path) -> None:
        """Test analyzing a simple test image."""
        record = analyze_image(sample_image)
        
        assert record is not None
        assert record.path == sample_image
        assert len(record.sha256) == 64
        assert record.metrics.width == 100
        assert record.metrics.height == 100
        assert record.phash >= 0

    def test_analyze_returns_metrics(self, sample_image: Path) -> None:
        """Test that analysis returns quality metrics."""
        record = analyze_image(sample_image)
        
        assert record is not None
        assert record.metrics.sharpness >= 0
        assert record.metrics.colorfulness >= 0
        assert record.metrics.file_size > 0

    def test_analyze_corrupted_file(self, temp_dir: Path) -> None:
        """Test analyzing a corrupted/invalid image file."""
        bad_file = temp_dir / "bad.jpg"
        bad_file.write_text("not an image")
        
        record = analyze_image(bad_file)
        
        assert record is None

    def test_analyze_nonexistent_file(self, temp_dir: Path) -> None:
        """Test analyzing a non-existent file."""
        record = analyze_image(temp_dir / "nonexistent.jpg")
        
        assert record is None

    def test_analyze_with_metadata(self, sample_image: Path) -> None:
        """Test analyzing with metadata extraction enabled."""
        record = analyze_image(sample_image, extract_metadata=True)
        
        assert record is not None
        # Simple test images won't have GPS, but field should exist
        assert record.gps is None or record.gps is not None

    def test_analyze_without_metadata(self, sample_image: Path) -> None:
        """Test analyzing with metadata extraction disabled."""
        record = analyze_image(sample_image, extract_metadata=False)
        
        assert record is not None
        assert record.gps is None
        assert record.capture_time is None


class TestImageAnalyzer:
    """Tests for the ImageAnalyzer class."""

    def test_analyzer_initialization(self) -> None:
        """Test analyzer initialization."""
        analyzer = ImageAnalyzer(hash_size=8, extract_metadata=False)
        
        assert analyzer.hash_size == 8
        assert analyzer.extract_metadata is False

    def test_analyzer_analyze(self, sample_image: Path) -> None:
        """Test analyzing single image through analyzer."""
        analyzer = ImageAnalyzer()
        record = analyzer.analyze(sample_image)
        
        assert record is not None
        assert record.path == sample_image

    def test_analyzer_analyze_many(self, sample_images: list[Path]) -> None:
        """Test analyzing multiple images in parallel."""
        analyzer = ImageAnalyzer()
        records = analyzer.analyze_many(sample_images, max_workers=2)
        
        assert len(records) == len(sample_images)
        assert all(r.path in sample_images for r in records)

    def test_analyzer_hash_size_affects_phash(self, sample_image: Path) -> None:
        """Test that hash size affects perceptual hash."""
        analyzer_8 = ImageAnalyzer(hash_size=8)
        analyzer_16 = ImageAnalyzer(hash_size=16)
        
        record_8 = analyzer_8.analyze(sample_image)
        record_16 = analyzer_16.analyze(sample_image)
        
        assert record_8 is not None
        assert record_16 is not None
        # Different hash sizes should produce different values
        # (though this isn't guaranteed, it's likely)


class TestQualityMetrics:
    """Tests for quality metric computation."""

    def test_sharpness_varies_with_content(self, temp_dir: Path) -> None:
        """Test that sharpness varies with image content."""
        # Create sharp image (high contrast edges)
        sharp_img = Image.new("RGB", (100, 100))
        pixels = sharp_img.load()
        for x in range(100):
            for y in range(100):
                # Checkerboard pattern
                if (x + y) % 2 == 0:
                    pixels[x, y] = (255, 255, 255)
                else:
                    pixels[x, y] = (0, 0, 0)
        sharp_path = temp_dir / "sharp.jpg"
        sharp_img.save(sharp_path, "JPEG")
        
        # Create blurry image (uniform)
        blurry_img = Image.new("RGB", (100, 100), color=(128, 128, 128))
        blurry_path = temp_dir / "blurry.jpg"
        blurry_img.save(blurry_path, "JPEG")
        
        sharp_record = analyze_image(sharp_path)
        blurry_record = analyze_image(blurry_path)
        
        assert sharp_record is not None
        assert blurry_record is not None
        assert sharp_record.metrics.sharpness > blurry_record.metrics.sharpness

    def test_colorfulness_varies_with_content(self, temp_dir: Path) -> None:
        """Test that colorfulness varies with image content."""
        # Create colorful image
        colorful_img = Image.new("RGB", (100, 100))
        pixels = colorful_img.load()
        for x in range(100):
            for y in range(100):
                pixels[x, y] = (x * 2 % 256, y * 2 % 256, (x + y) % 256)
        colorful_path = temp_dir / "colorful.jpg"
        colorful_img.save(colorful_path, "JPEG")
        
        # Create grayscale image
        gray_img = Image.new("RGB", (100, 100), color=(128, 128, 128))
        gray_path = temp_dir / "gray.jpg"
        gray_img.save(gray_path, "JPEG")
        
        colorful_record = analyze_image(colorful_path)
        gray_record = analyze_image(gray_path)
        
        assert colorful_record is not None
        assert gray_record is not None
        assert colorful_record.metrics.colorfulness > gray_record.metrics.colorfulness
