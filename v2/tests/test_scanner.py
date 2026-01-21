"""Tests for the scanner module."""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from image_deduper_v2.scanner import ImageScanner, iter_image_files


class TestIterImageFiles:
    """Tests for the iter_image_files generator."""

    def test_scan_directory(self, temp_dir: Path) -> None:
        """Test scanning a directory for images."""
        # Create test images
        (temp_dir / "image1.jpg").touch()
        (temp_dir / "image2.png").touch()
        (temp_dir / "document.txt").touch()  # Should be ignored
        
        files = list(iter_image_files(
            [temp_dir],
            allowed_extensions={".jpg", ".png"},
            recursive=False,
        ))
        
        assert len(files) == 2
        assert all(f.suffix.lower() in {".jpg", ".png"} for f in files)

    def test_scan_recursive(self, temp_dir: Path) -> None:
        """Test recursive directory scanning."""
        subdir = temp_dir / "subdir"
        subdir.mkdir()
        
        (temp_dir / "image1.jpg").touch()
        (subdir / "image2.jpg").touch()
        
        files = list(iter_image_files(
            [temp_dir],
            allowed_extensions={".jpg"},
            recursive=True,
        ))
        
        assert len(files) == 2

    def test_scan_non_recursive(self, temp_dir: Path) -> None:
        """Test non-recursive directory scanning."""
        subdir = temp_dir / "subdir"
        subdir.mkdir()
        
        (temp_dir / "image1.jpg").touch()
        (subdir / "image2.jpg").touch()
        
        files = list(iter_image_files(
            [temp_dir],
            allowed_extensions={".jpg"},
            recursive=False,
        ))
        
        assert len(files) == 1

    def test_scan_single_file(self, sample_image: Path) -> None:
        """Test scanning a single file."""
        files = list(iter_image_files(
            [sample_image],
            allowed_extensions={".jpg"},
            recursive=False,
        ))
        
        assert len(files) == 1
        assert files[0] == sample_image

    def test_scan_nonexistent_path(self, temp_dir: Path) -> None:
        """Test scanning a non-existent path."""
        files = list(iter_image_files(
            [temp_dir / "nonexistent"],
            allowed_extensions={".jpg"},
            recursive=False,
        ))
        
        assert len(files) == 0

    def test_scan_size_filter(self, temp_dir: Path) -> None:
        """Test filtering by file size."""
        # Create small image
        small_img = Image.new("RGB", (10, 10))
        small_path = temp_dir / "small.jpg"
        small_img.save(small_path, "JPEG")
        
        # Create larger image
        large_img = Image.new("RGB", (500, 500))
        large_path = temp_dir / "large.jpg"
        large_img.save(large_path, "JPEG")
        
        small_size = small_path.stat().st_size
        large_size = large_path.stat().st_size
        
        # Filter to only get large images
        files = list(iter_image_files(
            [temp_dir],
            allowed_extensions={".jpg"},
            recursive=False,
            min_size_bytes=small_size + 1,
        ))
        
        assert len(files) == 1
        assert files[0] == large_path


class TestImageScanner:
    """Tests for the ImageScanner class."""

    def test_scanner_initialization(self, temp_dir: Path) -> None:
        """Test scanner initialization."""
        scanner = ImageScanner(
            input_paths=[temp_dir],
            allowed_extensions={".jpg", ".png"},
            recursive=True,
        )
        
        assert scanner.input_paths == [temp_dir]
        assert scanner.allowed_extensions == {".jpg", ".png"}

    def test_scanner_scan(self, sample_images: list[Path]) -> None:
        """Test scanner scan method."""
        parent = sample_images[0].parent
        
        scanner = ImageScanner(
            input_paths=[parent],
            allowed_extensions={".jpg"},
            recursive=False,
        )
        
        files = list(scanner.scan())
        
        assert len(files) == len(sample_images)

    def test_scanner_scan_to_list(self, sample_images: list[Path]) -> None:
        """Test scanner scan_to_list method."""
        parent = sample_images[0].parent
        
        scanner = ImageScanner(
            input_paths=[parent],
            allowed_extensions={".jpg"},
            recursive=False,
        )
        
        files = scanner.scan_to_list()
        
        assert isinstance(files, list)
        assert len(files) == len(sample_images)

    def test_scanner_count(self, sample_images: list[Path]) -> None:
        """Test scanner count method."""
        parent = sample_images[0].parent
        
        scanner = ImageScanner(
            input_paths=[parent],
            allowed_extensions={".jpg"},
            recursive=False,
        )
        
        count = scanner.count()
        
        assert count == len(sample_images)
