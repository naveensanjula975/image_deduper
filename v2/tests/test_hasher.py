"""Tests for the hasher module."""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from image_deduper_v2.hasher import (
    compute_ahash,
    compute_dhash,
    compute_phash,
    hamming_distance,
    hashes_are_similar,
    sha256_file,
    ImageHasher,
)


class TestSha256:
    """Tests for SHA-256 file hashing."""

    def test_sha256_file(self, sample_image: Path) -> None:
        """Test computing SHA-256 hash of a file."""
        digest = sha256_file(sample_image)
        
        assert isinstance(digest, str)
        assert len(digest) == 64  # SHA-256 hex digest length
        assert all(c in "0123456789abcdef" for c in digest)

    def test_sha256_identical_files(self, temp_dir: Path) -> None:
        """Test that identical files produce the same hash."""
        img = Image.new("RGB", (50, 50), color=(100, 100, 100))
        
        path1 = temp_dir / "file1.jpg"
        path2 = temp_dir / "file2.jpg"
        
        img.save(path1, "JPEG")
        img.save(path2, "JPEG")
        
        hash1 = sha256_file(path1)
        hash2 = sha256_file(path2)
        
        assert hash1 == hash2

    def test_sha256_different_files(self, temp_dir: Path) -> None:
        """Test that different files produce different hashes."""
        img1 = Image.new("RGB", (50, 50), color=(100, 100, 100))
        img2 = Image.new("RGB", (50, 50), color=(200, 200, 200))
        
        path1 = temp_dir / "file1.jpg"
        path2 = temp_dir / "file2.jpg"
        
        img1.save(path1, "JPEG")
        img2.save(path2, "JPEG")
        
        hash1 = sha256_file(path1)
        hash2 = sha256_file(path2)
        
        assert hash1 != hash2


class TestPerceptualHash:
    """Tests for perceptual hash functions."""

    def test_compute_ahash(self) -> None:
        """Test computing average hash."""
        img = Image.new("RGB", (100, 100), color=(255, 0, 0))
        ahash = compute_ahash(img, size=8)
        
        assert isinstance(ahash, int)
        assert ahash >= 0

    def test_compute_dhash(self) -> None:
        """Test computing difference hash."""
        img = Image.new("RGB", (100, 100), color=(0, 255, 0))
        dhash = compute_dhash(img, size=8)
        
        assert isinstance(dhash, int)
        assert dhash >= 0

    def test_ahash_similar_images(self) -> None:
        """Test that similar images have similar aHash."""
        # Create two similar images (slightly different shades)
        img1 = Image.new("RGB", (100, 100), color=(255, 0, 0))
        img2 = Image.new("RGB", (100, 100), color=(250, 5, 5))
        
        hash1 = compute_ahash(img1, size=8)
        hash2 = compute_ahash(img2, size=8)
        
        distance = hamming_distance(hash1, hash2)
        assert distance <= 10  # Should be very similar

    def test_ahash_different_images(self) -> None:
        """Test that different images have different aHash."""
        img1 = Image.new("RGB", (100, 100), color=(255, 0, 0))
        img2 = Image.new("RGB", (100, 100), color=(0, 0, 255))
        
        hash1 = compute_ahash(img1, size=8)
        hash2 = compute_ahash(img2, size=8)
        
        distance = hamming_distance(hash1, hash2)
        # Different colors should have notable distance
        assert distance > 0


class TestHammingDistance:
    """Tests for Hamming distance calculation."""

    def test_hamming_identical(self) -> None:
        """Test Hamming distance of identical values."""
        assert hamming_distance(0b1010, 0b1010) == 0
        assert hamming_distance(0, 0) == 0
        assert hamming_distance(0xFF, 0xFF) == 0

    def test_hamming_single_bit(self) -> None:
        """Test Hamming distance with single bit difference."""
        assert hamming_distance(0b0000, 0b0001) == 1
        assert hamming_distance(0b1010, 0b1011) == 1
        assert hamming_distance(0b1000, 0b0000) == 1

    def test_hamming_multiple_bits(self) -> None:
        """Test Hamming distance with multiple bit differences."""
        assert hamming_distance(0b0000, 0b1111) == 4
        assert hamming_distance(0b1010, 0b0101) == 4
        assert hamming_distance(0x00, 0xFF) == 8

    def test_hashes_are_similar(self) -> None:
        """Test similarity comparison function."""
        assert hashes_are_similar(0b1010, 0b1010, threshold=5) is True
        assert hashes_are_similar(0b1010, 0b1011, threshold=5) is True
        assert hashes_are_similar(0b0000, 0b1111, threshold=3) is False


class TestImageHasher:
    """Tests for the ImageHasher class."""

    def test_hasher_initialization(self) -> None:
        """Test ImageHasher initialization."""
        hasher = ImageHasher(hash_size=16)
        
        assert hasher.hash_size == 16
        assert hasher.total_bits == 256

    def test_hasher_compute_file_hash(self, sample_image: Path) -> None:
        """Test computing file hash through ImageHasher."""
        hasher = ImageHasher()
        digest = hasher.compute_file_hash(sample_image)
        
        assert len(digest) == 64

    def test_hasher_compute_perceptual_hash(self) -> None:
        """Test computing perceptual hash through ImageHasher."""
        hasher = ImageHasher(hash_size=8)
        img = Image.new("RGB", (100, 100), color=(255, 0, 0))
        
        phash = hasher.compute_perceptual_hash(img)
        
        assert isinstance(phash, int)

    def test_hasher_are_similar(self) -> None:
        """Test similarity check through ImageHasher."""
        hasher = ImageHasher(hash_size=8)
        
        # Create similar images
        img1 = Image.new("RGB", (100, 100), color=(255, 0, 0))
        img2 = Image.new("RGB", (100, 100), color=(250, 5, 5))
        
        hash1 = hasher.compute_average_hash(img1)
        hash2 = hasher.compute_average_hash(img2)
        
        assert hasher.are_similar(hash1, hash2)
