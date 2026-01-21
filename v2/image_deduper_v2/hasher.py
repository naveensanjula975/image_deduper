"""Hash computation module.

This module provides functions for computing cryptographic
and perceptual hashes of images.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import BinaryIO

from PIL import Image, ImageOps

from image_deduper_v2.exceptions import HashingError


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 hash of a file.
    
    Reads the file in chunks to handle large files efficiently.
    
    Args:
        path: Path to the file.
        chunk_size: Size of chunks to read in bytes.
        
    Returns:
        Hexadecimal digest of the SHA-256 hash.
        
    Raises:
        HashingError: If the file cannot be read.
        
    Example:
        >>> digest = sha256_file(Path("image.jpg"))
        >>> print(digest)  # e.g., "a1b2c3..."
    """
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError as e:
        raise HashingError(
            f"Failed to compute SHA-256 for {path}",
            details=str(e),
        ) from e


def sha256_stream(stream: BinaryIO, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 hash from a binary stream.
    
    Args:
        stream: Binary file-like object.
        chunk_size: Size of chunks to read.
        
    Returns:
        Hexadecimal digest of the SHA-256 hash.
    """
    h = hashlib.sha256()
    for chunk in iter(lambda: stream.read(chunk_size), b""):
        h.update(chunk)
    return h.hexdigest()


def compute_ahash(img: Image.Image, size: int = 16) -> int:
    """Compute average hash (aHash) for an image.
    
    The average hash compares each pixel to the mean intensity
    and generates a bitstring based on whether it's above or below.
    
    Args:
        img: PIL Image object.
        size: Hash grid size (hash bits = size * size).
        
    Returns:
        Integer representing the hash as a bitset.
        
    Example:
        >>> with Image.open("photo.jpg") as img:
        ...     hash_value = compute_ahash(img)
    """
    # Convert to grayscale and resize
    gray = ImageOps.grayscale(img)
    resized = gray.resize((size, size), Image.Resampling.LANCZOS)
    
    # Get pixel data
    pixels = list(resized.getdata())
    
    # Compute average
    avg = sum(pixels) / len(pixels) if pixels else 0.0
    
    # Build bitstring
    bits = 0
    for i, px in enumerate(pixels):
        if px >= avg:
            bits |= 1 << i
    
    return bits


def compute_dhash(img: Image.Image, size: int = 16) -> int:
    """Compute difference hash (dHash) for an image.
    
    The difference hash compares adjacent pixels, generating
    bits based on whether the left pixel is brighter.
    
    Args:
        img: PIL Image object.
        size: Hash grid size (hash bits = size * size).
        
    Returns:
        Integer representing the hash as a bitset.
    """
    # Convert to grayscale and resize (one extra column for differences)
    gray = ImageOps.grayscale(img)
    resized = gray.resize((size + 1, size), Image.Resampling.LANCZOS)
    
    # Get pixel data
    pixels = list(resized.getdata())
    
    # Build bitstring based on horizontal differences
    bits = 0
    bit_index = 0
    for row in range(size):
        for col in range(size):
            idx = row * (size + 1) + col
            if pixels[idx] > pixels[idx + 1]:
                bits |= 1 << bit_index
            bit_index += 1
    
    return bits


def compute_phash(img: Image.Image, size: int = 16) -> int:
    """Compute perceptual hash (pHash) using DCT.
    
    This simplified implementation uses the imagehash library's
    algorithm if available, otherwise falls back to aHash.
    
    Args:
        img: PIL Image object.
        size: Hash size parameter.
        
    Returns:
        Integer representing the hash as a bitset.
    """
    try:
        import imagehash
        
        # Use imagehash library for proper pHash
        phash = imagehash.phash(img, hash_size=size)
        
        # Convert to integer
        return int(str(phash), 16)
    except ImportError:
        # Fall back to aHash if imagehash not available
        return compute_ahash(img, size)


def hamming_distance(hash1: int, hash2: int) -> int:
    """Compute Hamming distance between two hash values.
    
    The Hamming distance is the number of bit positions
    where the corresponding bits differ.
    
    Args:
        hash1: First hash value.
        hash2: Second hash value.
        
    Returns:
        Number of differing bits.
        
    Example:
        >>> h1 = compute_ahash(img1)
        >>> h2 = compute_ahash(img2)
        >>> distance = hamming_distance(h1, h2)
        >>> if distance < 10:
        ...     print("Images are similar")
    """
    return (hash1 ^ hash2).bit_count()


def hashes_are_similar(
    hash1: int,
    hash2: int,
    threshold: int = 10,
) -> bool:
    """Check if two hashes are similar within a threshold.
    
    Args:
        hash1: First hash value.
        hash2: Second hash value.
        threshold: Maximum Hamming distance to consider similar.
        
    Returns:
        True if the Hamming distance is less than or equal to threshold.
    """
    return hamming_distance(hash1, hash2) <= threshold


class ImageHasher:
    """Class for computing multiple hash types for an image.
    
    Provides a convenient interface for computing and comparing
    different hash types.
    
    Attributes:
        hash_size: Size parameter for hash computation.
    """

    def __init__(self, hash_size: int = 16) -> None:
        """Initialize the hasher.
        
        Args:
            hash_size: Size parameter for hash computation.
        """
        self._hash_size = hash_size

    @property
    def hash_size(self) -> int:
        """Return the hash size."""
        return self._hash_size

    @property
    def total_bits(self) -> int:
        """Return total number of bits in the hash.
        
        Returns:
            hash_size squared.
        """
        return self._hash_size * self._hash_size

    def compute_file_hash(self, path: Path) -> str:
        """Compute SHA-256 hash of a file.
        
        Args:
            path: Path to the file.
            
        Returns:
            Hexadecimal SHA-256 digest.
        """
        return sha256_file(path)

    def compute_perceptual_hash(self, img: Image.Image) -> int:
        """Compute perceptual hash of an image.
        
        Args:
            img: PIL Image object.
            
        Returns:
            Integer hash value.
        """
        return compute_phash(img, self._hash_size)

    def compute_average_hash(self, img: Image.Image) -> int:
        """Compute average hash of an image.
        
        Args:
            img: PIL Image object.
            
        Returns:
            Integer hash value.
        """
        return compute_ahash(img, self._hash_size)

    def are_similar(self, hash1: int, hash2: int, threshold: int | None = None) -> bool:
        """Check if two hashes are similar.
        
        Args:
            hash1: First hash value.
            hash2: Second hash value.
            threshold: Optional custom threshold (default based on hash size).
            
        Returns:
            True if hashes are similar.
        """
        if threshold is None:
            # Default threshold is about 10% of total bits
            threshold = max(1, self.total_bits // 10)
        return hashes_are_similar(hash1, hash2, threshold)
