"""Shared test fixtures and utilities."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Generator

import pytest
from PIL import Image


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests.
    
    Yields:
        Path to the temporary directory.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_image(temp_dir: Path) -> Path:
    """Create a sample test image.
    
    Args:
        temp_dir: Temporary directory fixture.
        
    Returns:
        Path to the created image.
    """
    img_path = temp_dir / "sample.jpg"
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    img.save(img_path, "JPEG")
    return img_path


@pytest.fixture
def sample_images(temp_dir: Path) -> list[Path]:
    """Create multiple sample test images.
    
    Args:
        temp_dir: Temporary directory fixture.
        
    Returns:
        List of paths to created images.
    """
    paths = []
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
    ]
    
    for i, color in enumerate(colors):
        img_path = temp_dir / f"image_{i}.jpg"
        img = Image.new("RGB", (200, 200), color=color)
        img.save(img_path, "JPEG")
        paths.append(img_path)
    
    return paths


@pytest.fixture
def duplicate_images(temp_dir: Path) -> list[Path]:
    """Create duplicate test images (identical content).
    
    Args:
        temp_dir: Temporary directory fixture.
        
    Returns:
        List of paths to duplicate images.
    """
    paths = []
    
    # Create original
    img = Image.new("RGB", (150, 150), color=(128, 128, 128))
    
    # Save multiple copies
    for i in range(3):
        img_path = temp_dir / f"duplicate_{i}.jpg"
        img.save(img_path, "JPEG")
        paths.append(img_path)
    
    return paths


@pytest.fixture
def varied_quality_images(temp_dir: Path) -> list[Path]:
    """Create images with varying quality characteristics.
    
    Args:
        temp_dir: Temporary directory fixture.
        
    Returns:
        List of paths to images with different qualities.
    """
    import random
    
    paths = []
    sizes = [(100, 100), (200, 200), (400, 400), (800, 800)]
    
    for i, size in enumerate(sizes):
        img_path = temp_dir / f"quality_{i}.jpg"
        
        # Create image with some texture
        img = Image.new("RGB", size)
        pixels = img.load()
        
        for x in range(size[0]):
            for y in range(size[1]):
                r = (x * 7 + y * 3) % 256
                g = (x * 5 + y * 11) % 256
                b = (x * 13 + y * 2) % 256
                pixels[x, y] = (r, g, b)
        
        img.save(img_path, "JPEG", quality=90)
        paths.append(img_path)
    
    return paths


@pytest.fixture
def output_dir(temp_dir: Path) -> Path:
    """Create an output directory for tests.
    
    Args:
        temp_dir: Temporary directory fixture.
        
    Returns:
        Path to the output directory.
    """
    output = temp_dir / "output"
    output.mkdir()
    return output


def create_test_image(
    path: Path,
    size: tuple[int, int] = (100, 100),
    color: tuple[int, int, int] = (255, 0, 0),
    format: str = "JPEG",
) -> Path:
    """Create a test image with specified properties.
    
    Args:
        path: Path to save the image.
        size: Image dimensions (width, height).
        color: RGB color tuple.
        format: Image format (JPEG, PNG, etc.).
        
    Returns:
        Path to the created image.
    """
    img = Image.new("RGB", size, color=color)
    img.save(path, format)
    return path
