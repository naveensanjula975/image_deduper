"""Tests for the configuration module."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from image_deduper_v2.config import Settings, ScoringWeights


class TestScoringWeights:
    """Tests for ScoringWeights model."""

    def test_default_values(self) -> None:
        """Test default weight values."""
        weights = ScoringWeights()
        
        assert weights.sharpness == 1.0
        assert weights.colorfulness == 0.3
        assert weights.resolution == 0.5
        assert weights.gps_presence == 0.2

    def test_custom_values(self) -> None:
        """Test custom weight values."""
        weights = ScoringWeights(
            sharpness=2.0,
            colorfulness=0.5,
            resolution=1.0,
            gps_presence=0.1,
        )
        
        assert weights.sharpness == 2.0
        assert weights.colorfulness == 0.5
        assert weights.resolution == 1.0
        assert weights.gps_presence == 0.1

    def test_validation_min_value(self) -> None:
        """Test that negative values are rejected."""
        with pytest.raises(ValidationError):
            ScoringWeights(sharpness=-1.0)

    def test_validation_max_value(self) -> None:
        """Test that values above 10 are rejected."""
        with pytest.raises(ValidationError):
            ScoringWeights(sharpness=15.0)


class TestSettings:
    """Tests for Settings model."""

    def test_minimal_settings(self, temp_dir: Path) -> None:
        """Test creating settings with minimal required fields."""
        input_path = temp_dir / "input"
        input_path.mkdir()
        output_path = temp_dir / "output"
        
        settings = Settings(
            input_paths=[input_path],
            output_dir=output_path,
        )
        
        assert settings.input_paths == [input_path]
        assert settings.output_dir == output_path
        assert settings.target_count == 200  # Default

    def test_all_settings(self, temp_dir: Path) -> None:
        """Test creating settings with all fields specified."""
        input_path = temp_dir / "input"
        input_path.mkdir()
        
        settings = Settings(
            input_paths=[input_path],
            output_dir=temp_dir / "output",
            target_count=500,
            phash_size=8,
            hamming_threshold=5,
            recursive=False,
            workers=4,
            enable_gps_filter=True,
            gps_distance_threshold=100.0,
        )
        
        assert settings.target_count == 500
        assert settings.phash_size == 8
        assert settings.hamming_threshold == 5
        assert settings.recursive is False
        assert settings.workers == 4
        assert settings.enable_gps_filter is True
        assert settings.gps_distance_threshold == 100.0

    def test_path_coercion_string(self, temp_dir: Path) -> None:
        """Test that string paths are coerced to Path objects."""
        input_path = temp_dir / "input"
        input_path.mkdir()
        
        settings = Settings(
            input_paths=[str(input_path)],
            output_dir=str(temp_dir / "output"),
        )
        
        assert isinstance(settings.input_paths[0], Path)
        assert isinstance(settings.output_dir, Path)

    def test_path_coercion_single(self, temp_dir: Path) -> None:
        """Test that single path is coerced to list."""
        input_path = temp_dir / "input"
        input_path.mkdir()
        
        settings = Settings(
            input_paths=input_path,  # type: ignore
            output_dir=temp_dir / "output",
        )
        
        assert len(settings.input_paths) == 1
        assert settings.input_paths[0] == input_path

    def test_extension_normalization(self, temp_dir: Path) -> None:
        """Test that extensions are normalized to lowercase with dot."""
        input_path = temp_dir / "input"
        input_path.mkdir()
        
        settings = Settings(
            input_paths=[input_path],
            output_dir=temp_dir / "output",
            allowed_extensions=["JPG", "png", ".WEBP"],
        )
        
        assert ".jpg" in settings.allowed_extensions_set
        assert ".png" in settings.allowed_extensions_set
        assert ".webp" in settings.allowed_extensions_set

    def test_input_path_validation(self, temp_dir: Path) -> None:
        """Test that non-existent input paths are rejected."""
        with pytest.raises(ValidationError):
            Settings(
                input_paths=[temp_dir / "nonexistent"],
                output_dir=temp_dir / "output",
            )

    def test_ensure_output_dir(self, temp_dir: Path) -> None:
        """Test that output directory is created."""
        input_path = temp_dir / "input"
        input_path.mkdir()
        output_path = temp_dir / "output"
        
        settings = Settings(
            input_paths=[input_path],
            output_dir=output_path,
        )
        
        assert not output_path.exists()
        settings.ensure_output_dir()
        assert output_path.exists()

    def test_max_file_size_bytes(self, temp_dir: Path) -> None:
        """Test max_file_size_bytes property."""
        input_path = temp_dir / "input"
        input_path.mkdir()
        
        settings = Settings(
            input_paths=[input_path],
            output_dir=temp_dir / "output",
            max_file_size_mb=10.0,
        )
        
        assert settings.max_file_size_bytes == 10 * 1024 * 1024

    def test_from_json(self, temp_dir: Path) -> None:
        """Test loading settings from JSON file."""
        input_path = temp_dir / "input"
        input_path.mkdir()
        
        config = {
            "input_paths": [str(input_path)],
            "output_dir": str(temp_dir / "output"),
            "target_count": 300,
            "phash_size": 12,
        }
        
        config_path = temp_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f)
        
        settings = Settings.from_json(config_path)
        
        assert settings.target_count == 300
        assert settings.phash_size == 12

    def test_to_dict(self, temp_dir: Path) -> None:
        """Test converting settings to dictionary."""
        input_path = temp_dir / "input"
        input_path.mkdir()
        
        settings = Settings(
            input_paths=[input_path],
            output_dir=temp_dir / "output",
            target_count=250,
        )
        
        data = settings.to_dict()
        
        assert isinstance(data, dict)
        assert data["target_count"] == 250
        assert isinstance(data["input_paths"], list)
