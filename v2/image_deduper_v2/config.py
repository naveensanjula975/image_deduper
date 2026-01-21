"""Configuration settings using Pydantic v2.

This module defines the application configuration using Pydantic v2's
BaseSettings for automatic environment variable loading and validation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ScoringWeights(BaseModel):
    """Weights for quality score computation.
    
    These weights determine how different metrics contribute to the
    final quality score. All weights should be non-negative.
    
    Attributes:
        sharpness: Weight for sharpness metric (Laplacian variance).
        colorfulness: Weight for colorfulness metric.
        resolution: Weight for resolution (pixel count).
        gps_presence: Bonus weight for images with GPS data.
    """
    
    sharpness: float = Field(default=1.0, ge=0.0, le=10.0)
    colorfulness: float = Field(default=0.3, ge=0.0, le=10.0)
    resolution: float = Field(default=0.5, ge=0.0, le=10.0)
    gps_presence: float = Field(default=0.2, ge=0.0, le=10.0)


class Settings(BaseSettings):
    """Application settings for image deduplication.
    
    Settings can be loaded from:
    - Constructor arguments
    - Environment variables (prefixed with IMG_DEDUP_)
    - Configuration files (JSON/YAML)
    
    Attributes:
        input_paths: Input file/folder paths to scan for images.
        output_dir: Directory to write selected images.
        target_count: Number of unique images to output.
        phash_size: Hash grid size (phash bits = phash_size * phash_size).
        hamming_threshold: Maximum Hamming distance for near-duplicates.
        allowed_extensions: Allowed image file extensions.
        recursive: Whether to scan folders recursively.
        workers: Number of worker threads for parallel processing.
        copy_mode: If True, copy files; if False, hardlink when possible.
        overwrite_output: Whether to overwrite existing output directory.
        enable_gps_filter: Whether to enable GPS-based deduplication.
        gps_distance_threshold: Distance in meters for GPS clustering.
        prefer_gps_images: Prefer images with GPS data in selection.
        enable_metadata_extraction: Whether to extract EXIF metadata.
        scoring_weights: Weights for quality score computation.
        min_resolution: Minimum image resolution (width * height).
        max_file_size_mb: Maximum file size in megabytes (0 = no limit).
        report_path: Optional path to write JSON report.
    """

    model_config = SettingsConfigDict(
        env_prefix="IMG_DEDUP_",
        case_sensitive=False,
        env_nested_delimiter="__",
        extra="ignore",
    )

    # Required settings
    input_paths: list[Path] = Field(..., min_length=1)
    output_dir: Path

    # Deduplication settings
    target_count: int = Field(default=200, ge=1, le=100000)
    phash_size: int = Field(default=16, ge=4, le=32)
    hamming_threshold: int = Field(default=10, ge=0, le=128)

    # File filtering
    allowed_extensions: list[str] = Field(
        default_factory=lambda: [
            ".jpg", ".jpeg", ".png", ".webp", ".bmp", 
            ".tif", ".tiff", ".gif", ".heic", ".heif"
        ]
    )
    recursive: bool = Field(default=True)
    min_resolution: int = Field(default=0, ge=0)
    max_file_size_mb: float = Field(default=0.0, ge=0.0)

    # Processing settings
    workers: int = Field(default=8, ge=1, le=128)
    copy_mode: bool = Field(default=True)
    overwrite_output: bool = Field(default=False)

    # GPS settings
    enable_gps_filter: bool = Field(default=False)
    gps_distance_threshold: float = Field(default=50.0, ge=0.0, le=50000.0)
    prefer_gps_images: bool = Field(default=True)

    # Metadata settings
    enable_metadata_extraction: bool = Field(default=True)

    # Scoring settings
    scoring_weights: ScoringWeights = Field(default_factory=ScoringWeights)

    # Report settings
    report_path: Path | None = Field(default=None)

    @field_validator("input_paths", mode="before")
    @classmethod
    def coerce_input_paths(cls, v: Any) -> list[Path]:
        """Coerce input paths to list of Path objects.
        
        Args:
            v: Raw input value (string, Path, or list).
            
        Returns:
            List of Path objects.
            
        Raises:
            ValueError: If input cannot be coerced to paths.
        """
        if isinstance(v, (str, Path)):
            return [Path(v)]
        if isinstance(v, list):
            return [Path(x) for x in v]
        raise ValueError("input_paths must be a path or list of paths")

    @field_validator("output_dir", mode="before")
    @classmethod
    def coerce_output_dir(cls, v: Any) -> Path:
        """Coerce output directory to Path object.
        
        Args:
            v: Raw input value.
            
        Returns:
            Path object.
        """
        return Path(v)

    @field_validator("allowed_extensions", mode="before")
    @classmethod
    def normalize_extensions(cls, v: Any) -> list[str]:
        """Normalize file extensions to lowercase with leading dot.
        
        Args:
            v: Raw input value.
            
        Returns:
            List of normalized extensions.
        """
        if v is None:
            return [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif"]
        if isinstance(v, str):
            ext = v.lower()
            return [ext if ext.startswith(".") else f".{ext}"]
        if isinstance(v, list):
            result: list[str] = []
            for x in v:
                s = str(x).lower()
                result.append(s if s.startswith(".") else f".{s}")
            return result
        raise ValueError("allowed_extensions must be a list of extensions")

    @model_validator(mode="after")
    def validate_paths(self) -> Self:
        """Validate that input paths exist.
        
        Returns:
            Self after validation.
            
        Raises:
            ValueError: If any input path does not exist.
        """
        for p in self.input_paths:
            if not p.exists():
                raise ValueError(f"Input path does not exist: {p}")
        return self

    @property
    def allowed_extensions_set(self) -> set[str]:
        """Return normalized set of allowed extensions.
        
        Returns:
            Set of lowercase extensions with leading dots.
        """
        return {e.lower() for e in self.allowed_extensions}

    @property
    def max_file_size_bytes(self) -> int:
        """Return maximum file size in bytes.
        
        Returns:
            Maximum file size in bytes, or 0 for no limit.
        """
        if self.max_file_size_mb <= 0:
            return 0
        return int(self.max_file_size_mb * 1024 * 1024)

    def ensure_output_dir(self) -> None:
        """Ensure output directory exists and is ready.
        
        Creates the output directory if it doesn't exist.
        
        Raises:
            ValueError: If output_dir exists but is not a directory.
        """
        if self.output_dir.exists() and not self.output_dir.is_dir():
            raise ValueError(f"output_dir must be a directory: {self.output_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_json(cls, path: str | Path) -> "Settings":
        """Load settings from a JSON file.
        
        Args:
            path: Path to the JSON configuration file.
            
        Returns:
            Settings instance.
            
        Raises:
            FileNotFoundError: If the file doesn't exist.
            json.JSONDecodeError: If the file contains invalid JSON.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Settings":
        """Load settings from a YAML file.
        
        Args:
            path: Path to the YAML configuration file.
            
        Returns:
            Settings instance.
            
        Raises:
            FileNotFoundError: If the file doesn't exist.
            ImportError: If PyYAML is not installed.
        """
        try:
            import yaml
        except ImportError as e:
            raise ImportError(
                "PyYAML is required for YAML config files. "
                "Install with: pip install pyyaml"
            ) from e
        
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        """Convert settings to a dictionary.
        
        Returns:
            Dictionary representation of settings.
        """
        return self.model_dump(mode="json")
