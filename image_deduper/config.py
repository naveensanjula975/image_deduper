from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from pydantic import BaseSettings, Field, validator


class AppSettings(BaseSettings):
    """Application settings for image deduplication and selection.

    Attributes:
        input_paths: Input file/folder paths to scan for images.
        output_dir: Directory to write selected images.
        target_count: Number of unique images to output.
        phash_size: Hash grid size; phash bits = phash_size * phash_size.
        hamming_threshold: Maximum Hamming distance to consider near-duplicate.
        allowed_extensions: Allowed image file extensions.
        recursive: Whether to scan folders recursively.
        workers: Number of worker threads for hashing and analysis.
        copy_mode: Whether to copy files; if False, hardlink when possible, else copy.
        overwrite_output: Whether to overwrite existing output directory contents.
    """

    input_paths: List[Path] = Field(..., min_items=1)
    output_dir: Path = Field(...)

    target_count: int = Field(200, ge=1, le=50000)
    phash_size: int = Field(8, ge=4, le=32)
    hamming_threshold: int = Field(8, ge=0, le=64)

    allowed_extensions: List[str] = Field(
        default_factory=lambda: [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif"]
    )
    recursive: bool = Field(True)
    workers: int = Field(8, ge=1, le=128)

    copy_mode: bool = Field(True)
    overwrite_output: bool = Field(False)

    class Config:
        env_prefix = "IMG_DEDUP_"
        case_sensitive = False

    @validator("input_paths", pre=True)
    def _coerce_input_paths(cls, v: object) -> List[Path]:
        if isinstance(v, (str, Path)):
            return [Path(v)]
        if isinstance(v, list):
            return [Path(x) for x in v]
        raise ValueError("input_paths must be a path or list of paths")

    @validator("output_dir", pre=True)
    def _coerce_output_dir(cls, v: object) -> Path:
        return Path(v)

    @validator("allowed_extensions", pre=True)
    def _normalize_exts(cls, v: object) -> List[str]:
        if v is None:
            return [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif"]
        if isinstance(v, (str, Path)):
            return [str(v).lower() if str(v).startswith(".") else f".{str(v).lower()}"]
        if isinstance(v, list):
            exts = []
            for x in v:
                s = str(x).lower()
                exts.append(s if s.startswith(".") else f".{s}")
            return exts
        raise ValueError("allowed_extensions must be a list of extensions")

    @property
    def allowed_extensions_set(self) -> set[str]:
        """Returns a normalized set of allowed extensions."""
        return {e.lower() for e in self.allowed_extensions}

    def ensure_output_dir(self) -> None:
        """Ensures output directory exists and is ready."""
        if self.output_dir.exists() and not self.output_dir.is_dir():
            raise ValueError(f"output_dir must be a directory: {self.output_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
