from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ImageRecord:
    """Stores analysis results for an image."""

    path: Path
    file_size: int
    width: int
    height: int
    sha256: str
    phash: int
    quality: float

    @property
    def pixels(self) -> int:
        """Returns total pixels."""
        return self.width * self.height

    @property
    def ext(self) -> str:
        """Returns the file extension."""
        return self.path.suffix.lower()
