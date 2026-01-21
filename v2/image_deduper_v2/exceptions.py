"""Custom exceptions for the image deduper package.

This module defines a hierarchy of exceptions for fine-grained error handling
throughout the deduplication pipeline.
"""
from __future__ import annotations


class ImageDeduperError(Exception):
    """Base exception for all image deduper errors.
    
    Attributes:
        message: Human-readable error description.
        details: Optional additional context.
    """

    def __init__(self, message: str, details: str | None = None) -> None:
        """Initialize the exception.
        
        Args:
            message: Human-readable error description.
            details: Optional additional context for debugging.
        """
        super().__init__(message)
        self.message = message
        self.details = details

    def __str__(self) -> str:
        """Return string representation including details if present."""
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class ConfigurationError(ImageDeduperError):
    """Raised when configuration validation fails.
    
    Examples:
        - Invalid input path
        - Invalid output directory
        - Conflicting settings
    """
    pass


class AnalysisError(ImageDeduperError):
    """Raised when image analysis fails.
    
    Examples:
        - Corrupted image file
        - Unsupported format
        - Missing required metadata
    """

    def __init__(
        self,
        message: str,
        file_path: str | None = None,
        details: str | None = None,
    ) -> None:
        """Initialize the analysis error.
        
        Args:
            message: Human-readable error description.
            file_path: Path to the problematic file.
            details: Optional additional context.
        """
        super().__init__(message, details)
        self.file_path = file_path


class ExportError(ImageDeduperError):
    """Raised when file export operations fail.
    
    Examples:
        - Permission denied
        - Disk full
        - Invalid destination
    """

    def __init__(
        self,
        message: str,
        source_path: str | None = None,
        destination_path: str | None = None,
        details: str | None = None,
    ) -> None:
        """Initialize the export error.
        
        Args:
            message: Human-readable error description.
            source_path: Path to the source file.
            destination_path: Path to the destination file.
            details: Optional additional context.
        """
        super().__init__(message, details)
        self.source_path = source_path
        self.destination_path = destination_path


class HashingError(ImageDeduperError):
    """Raised when hash computation fails.
    
    Examples:
        - File read error during hashing
        - Image too small for perceptual hash
    """
    pass


class DeduplicationError(ImageDeduperError):
    """Raised when deduplication process fails.
    
    Examples:
        - Clustering algorithm failure
        - Invalid threshold parameters
    """
    pass
