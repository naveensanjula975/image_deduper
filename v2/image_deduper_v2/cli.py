"""Command-line interface for the image deduper.

This module provides the CLI entry point and argument parsing.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List

from image_deduper_v2 import ImageDeduper, Settings
from image_deduper_v2.utils.logging import configure_logging


def parse_args(args: List[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.
    
    Args:
        args: Optional list of arguments (defaults to sys.argv).
        
    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        prog="image-deduper",
        description=(
            "Deduplicate images using hash, perceptual hash, and GPS location. "
            "Selects top-N images by quality and exports to output directory."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--input", "-i",
        nargs="+",
        required=True,
        metavar="PATH",
        help="Input paths (files or folders) to scan for images",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        metavar="DIR",
        help="Output directory for selected images",
    )

    # Selection settings
    parser.add_argument(
        "--target", "-n",
        type=int,
        default=200,
        metavar="N",
        help="Target number of images to select",
    )

    # Hash settings
    parser.add_argument(
        "--phash-size",
        type=int,
        default=16,
        metavar="N",
        help="Perceptual hash grid size (bits = N²)",
    )
    parser.add_argument(
        "--hamming",
        type=int,
        default=10,
        metavar="N",
        help="Hamming distance threshold for near-duplicates",
    )

    # GPS settings
    parser.add_argument(
        "--enable-gps",
        action="store_true",
        help="Enable GPS-based deduplication",
    )
    parser.add_argument(
        "--gps-distance",
        type=float,
        default=50.0,
        metavar="M",
        help="GPS distance threshold in meters",
    )
    parser.add_argument(
        "--prefer-gps",
        action="store_true",
        help="Prefer images with GPS data when selecting best",
    )

    # Processing settings
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Disable recursive directory scanning",
    )
    parser.add_argument(
        "--workers", "-j",
        type=int,
        default=8,
        metavar="N",
        help="Number of worker threads for parallel processing",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Disable EXIF metadata extraction",
    )

    # Output settings
    parser.add_argument(
        "--hardlink",
        action="store_true",
        help="Use hardlinks instead of copying files",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output directory contents",
    )
    parser.add_argument(
        "--report",
        type=str,
        metavar="PATH",
        help="Generate JSON report at specified path",
    )

    # Logging settings
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        metavar="PATH",
        help="Write logs to file",
    )

    # Config file
    parser.add_argument(
        "--config",
        type=str,
        metavar="PATH",
        help="Load settings from JSON or YAML config file",
    )

    return parser.parse_args(args)


def main(args: List[str] | None = None) -> int:
    """Main entry point for the CLI.
    
    Args:
        args: Optional list of arguments.
        
    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    parsed = parse_args(args)

    # Configure logging
    level = getattr(logging, parsed.log_level.upper(), logging.INFO)
    logger = configure_logging(
        level=level,
        log_file=parsed.log_file,
    )

    try:
        # Load settings from config file or CLI arguments
        if parsed.config:
            config_path = Path(parsed.config)
            if config_path.suffix.lower() in (".yaml", ".yml"):
                settings = Settings.from_yaml(config_path)
            else:
                settings = Settings.from_json(config_path)
            
            # Override with CLI arguments
            settings = _apply_cli_overrides(settings, parsed)
        else:
            settings = _settings_from_args(parsed)

        logger.info(f"Input paths: {settings.input_paths}")
        logger.info(f"Output directory: {settings.output_dir}")
        logger.info(f"Target count: {settings.target_count}")

        # Create and run pipeline
        deduper = ImageDeduper(settings, logger=logger)
        result = deduper.run()

        if result.exported_count > 0:
            logger.info("Deduplication completed successfully!")
            return 0
        else:
            logger.warning("No images were exported")
            return 1

    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return 130
    except Exception as e:
        logger.exception(f"Error: {e}")
        return 1


def _settings_from_args(args: argparse.Namespace) -> Settings:
    """Create Settings from parsed arguments.
    
    Args:
        args: Parsed command-line arguments.
        
    Returns:
        Settings instance.
    """
    return Settings(
        input_paths=[Path(p) for p in args.input],
        output_dir=Path(args.output),
        target_count=args.target,
        phash_size=args.phash_size,
        hamming_threshold=args.hamming,
        recursive=not args.no_recursive,
        workers=args.workers,
        copy_mode=not args.hardlink,
        overwrite_output=args.overwrite,
        enable_gps_filter=args.enable_gps,
        gps_distance_threshold=args.gps_distance,
        prefer_gps_images=args.prefer_gps,
        enable_metadata_extraction=not args.no_metadata,
        report_path=Path(args.report) if args.report else None,
    )


def _apply_cli_overrides(
    settings: Settings,
    args: argparse.Namespace,
) -> Settings:
    """Apply CLI argument overrides to loaded settings.
    
    Args:
        settings: Base settings from config file.
        args: Parsed CLI arguments.
        
    Returns:
        Settings with CLI overrides applied.
    """
    # Build override dict from CLI args
    overrides = {
        "input_paths": [Path(p) for p in args.input],
        "output_dir": Path(args.output),
    }

    # Only override if explicitly specified
    if args.target != 200:
        overrides["target_count"] = args.target
    if args.phash_size != 16:
        overrides["phash_size"] = args.phash_size
    if args.hamming != 10:
        overrides["hamming_threshold"] = args.hamming
    if args.no_recursive:
        overrides["recursive"] = False
    if args.workers != 8:
        overrides["workers"] = args.workers
    if args.hardlink:
        overrides["copy_mode"] = False
    if args.overwrite:
        overrides["overwrite_output"] = True
    if args.enable_gps:
        overrides["enable_gps_filter"] = True
    if args.gps_distance != 50.0:
        overrides["gps_distance_threshold"] = args.gps_distance
    if args.prefer_gps:
        overrides["prefer_gps_images"] = True
    if args.no_metadata:
        overrides["enable_metadata_extraction"] = False
    if args.report:
        overrides["report_path"] = Path(args.report)

    # Create new settings with overrides
    return settings.model_copy(update=overrides)


if __name__ == "__main__":
    sys.exit(main())
