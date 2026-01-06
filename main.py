from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

from image_deduper import AppSettings, ImageDeduper
from image_deduper.logging_utils import configure_logging


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
        Parsed args namespace.
    """
    parser = argparse.ArgumentParser(
        prog="image_deduper",
        description="Deduplicate images using hash, perceptual hash, and GPS location.",
    )
    parser.add_argument("--input", nargs="+", required=True, help="Input paths (files or folders)")
    parser.add_argument("--output", required=True, help="Output directory for selected images")
    parser.add_argument("--target", type=int, default=200, help="Target number of images to select")
    parser.add_argument("--phash-size", type=int, default=8, help="Perceptual hash grid size")
    parser.add_argument("--hamming", type=int, default=8, help="Hamming distance threshold")
    parser.add_argument("--no-recursive", action="store_true", help="Disable recursive scanning")
    parser.add_argument("--workers", type=int, default=8, help="Number of worker threads")
    parser.add_argument("--hardlink", action="store_true", help="Use hardlinks instead of copying")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--enable-gps", action="store_true", help="Enable GPS-based deduplication")
    parser.add_argument("--gps-distance", type=float, default=50.0, help="GPS distance threshold in meters")
    parser.add_argument("--prefer-gps", action="store_true", help="Prefer images with GPS data")
    parser.add_argument("--no-metadata", action="store_true", help="Disable metadata extraction")
    return parser.parse_args()


def main() -> int:
    """Entrypoint.

    Returns:
        Process exit code.
    """
    args = parse_args()
    level = getattr(logging, str(args.log_level).upper(), logging.INFO)
    logger = configure_logging(level=level)

    settings = AppSettings(
        input_paths=[Path(p) for p in args.input],
        output_dir=Path(args.output),
        target_count=int(args.target),
        phash_size=int(args.phash_size),
        hamming_threshold=int(args.hamming),
        recursive=not bool(args.no_recursive),
        workers=int(args.workers),
        copy_mode=not bool(args.hardlink),
        overwrite_output=bool(args.overwrite),
        enable_gps_filter=bool(args.enable_gps),
        gps_distance_threshold=float(args.gps_distance),
        prefer_gps_images=bool(args.prefer_gps),
        enable_metadata_extraction=not bool(args.no_metadata),
    )

    pipeline = ImageDeduper(settings=settings, logger=logger)
    pipeline.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
