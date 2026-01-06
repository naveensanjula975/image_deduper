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
    parser = argparse.ArgumentParser(prog="image_deduper")
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target", type=int, default=200)
    parser.add_argument("--phash-size", type=int, default=8)
    parser.add_argument("--hamming", type=int, default=8)
    parser.add_argument("--no-recursive", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--hardlink", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO")
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
    )

    pipeline = ImageDeduper(settings=settings, logger=logger)
    pipeline.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
