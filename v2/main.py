"""Main entry point script for the image deduper V2.

Run with: python main.py --input <path> --output <path>
"""
from __future__ import annotations

import sys

from image_deduper_v2.cli import main


if __name__ == "__main__":
    sys.exit(main())
