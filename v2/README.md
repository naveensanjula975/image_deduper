# Image Deduper V2

A production-grade Python image deduplication, quality analysis, and selection pipeline.

## Features

- 🔍 **Multi-strategy deduplication**: SHA-256 exact hash, perceptual hash, and GPS proximity
- 📊 **Quality scoring**: Sharpness, colorfulness, resolution, and custom metrics
- 🗺️ **GPS filtering**: Group images by geographic location
- 🚀 **Async processing**: Efficient parallel image analysis
- 📈 **Progress tracking**: Real-time progress callbacks
- 📝 **Report generation**: JSON and HTML export reports
- 🔌 **Extensible architecture**: Plugin system for custom scoring strategies

## Requirements

- Python 3.9+
- Pillow 10.0+
- Pydantic 2.0+
- imagehash 4.3+

## Installation

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Install in development mode (optional)
pip install -e .
```

## Quick Start

### Command Line

```bash
# Basic usage
python main.py --input ./photos --output ./selected --target 200

# With GPS deduplication
python main.py --input ./photos --output ./selected --target 200 --enable-gps --gps-distance 100

# Generate report
python main.py --input ./photos --output ./selected --report report.json

# Verbose logging
python main.py --input ./photos --output ./selected --log-level DEBUG
```

### Python API

```python
from image_deduper_v2 import ImageDeduper, Settings

# Configure settings
settings = Settings(
    input_paths=["./photos"],
    output_dir="./selected",
    target_count=200,
    enable_gps_filter=True,
    gps_distance_threshold=100.0,
)

# Create pipeline and run
deduper = ImageDeduper(settings)
results = deduper.run()

print(f"Selected {len(results.exported_paths)} images")
```

### Custom Scoring Strategy

```python
from image_deduper_v2.scoring import ScoringStrategy, ImageMetrics

class MyScorer(ScoringStrategy):
    """Custom scoring that prioritizes colorful images."""
    
    def compute_score(self, metrics: ImageMetrics) -> float:
        base = metrics.quality_score
        colorfulness_bonus = metrics.colorfulness * 0.2
        return base + colorfulness_bonus

# Use custom scorer
settings = Settings(input_paths=["./photos"], output_dir="./selected")
deduper = ImageDeduper(settings, scoring_strategy=MyScorer())
```

## Configuration

### Environment Variables

All settings can be configured via environment variables with the `IMG_DEDUP_` prefix:

```bash
export IMG_DEDUP_TARGET_COUNT=200
export IMG_DEDUP_PHASH_SIZE=16
export IMG_DEDUP_HAMMING_THRESHOLD=10
```

### Configuration File

Create a `config.yaml` or `config.json`:

```yaml
target_count: 200
phash_size: 16
hamming_threshold: 10
enable_gps_filter: true
gps_distance_threshold: 100.0
allowed_extensions:
  - .jpg
  - .jpeg
  - .png
  - .webp
```

Load with:

```python
settings = Settings.from_yaml("config.yaml")
# or
settings = Settings.from_json("config.json")
```

## CLI Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--input` | path(s) | required | Input directories or files |
| `--output` | path | required | Output directory |
| `--target` | int | 200 | Target number of images |
| `--phash-size` | int | 16 | Perceptual hash size |
| `--hamming` | int | 10 | Hamming distance threshold |
| `--enable-gps` | flag | false | Enable GPS deduplication |
| `--gps-distance` | float | 50.0 | GPS distance threshold (meters) |
| `--prefer-gps` | flag | false | Prefer images with GPS data |
| `--no-recursive` | flag | false | Disable recursive scanning |
| `--workers` | int | 8 | Number of worker threads |
| `--hardlink` | flag | false | Use hardlinks instead of copying |
| `--overwrite` | flag | false | Overwrite existing output |
| `--report` | path | none | Generate JSON report |
| `--log-level` | str | INFO | Logging level |

## Architecture

```
image_deduper_v2/
├── __init__.py          # Package exports
├── config.py            # Pydantic v2 settings
├── models.py            # Data models
├── scanner.py           # File discovery
├── analyzer.py          # Image analysis
├── hasher.py            # Hash computation
├── deduplicator.py      # Deduplication strategies
├── scorer.py            # Quality scoring
├── selector.py          # Image selection
├── exporter.py          # File export
├── reporter.py          # Report generation
├── pipeline.py          # Main orchestrator
├── protocols.py         # Type protocols
├── exceptions.py        # Custom exceptions
└── utils/
    ├── __init__.py
    ├── gps.py           # GPS calculations
    ├── logging.py       # Logging setup
    └── concurrency.py   # Async helpers
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=image_deduper_v2 --cov-report=html

# Run specific test file
pytest tests/test_deduplicator.py -v
```

## License

MIT License - see LICENSE file for details.
