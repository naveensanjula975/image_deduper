# Image Deduper

A production-grade Python utility for deduplicating images using SHA-256, Perceptual Hashing (aHash), and GPS metadata.

## Features
- **Triple Deduplication**: Exact Match (SHA-256) + Near-Duplicate (aHash) + Location-Based (GPS).
- **Quality Analysis**: Intelligently keeps the best version based on resolution and sharpness.
- **GPS Filtering**: Cluster images by coordinates with configurable distance thresholds.
- **Performance**: Multi-threaded analysis with LSH optimization.

## Setup
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

## Usage
### Basic
```powershell
python main.py --input path/to/photos --output path/to/results --target 200
```

### Advanced (with GPS)
```powershell
python main.py --input "C:\Photos" --output "./output" --target 200 --enable-gps --gps-distance 50 --prefer-gps
```

## Core Arguments
- `--input`: Space-separated file or folder paths.
- `--output`: Result directory.
- `--target`: Number of unique images to select (default: 200).
- `--enable-gps`: Enable location-based deduplication.
- `--gps-distance`: Meters to consider "same location" (default: 50.0).
- `--prefer-gps`: Favor images containing location metadata.
- `--hardlink`: Use hardlinks to save space (same drive only).
- `--overwrite`: Clear output folder before starting.
