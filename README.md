# Image Deduper

A powerful Python tool for deduplicating large collections of images using both exact hash matching and perceptual hashing (near-duplicate detection). It intelligently selects the best quality images when duplicates are found.

## Features

- **Exact Deduplication**: Uses SHA-256 to identify and remove identical files.
- **Near-Duplicate Detection**: Uses Perceptual Hashing (aHash) and Hamming distance to find similar images.
- **Quality-Aware Selection**: Automatically keeps the best version of an image based on:
  - Resolution (total pixels)
  - Sharpness (Laplacian variance)
  - File size
- **LSH Optimized**: Scalable near-duplicate clustering using Locality-Sensitive Hashing.
- **Multi-threaded**: Parallel processing for fast analysis of large image libraries.
- **Flexible Export**: Supports both copying and hardlinking to save disk space.

## Prerequisites

- Python 3.8+
- [Pillow](https://python-pillow.org/)
- [Pydantic v1](https://docs.pydantic.dev/1.10/)

## Setup

1. **Clone the repository** (or navigate to the project directory).

2. **Create a virtual environment** (recommended):
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

## Usage

Run the tool using `main.py`. You must specify at least one input path and an output directory.

### Basic Command

```powershell
python main.py --input "C:\Photos\Unsorted" --output "C:\Photos\Deduplicated" --target 500
```

### Advanced Options

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--input` | One or more space-separated paths (files or folders). | (Required) |
| `--output` | Directory where unique images will be saved. | (Required) |
| `--target` | Max number of unique images to export. | 200 |
| `--phash-size` | Perceptual hash grid size (e.g., 8 for 64-bit hash). | 8 |
| `--hamming` | Max Hamming distance for near-duplicates (lower = stricter). | 8 |
| `--no-recursive` | Disable recursive folder scanning. | False |
| `--workers` | Number of worker threads for parallel processing. | 8 |
| `--hardlink` | Use hardlinks instead of copying (saves space, same drive only). | False |
| `--overwrite` | Clear the output directory before exporting. | False |
| `--log-level` | Logging verbosity (DEBUG, INFO, WARNING, ERROR). | INFO |

### Example with multiple inputs and hardlinks

```powershell
python main.py --input folder1 folder2 image3.jpg --output processed --hardlink --overwrite --target 1000
```

## Internal Logic

1. **Discovery**: Scans input paths for images with allowed extensions.
2. **Analysis**: Computes SHA-256, Perceptual Hash, and quality metrics (sharpness * sqrt(pixels)) in parallel.
3. **Exact Dedupe**: Removes files with identical SHA-256.
4. **Near Dedupe**: Clusters images using LSH and Hamming distance. Only the highest quality image from each cluster is kept.
5. **Selection**: Sorts the remaining unique images by quality and takes the top N images as requested.
6. **Export**: Copies or hardlinks selected images to the output directory, renamed numerically (e.g., `0001.jpg`).
