# Flooded Area Detection using Morphological Operators

**FURB 2024.2 — Image Processing**  
Author: Leonardo Gian Pottmayer

## Overview

This project detects flooded areas by comparing two grayscale satellite images of the same region — one captured during the rainy season and one during the dry season. The pipeline uses a sequence of morphological and pre-processing operations to isolate pixels that represent flooding: areas covered by water in the rainy image that were not covered in the dry image.

## Pipeline

| Step | Operation | Purpose |
|------|-----------|---------|
| 1–2 | Load images | Read both images as grayscale; resize dry to match rainy dimensions |
| 3–4 | Histogram equalization | Normalize brightness across both captures |
| 5–6 | Inversion | Flip pixel values so water (dark) becomes bright |
| 7–8 | Brightness enhancement | Boost pixel values to widen the gap between water and non-water |
| 9–10 | Morphological operations | Dilate rainy (fill gaps), erode dry (remove noise) |
| 11–12 | Binarization | Threshold each image to isolate water pixels |
| 13 | Flood detection | Subtract dry mask from rainy mask to isolate flooded areas |
| 14 | Overlay visualization | Color flooded regions in blue over the dry-period image |

## Project Structure

```
cv-river-flooding-detection/
├── images/
│   ├── raw/
│   │   ├── rainy.png       # Satellite image — rainy season
│   │   └── dry.png         # Satellite image — dry season
│   └── debug/              # Output images for each pipeline step (generated)
├── script.py               # Standalone Python script
├── script.ipynb            # Jupyter notebook with step-by-step explanations
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

**Script:**
```bash
python script.py
```

**Notebook:**

Open `script.ipynb` in Jupyter or VS Code and run cells sequentially. Each step displays its output image inline before moving to the next.

## Configuration

All tunable parameters are in the `Config` dataclass at the top of `script.py` / `script.ipynb`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rainy_brightness` | `60` | Brightness offset added to the rainy image after inversion |
| `dry_brightness` | `50` | Brightness offset added to the dry image after inversion |
| `kernel_shape` | `(1, 1)` | Morphological kernel size |
| `dilate_iterations` | `1` | Dilation iterations applied to the rainy image |
| `erode_iterations` | `5` | Erosion iterations applied to the dry image |
| `rainy_threshold` | `205` | Binarization threshold for the rainy image |
| `dry_threshold` | `240` | Binarization threshold for the dry image |
| `overlay_color` | `(255, 120, 40)` | Flood highlight color in BGR (default: blue) |
| `apply_cleanup` | `False` | Apply morphological opening to the flood mask to reduce noise |
