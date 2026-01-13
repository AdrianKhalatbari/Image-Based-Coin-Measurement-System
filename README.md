# Coin Image Processing Pipeline

A modular Python pipeline for automated coin detection and segmentation using flat-field correction, edge-based preprocessing, and shape-based filtering.

## Overview

This project implements a complete image processing pipeline for detecting and segmenting coins from digital photographs. It addresses challenges like uneven illumination, sensor noise, textured backgrounds (checkerboards), and overlapping coins through a three-step approach:

1. **Intensity Calibration** - Flat-field correction to remove sensor artifacts
2. **Preprocessing** - Edge detection and morphological operations to separate coins
3. **Segmentation** - Contour analysis and shape filtering to extract individual coins

## Project Structure

```
root/
├── step1_calibration.py        # Step 1: Intensity calibration
├── step2_preprocessing.py      # Step 2: Preprocessing to separate coins
├── step3_segmentation.py       # Step 3: Coin extraction and segmentation
├── main.py                     # Main pipeline integration script
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── data/                       # Input data
│   ├── DIIP-images-bias/
│   ├── DIIP-images-dark/
│   ├── DIIP-images-flat/
│   ├── DIIP-images-measurements-1/
│   └── DIIP-images-measurements-2/
└── output/                     # Generated results
    ├── measurements_1/
    └── measurements_2/
```


## Installation

### Prerequisites

- Python 3.13 or higher
- pip package manager

### Setup

1. Clone or download this repository
2. Navigate to the project directory:
   ```bash
   cd Image-Based-Coin-Measurement-System
   ```
3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 📖 Usage

### Running the Complete Pipeline

Simply execute the main script:
```bash
   python CoinImageProcessingPipeline.py
   ```
Or if using Python 3 explicitly:
```bash
   python3 CoinImageProcessingPipeline.py
   ```

The script will:
1. Load and process bias, dark, and flat calibration frames
2. Apply flat-field correction to all measurement images
3. Preprocess images to separate coins from background
4. Detect and segment individual coins
5. Save all intermediate and final results to the `output/` directory


## Pipeline Details

### Step 1: Intensity Calibration

**Purpose:** Remove sensor noise and uneven illumination from images.

**Method:** Flat-field correction using the formula:
```
  Corrected = (Raw - Bias - Dark) / Flat
```

**Process:**
1. Compute master bias frame (average of multiple bias exposures)
2. Compute master dark frame (average of multiple dark exposures)
3. Compute master flat frame (average of flat exposures, corrected and normalized)
4. Apply correction formula to each measurement image

**Key Functions:**
- `calibrate_intensity()` - Computes master frames
- `apply_flat_field_correction()` - Applies correction to images

### Step 2: Preprocessing

**Purpose:** Separate coins from background and handle textured surfaces.

**Method:** Edge-based approach with morphological operations.

**Process:**
1. **Bilateral Filtering** - Removes texture (e.g., checkerboard patterns) while preserving coin edges
2. **CLAHE Enhancement** - Adaptive histogram equalization for better contrast
3. **Canny Edge Detection** - Identifies coin boundaries
4. **Morphological Operations** - Dilates, closes, and fills to create solid coin regions

**Key Functions:**
- `preprocess_image()` - Complete preprocessing pipeline

**Techniques:**
- Bilateral filter (11px, σ=100) for texture removal
- Canny edges (threshold: 20-80)
- Elliptical morphological kernels (3x3, 5x5)

### Step 3: Segmentation

**Purpose:** Detect and extract individual coins with precise measurements.

**Method:** Contour analysis with multi-criteria shape filtering.

**Process:**
1. **Contour Detection** - Find connected regions in binary mask
2. **Size Filtering** - Remove objects outside expected coin size range (0.6%-12% of image area)
3. **Shape Analysis** - Calculate multiple geometric features:
   - **Aspect Ratio** - Width/Height (should be ~1.0 for circles)
   - **Circularity** - 4π×Area/Perimeter² (1.0 for perfect circle)
   - **Solidity** - Area/ConvexHullArea (measures compactness)
   - **Extent** - Area/BoundingBoxArea (how well it fills bounding box)
4. **Multi-Criteria Filtering** - Accept coins that satisfy:
   - Aspect ratio between 0.5-1.8, AND
   - Circularity > 0.3 OR Solidity > 0.75 OR Extent > 0.65
5. **Visualization** - Draw contours, bounding boxes, centers, and labels

**Key Functions:**
- `segment_coins()` - Complete segmentation pipeline
- `create_visualization()` - Generate annotated output images

## Output Files

For each processed image, the pipeline generates:

| File | Description |
|------|-------------|
| `image_N_corrected.png` | After intensity correction (Step 1) |
| `image_N_enhanced.png` | After CLAHE enhancement (Step 2) |
| `image_N_binary.png` | Binary mask with coin regions (Step 2) |
| `image_N_segmented.png` | Final result with detected coins highlighted |
| `image_N_pipeline.png` | Side-by-side comparison of all stages |

**Visualization Legend:**
- **Green contours** - Detected coin boundaries
- **Blue rectangles** - Bounding boxes
- **Red dots** - Coin centers
- **Green labels** - Coin numbers

## Performance

**Overall Accuracy:** 70.4% (50 out of 71 coins detected)

### Detailed Results

**Measurements Set 1:**
```
| Image | Expected | Detected | Accuracy |
|-------|----------|----------|----------|
| _DSC1772 | 6 | 6 | 100% |
| _DSC1773 | 5 | 2 | 40% |
| _DSC1774 | 8 | 7 | 87.5% |
| _DSC1775 | 7 | 3 | 42.8% |
| _DSC1776 | 9 | 9 | 100% |
| _DSC1777 | 6 | 4 | 66.7% |
```
**Measurements Set 2:**
```
| Image | Expected | Detected | Accuracy |
|-------|----------|----------|----------|
| _DSC1778 | 4 | 2 | 50% |
| _DSC1779 | 8 | 5 | 62.5% |
| _DSC1780 | 4 | 4 | 100% |
| _DSC1781 | 5 | 3 | 60% |
| _DSC1782 | 9 | 6 | 66.7% |
| _DSC1783 | 5 | 3 | 60% |
```
### Strengths
- **Perfect detection on plain backgrounds** (100% accuracy on 3 images)
- **Robust intensity calibration** - Effectively removes sensor artifacts
- **Handles imperfect circles** - Flexible shape criteria
- **Works with various lighting conditions** - Adaptive thresholding

### Known Limitations
- **Coins on textured backgrounds** - Checkerboard patterns cause coins to merge
- **Touching/overlapping coins** - Detected as single blob
- **Very small coins** - May fall below minimum area threshold

### Future Improvements
To achieve higher accuracy, consider implementing:
- **Watershed segmentation** - Separate touching coins
- **Hough circle transform** - More robust circular object detection
- **Deep learning** - CNN-based coin detection
- **Color information** - Use RGB channels for better separation

## Technical Requirements

- **Python:** 3.13 or higher
- **NumPy:** 1.24.0+ (Numerical operations)
- **OpenCV:** 4.8.0+ (Image processing)
- **Matplotlib:** 3.7.0+ (Visualization)
- **scikit-image:** 0.21.0+ (Additional image processing utilities)

## Algorithm Details

### Flat-Field Correction Theory

Flat-field correction removes systematic artifacts in imaging:

- **Bias** - Electronic baseline signal (read-out noise)
- **Dark Current** - Thermal electrons generated in sensor
- **Flat Field** - Pixel-to-pixel sensitivity variations

Formula ensures corrected image has uniform response across sensor.

### Edge Detection Strategy

**Why Canny over simple thresholding?**
- Detects coin boundaries regardless of internal texture
- Robust to checkerboard patterns inside coins
- Provides clean, continuous edges for morphological filling

### Shape Metrics Explained

- **Circularity = 4π×A/P²**
  - Range: 0 to 1
  - 1.0 = perfect circle
  - <0.5 = highly irregular shape

- **Solidity = A/ConvexHullArea**
  - Range: 0 to 1
  - 1.0 = convex shape (no indentations)
  - <0.8 = concave or fragmented

- **Extent = A/BoundingBoxArea**
  - Range: 0 to 1
  - ~0.785 = perfect circle in square
  - <0.5 = poor bounding box fit
