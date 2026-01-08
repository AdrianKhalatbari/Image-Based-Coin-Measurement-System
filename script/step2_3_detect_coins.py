# This script is for run preprocessing and coin detection script and see the result (on corrected images)

# This script performs:
#  - Stage 1 intensity correction (bias/dark/flat)
#  - Stage 2 preprocessing for coin segmentation
#  - Stage 3 coin detection using Hough Circle Transform
#
# The checkerboard region is automatically excluded to avoid false detections.
# Results are saved as visualization images for inspection.

from __future__ import annotations
import sys
from pathlib import Path

# -------------------------------------------------------------------------
# Make sure the project root is on PYTHONPATH so that `src` can be imported
# -------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import cv2
import imageio.v3 as iio

from src.io_utils import list_image_files, read_image_as_float
from src.intensity_calibration import build_master_frames, correct_measurement
from src.preprocess import to_gray_float, robust_normalize
from src.coin_detection import detect_coins_hough, suppress_near_duplicates


# -------------------------------------------------------------------------
# Helper: estimate where the checkerboard starts (x coordinate)
# -------------------------------------------------------------------------
def estimate_checkerboard_start_x(img01: np.ndarray) -> int:
    """
    Estimate the x-position where the checkerboard begins.

    Strategy:
      - Compute Sobel gradient magnitude
      - Average gradient energy per column
      - Detect a strong, sustained increase in gradient energy
        (checkerboard has much stronger edges than background)

    Parameters
    ----------
    img01 : np.ndarray
        Grayscale image normalized to [0, 1].

    Returns
    -------
    x_start : int
        Estimated x-coordinate of checkerboard start.
    """
    h, w = img01.shape

    u8 = (np.clip(img01, 0.0, 1.0) * 255).astype(np.uint8)

    gx = cv2.Sobel(u8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(u8, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    col_energy = mag.mean(axis=0)

    # Smooth column-wise signal
    k = max(9, (w // 200) * 2 + 1)
    col_energy_s = cv2.GaussianBlur(col_energy.reshape(1, -1), (k, 1), 0).ravel()

    # Robust baseline (left half)
    left = col_energy_s[: w // 2]
    baseline = np.median(left)
    mad = np.median(np.abs(left - baseline)) + 1e-6

    threshold = baseline + 8.0 * mad
    high = col_energy_s > threshold

    run = max(20, w // 80)
    x_run = None
    for x in range(w - run):
        if high[x : x + run].all():
            x_run = x
            break

    # Fallback: strongest gradient jump in right half
    d = np.diff(col_energy_s, prepend=col_energy_s[0])
    search_lo = int(0.40 * w)
    x_edge = search_lo + np.argmax(d[search_lo:])

    x0 = x_edge if x_run is None else min(x_run, x_edge)

    # Clamp to reasonable range
    x0 = int(np.clip(x0, 0.45 * w, 0.92 * w))
    return x0


# -------------------------------------------------------------------------
# Visualization helper
# -------------------------------------------------------------------------
def draw_circles(img01: np.ndarray, circles, out_path: Path) -> None:
    """
    Draw detected circles on the image and save result.

    Parameters
    ----------
    img01 : np.ndarray
        Grayscale image in [0, 1].
    circles : list
        List of detected circle objects (cx, cy, r).
    out_path : Path
        Output file path.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    vis = (np.clip(img01, 0.0, 1.0) * 255).astype(np.uint8)
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    for c in circles:
        center = (int(round(c.cx)), int(round(c.cy)))
        radius = int(round(c.r))
        cv2.circle(vis, center, radius, (0, 255, 0), 2)
        cv2.circle(vis, center, 3, (0, 0, 255), -1)

    iio.imwrite(out_path, vis)


# -------------------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------------------
def main() -> None:
    root = PROJECT_ROOT

    bias_dir = root / "data" / "raw" / "bias"
    dark_dir = root / "data" / "raw" / "dark"
    flat_dir = root / "data" / "raw" / "flat"
    meas_dir = root / "data" / "raw" / "measurements_1"

    out_dir = root / "outputs" / "step2_3_detection"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # 1) Build master calibration frames
    # ---------------------------------------------------------------------
    masters = build_master_frames(bias_dir, dark_dir, flat_dir)

    meas_files = list_image_files(meas_dir)
    if not meas_files:
        raise FileNotFoundError(f"No measurement images found in {meas_dir}")

    # ---------------------------------------------------------------------
    # Process each measurement image
    # ---------------------------------------------------------------------
    for meas_path in meas_files:
        measurement = read_image_as_float(meas_path)

        # 2) Intensity correction
        corrected = correct_measurement(measurement, masters)

        # 3) Preprocessing
        gray = to_gray_float(corrected)
        # 4) Preprocessing for detection
        gray = to_gray_float(corrected)

        # NOTE: robust_normalize signature is (p_low, p_high)
        img01 = robust_normalize(gray, 1, 99)

        # --- ROI: cut off checkerboard + add a safety margin to avoid boundary artifacts ---
        h, w = img01.shape[:2]
        x0 = estimate_checkerboard_start_x(img01)

        margin = int(0.06 * w)  # 6% of width as safety margin (good default)
        x_roi = max(0, x0 - margin)

        roi = img01[:, :x_roi].copy()

        # --- Contrast stabilization (helps across different exposures) ---
        roi = apply_clahe_gray01(roi, clip_limit=2.0, tile_grid_size=(8, 8))

        # --- Mild blur to stabilize Hough ---
        roi_blur = cv2.GaussianBlur(roi, (9, 9), 1.5)

        # 5) Detect circles (make it a bit stricter to reduce false positives)
        circles = detect_coins_hough(
            roi_blur,
            dp=1.2,
            min_dist=190.0,
            param1=140.0,
            param2=75.0,   # was 60 -> stricter
            min_radius=90,
            max_radius=235,
        )

        circles = suppress_near_duplicates(circles, min_center_dist=70.0)

        # --- Post-filter: remove circles that don't look like coins ---
        circles = filter_circles_by_contrast(
            roi, circles,
            inner_scale=0.85,
            outer_scale=1.25,
            min_delta=0.03
        )

        print(
            f"{meas_path.name}: detected {len(circles)} coins (checkerboard ~x={x0}, roi width={x_roi}/{w}, margin={margin})"
        )

        # -----------------------------------------------------------------
        # Save visualization
        # -----------------------------------------------------------------
        out_img = out_dir / f"{meas_path.stem}_circles.png"
        draw_circles(img01, circles, out_img)

    print(f"\nSaved circle overlays to: {out_dir}")

def apply_clahe_gray01(img01: np.ndarray, clip_limit: float = 2.0, tile_grid_size=(8, 8)) -> np.ndarray:
    """
    Apply CLAHE to a grayscale float image in [0,1], returns float in [0,1].
    Helps stabilize contrast across images.
    """
    u8 = (np.clip(img01, 0.0, 1.0) * 255.0).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    out = clahe.apply(u8)
    return out.astype(np.float32) / 255.0


def filter_circles_by_contrast(
    img01: np.ndarray,
    circles,
    inner_scale: float = 0.85,
    outer_scale: float = 1.25,
    min_delta: float = 0.03,
) -> list:
    """
    Keep circles that look like real coins by checking intensity contrast:
    compare mean intensity in an inner disk vs an outer ring.
    If (outer_mean - inner_mean) is large enough => likely a dark coin on bright background.
    (Sign may vary depending on your images; we handle both by abs delta.)
    """
    h, w = img01.shape[:2]
    kept = []

    for c in circles:
        cx, cy, r = float(c.cx), float(c.cy), float(c.r)
        if r <= 0:
            continue

        x0 = int(max(0, np.floor(cx - outer_scale * r)))
        x1 = int(min(w, np.ceil(cx + outer_scale * r)))
        y0 = int(max(0, np.floor(cy - outer_scale * r)))
        y1 = int(min(h, np.ceil(cy + outer_scale * r)))

        patch = img01[y0:y1, x0:x1]
        if patch.size == 0:
            continue

        yy, xx = np.mgrid[y0:y1, x0:x1]
        dist2 = (xx - cx) ** 2 + (yy - cy) ** 2

        inner_r2 = (inner_scale * r) ** 2
        outer_r2 = (outer_scale * r) ** 2

        inner_mask = dist2 <= inner_r2
        ring_mask = (dist2 > r**2) & (dist2 <= outer_r2)

        # Need enough pixels to be meaningful
        if inner_mask.sum() < 200 or ring_mask.sum() < 200:
            continue

        inner_mean = float(patch[inner_mask].mean())
        ring_mean = float(patch[ring_mask].mean())

        delta = abs(ring_mean - inner_mean)
        if delta >= min_delta:
            kept.append(c)

    return kept


if __name__ == "__main__":
    main()
