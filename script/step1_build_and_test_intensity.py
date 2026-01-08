# This script is for testing step 1: it builds the masters, corrects a sample measurement, and saves the output.
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from pathlib import Path
import numpy as np
import imageio.v3 as iio
from src.io_utils import list_image_files, read_image_as_float
from src.intensity_calibration import build_master_frames, correct_measurement



def save_as_uint16_png(img: np.ndarray, out_path: Path) -> None:
    """
    Save a float image as a 16-bit PNG for quick visual inspection.

    Notes
    -----
    This is just for debugging/visualization.
    For actual processing keep float arrays in memory or save as .npy.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Robust normalization for visualization (1st-99th percentile)
    lo, hi = np.percentile(img, [1, 99])
    img_n = (img - lo) / (hi - lo + 1e-12)
    img_n = np.clip(img_n, 0.0, 1.0)

    img_u16 = (img_n * 65535.0).astype(np.uint16)
    iio.imwrite(out_path, img_u16)


def main() -> None:
    root = Path(__file__).resolve().parents[1]  # project_root
    bias_dir = root / "data" / "raw" / "bias"
    dark_dir = root / "data" / "raw" / "dark"
    flat_dir = root / "data" / "raw" / "flat"

    meas_dir = root / "data" / "raw" / "measurements_1"
    out_dir = root / "outputs" / "step1_intensity"

    # 1) Build master frames (mean images)
    masters = build_master_frames(bias_dir=bias_dir, dark_dir=dark_dir, flat_dir=flat_dir)

    # Save masters as .npy (recommended for later steps)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "master_bias.npy", masters.bias)
    np.save(out_dir / "master_dark.npy", masters.dark)
    np.save(out_dir / "master_flat_scaled.npy", masters.flat)

    # 2) Pick one measurement image to test
    meas_files = list_image_files(meas_dir)
    if len(meas_files) == 0:
        raise FileNotFoundError(f"No measurement images found under: {meas_dir}")

    meas_path = meas_files[0]
    measurement = read_image_as_float(meas_path)

    # 3) Apply correction
    corrected = correct_measurement(measurement=measurement, masters=masters)

    # 4) Save for quick inspection
    save_as_uint16_png(corrected, out_dir / "corrected_preview.png")

    print("Step 1 done.")
    print(f"Test measurement: {meas_path.name}")
    print(f"Saved masters to: {out_dir}")
    print(f"Saved corrected preview to: {out_dir / 'corrected_preview.png'}")


if __name__ == "__main__":
    main()