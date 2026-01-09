# scripts/step1_build_and_test_intensity.py
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import imageio.v3 as iio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.io_utils import read_image_as_float
from src.intensity_calibration import build_master_frames, correct_measurement


def save_preview_png(img: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lo, hi = np.percentile(img, [1, 99])
    img_n = (img - lo) / (hi - lo + 1e-12)
    img_n = np.clip(img_n, 0.0, 1.0)
    img_u16 = (img_n * 65535.0).astype(np.uint16)
    iio.imwrite(out_path, img_u16)


def main() -> None:
    # <<< مسیرها را مطابق پروژه‌ی خودت تنظیم کن >>>
    data_root = PROJECT_ROOT / "data"   # یا هرچی در پروژه داری
    bias_dir = data_root / "raw" / "Bias"
    dark_dir = data_root / "raw" / "Dark"
    flat_dir = data_root / "raw" / "Flat"
    meas_path = data_root / "raw" / "Measurements_1" / "_DSC1772.JPG"

    out_dir = PROJECT_ROOT / "outputs" / "step1_intensity"
    out_dir.mkdir(parents=True, exist_ok=True)

    masters = build_master_frames(bias_dir, dark_dir, flat_dir, flat_blur_sigma=60.0)

    np.save(out_dir / "master_bias.npy", masters.bias)
    np.save(out_dir / "master_dark.npy", masters.dark)
    np.save(out_dir / "master_flat.npy", masters.flat)

    measurement = read_image_as_float(meas_path)
    corrected = correct_measurement(measurement, masters)

    np.save(out_dir / "corrected_example.npy", corrected)
    save_preview_png(corrected, out_dir / "corrected_preview.png")

    print("Step1 OK")
    print("Saved:", out_dir)


if __name__ == "__main__":
    main()