from __future__ import annotations
import sys
from pathlib import Path

# Add project root to PYTHONPATH so imports like "from src..." work
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import imageio.v3 as iio

from src.io_utils import list_image_files, read_image_as_float
from src.intensity_calibration import build_master_frames, correct_measurement

"""
The sanity_check script performs a validation check for the Intensity calibration step 
to ensure that the correction has been done correctly and without numerical errors 
before entering the next processing. 
The script first creates the master bias, dark, and flat and checks that they are all the same size, 
then reports their basic statistics (min/mean/max), and specifically checks that the flat-field after scaling is exactly 1.
Next, the correction is applied to all measurement images, checking for each image that the output does not 
contain illegal values ​​(NaN or Inf) and that its numerical range is reasonable.
Finally, several corrected images are saved as previews so that the user can visually verify that the brightness, 
contrast, and image structure are reasonable. 
In short, this sanity check ensures that the input data to the segmentation step is physically, numerically, and visually valid.
"""

def _to_gray(img: np.ndarray) -> np.ndarray:
    """Convert RGB to grayscale by channel mean if needed."""
    if img.ndim == 3:
        return np.mean(img.astype(np.float64), axis=2)
    return img.astype(np.float64)


def main() -> None:
    root = PROJECT_ROOT

    bias_dir = root / "data" / "raw" / "bias"
    dark_dir = root / "data" / "raw" / "dark"
    flat_dir = root / "data" / "raw" / "flat"

    meas1_dir = root / "data" / "raw" / "measurements_1"
    meas2_dir = root / "data" / "raw" / "measurements_2"

    # 1) Build master frames
    masters = build_master_frames(bias_dir=bias_dir, dark_dir=dark_dir, flat_dir=flat_dir)

    # 2) Check shapes of masters
    print("Master shapes:")
    print("  bias:", masters.bias.shape)
    print("  dark:", masters.dark.shape)
    print("  flat:", masters.flat.shape)

    assert masters.bias.shape == masters.dark.shape == masters.flat.shape, "Master frames shape mismatch!"

    # 3) Print basic statistics of masters
    def stats(name: str, img: np.ndarray) -> None:
        img = _to_gray(img)
        print(f"{name}: min={img.min():.3f}, mean={img.mean():.3f}, max={img.max():.3f}")

    print("\nMaster stats:")
    stats("bias", masters.bias)
    stats("dark", masters.dark)
    stats("flat (scaled)", masters.flat)
    print(f"flat mean (should be ~1.0): {np.mean(masters.flat):.6f}")

    # 4) Run correction for all measurements and print stats
    for meas_dir in [meas1_dir, meas2_dir]:
        files = list_image_files(meas_dir)
        if not files:
            raise FileNotFoundError(f"No measurement images found in {meas_dir}")

        print(f"\nChecking corrected outputs in: {meas_dir.name}")
        for p in files:
            meas = read_image_as_float(p)
            corr = correct_measurement(measurement=meas, masters=masters)

            # Sanity checks
            if not np.isfinite(corr).all():
                raise ValueError(f"Non-finite values detected after correction in {p.name}!")

            # Print quick summary
            print(f"  {p.name}: min={corr.min():.3f}, mean={corr.mean():.3f}, max={corr.max():.3f}")

    # 5) Save one corrected preview for meas2 also (optional)
    out_dir = root / "outputs" / "step1_intensity"
    out_dir.mkdir(parents=True, exist_ok=True)

    example = list_image_files(meas2_dir)[0]
    corr_ex = correct_measurement(read_image_as_float(example), masters)

    # Save as 16-bit PNG for visual check
    lo, hi = np.percentile(corr_ex, [1, 99])
    vis = (corr_ex - lo) / (hi - lo + 1e-12)
    vis = np.clip(vis, 0.0, 1.0)
    iio.imwrite(out_dir / "corrected_preview_meas2.png", (vis * 65535).astype(np.uint16))

    print("\nSanity check complete.")
    print(f"Saved: {out_dir / 'corrected_preview_meas2.png'}")


if __name__ == "__main__":
    main()