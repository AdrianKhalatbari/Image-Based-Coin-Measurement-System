# scripts/step2_segment_preview.py
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import imageio.v3 as iio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.segmentation import segment_coins


def save_png(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, (mask * 255).astype(np.uint8))


def main():
    inp = PROJECT_ROOT / "outputs" / "step1_intensity" / "corrected_example.npy"
    out_dir = PROJECT_ROOT / "outputs" / "step2_segmentation"
    out_dir.mkdir(parents=True, exist_ok=True)

    img = np.load(inp)
    res = segment_coins(img)

    save_png(res.mask_raw_roi, out_dir / "mask_raw_roi.png")
    save_png(res.mask_roi, out_dir / "mask_roi.png")
    save_png(res.mask, out_dir / "mask_full.png")

    print("ROI:", res.roi)
    print("Raw ROI foreground ratio:", float(res.mask_raw_roi.mean()))
    print("Filtered ROI foreground ratio:", float(res.mask_roi.mean()))
    print("Coins kept:", len(res.stats))
    if res.stats:
        for i, s in enumerate(res.stats, 1):
            print(i, s["bbox_full"], "area=", s["area"], "circ=", round(s["circularity"], 3))

    print("Saved:")
    print(" -", out_dir / "mask_raw_roi.png")
    print(" -", out_dir / "mask_roi.png")
    print(" -", out_dir / "mask_full.png")


if __name__ == "__main__":
    main()