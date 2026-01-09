# This file creates the “master bias/dark/flat” and performs the correction.

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from src.io_utils import list_image_files, read_image_as_float


@dataclass
class MasterFrames:
    bias: np.ndarray      # B_hat
    dark: np.ndarray      # D_hat  (bias-corrected)
    flat: np.ndarray      # F_hat  (illumination field, mean=1)


def _to_gray(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float64)
    if img.ndim == 3:
        img = img.mean(axis=2)
    return img


def _mean_stack(images: List[np.ndarray]) -> np.ndarray:
    if len(images) == 0:
        raise ValueError("No images found to average.")
    ref_shape = images[0].shape
    for im in images:
        if im.shape != ref_shape:
            raise ValueError("All images must have the same shape.")
    return np.mean(np.stack(images, axis=0), axis=0)


def build_master_from_dir(dir_path: Path) -> np.ndarray:
    files = list_image_files(dir_path)
    if len(files) == 0:
        raise FileNotFoundError(f"No image files found under: {dir_path}")
    imgs = [_to_gray(read_image_as_float(p)) for p in files]
    return _mean_stack(imgs)


def _gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """
    Large-scale smoothing to estimate illumination field.
    Uses OpenCV if available, otherwise falls back to a simple separable approximation.
    """
    try:
        import cv2
        # ksize=0 lets OpenCV compute from sigma
        return cv2.GaussianBlur(image.astype(np.float32), ksize=(0, 0), sigmaX=sigma).astype(np.float64)
    except Exception:
        # Fallback: very simple blur via repeated box filter
        # (Not as good as Gaussian, but prevents division by checkerboard pattern.)
        out = image.copy()
        for _ in range(10):
            out = (out +
                   np.roll(out, 1, 0) + np.roll(out, -1, 0) +
                   np.roll(out, 1, 1) + np.roll(out, -1, 1)) / 5.0
        return out


def build_master_frames(
    bias_dir: Path,
    dark_dir: Path,
    flat_dir: Path,
    flat_blur_sigma: float = 60.0,
    eps: float = 1e-12
) -> MasterFrames:
    """
    Build properly corrected masters.

    Steps:
      B = mean(bias)
      D = mean(dark) - B
      F_raw = mean(flat) - B - D
      F_illum = smooth(F_raw)   (remove checkerboard/pattern)
      F = F_illum / mean(F_illum)
    """
    bias_hat = build_master_from_dir(bias_dir)

    dark_raw = build_master_from_dir(dark_dir)
    dark_hat = dark_raw - bias_hat

    flat_raw = build_master_from_dir(flat_dir)
    flat_corr = flat_raw - bias_hat - dark_hat

    # IMPORTANT: remove checkerboard/pattern by strong smoothing
    flat_illum = _gaussian_blur(flat_corr, sigma=flat_blur_sigma)

    # Avoid zeros / negatives causing division problems
    flat_illum = np.clip(flat_illum, eps, None)

    flat_hat = flat_illum / (float(flat_illum.mean()) + eps)

    return MasterFrames(bias=bias_hat, dark=dark_hat, flat=flat_hat)


def correct_measurement(measurement: np.ndarray, masters: MasterFrames, eps: float = 1e-12) -> np.ndarray:
    meas = _to_gray(measurement)
    numerator = meas - masters.bias - masters.dark
    return numerator / (masters.flat + eps)