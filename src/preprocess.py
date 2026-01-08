# This file is for noise reduction, brightness/contrast uniformity, preparation for edge detection

from __future__ import annotations
import numpy as np


def to_gray_float(img: np.ndarray) -> np.ndarray:
    """
    Convert image to grayscale float64.
    If an image is RGB, average channels.
    """
    img = img.astype(np.float64)
    if img.ndim == 3:
        img = np.mean(img, axis=2)
    return img


def robust_normalize(img: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    """
    Normalize an image robustly to [0, 1] using percentiles.
    Good for handling outliers/specular highlights.
    """
    lo, hi = np.percentile(img, [p_low, p_high])
    out = (img - lo) / (hi - lo + 1e-12)
    return np.clip(out, 0.0, 1.0)