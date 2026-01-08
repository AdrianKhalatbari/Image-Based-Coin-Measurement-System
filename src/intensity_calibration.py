# This file creates the “master bias/dark/flat” and performs the correction.

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import numpy as np

from src.io_utils import list_image_files, read_image_as_float


@dataclass
class MasterFrames:
    """
    Container for master calibration images used in intensity correction.
    """
    bias: np.ndarray      # B_hat
    dark: np.ndarray      # D_hat
    flat: np.ndarray      # F_hat (already scaled to mean=1)


def _mean_stack(images: List[np.ndarray]) -> np.ndarray:
    """
    Compute per-pixel mean of a list of images.
    Assumes all images have identical shape.
    """
    if len(images) == 0:
        raise ValueError("No images found to average.")
    stack = np.stack(images, axis=0)
    return np.mean(stack, axis=0)


def build_master_from_dir(dir_path: Path) -> np.ndarray:
    """
    Build a master image (mean image) from all image files in a directory (recursive).

    Parameters
    ----------
    dir_path : Path
        Directory containing calibration frames.

    Returns
    -------
    np.ndarray
        Mean image as float64.
    """
    files = list_image_files(dir_path)
    if len(files) == 0:
        raise FileNotFoundError(f"No image files found under: {dir_path}")

    imgs = [read_image_as_float(p) for p in files]

    # If images are RGB, convert to grayscale by luminance-like average (simple mean).
    # You can change this later if your pipeline prefers a specific channel.
    if imgs[0].ndim == 3:
        imgs = [np.mean(im, axis=2) for im in imgs]

    return _mean_stack(imgs)


def scale_flat_to_mean_one(flat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Scale a flat-field image so that its mean becomes 1.0, as required by the assignment.

    Parameters
    ----------
    flat : np.ndarray
        Flat-field image (mean image).
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    np.ndarray
        Scaled flat-field image with mean ~ 1.
    """
    m = float(np.mean(flat))
    return flat / (m + eps)


def build_master_frames(bias_dir: Path, dark_dir: Path, flat_dir: Path) -> MasterFrames:
    """
    Build master bias, dark, and flat frames from directories.
    Flat is scaled to mean=1 as required.  [oai_citation:4‡Assignment_description.pdf](sediment://file_00000000c94071f7a735a3e054605241)
    """
    bias_hat = build_master_from_dir(bias_dir)
    dark_hat = build_master_from_dir(dark_dir)
    flat_hat = build_master_from_dir(flat_dir)

    flat_hat = scale_flat_to_mean_one(flat_hat)

    return MasterFrames(bias=bias_hat, dark=dark_hat, flat=flat_hat)


def correct_measurement(
    measurement: np.ndarray,
    masters: MasterFrames,
    eps: float = 1e-12
) -> np.ndarray:
    """
    Apply intensity correction to a measurement image using master bias/dark/flat.

    Implements the standard correction:
        R = (measurement - B_hat - D_hat) / F_hat_scaled

    Notes
    -----
    - Flat is assumed scaled to mean=1.  [oai_citation:5‡Assignment_description.pdf](sediment://file_00000000c94071f7a735a3e054605241)
    - Output is float64.

    Parameters
    ----------
    measurement : np.ndarray
        Raw measurement image.
    masters : MasterFrames
        Master calibration images.
    eps : float
        Avoid division by zero.

    Returns
    -------
    np.ndarray
        Corrected image.
    """
    # Convert possible RGB measurement to grayscale in the same way as masters.
    meas = measurement.astype(np.float64)
    if meas.ndim == 3:
        meas = np.mean(meas, axis=2)

    numerator = meas - masters.bias - masters.dark
    denom = masters.flat

    return numerator / (denom + eps)