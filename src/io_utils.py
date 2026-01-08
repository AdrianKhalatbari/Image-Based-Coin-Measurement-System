# This file is for reading images from folders and converting them into a float array.

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple

import numpy as np
import imageio.v3 as iio


def list_image_files(root: Path, exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")) -> List[Path]:
    """
    Recursively list image files under a directory.

    Parameters
    ----------
    root : Path
        Root directory to search.
    exts : tuple[str, ...]
        Allowed image extensions.

    Returns
    -------
    List[Path]
        Sorted list of image file paths.
    """
    files: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    files.sort()
    return files


def read_image_as_float(path: Path) -> np.ndarray:
    """
    Read an image from the disk and convert it to a float64 numpy array.

    Notes
    -----
    - Keeps original dynamic range (e.g., 0..255 or 0..65535) but converts dtype to float64.
    - Returns a 2D array for grayscale images, or 3D for color images.

    Parameters
    ----------
    path : Path
        Image path.

    Returns
    -------
    np.ndarray
        Image as float64.
    """
    img = iio.imread(path)
    return np.asarray(img, dtype=np.float64)