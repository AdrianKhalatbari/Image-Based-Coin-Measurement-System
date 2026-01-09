# This file is for reading images from folders and converting them into a float array.

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple

import numpy as np
import imageio.v3 as iio

def list_image_files(
    root: Path,
    exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
) -> List[Path]:
    """
    Recursively list image files under a directory.
    """
    files: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    files.sort()
    return files


def read_image_as_float(path: Path) -> np.ndarray:
    """
    Read an image from disk and convert it to float64 numpy array.
    Keeps original dynamic range (uint8 -> 0..255, uint16 -> 0..65535).
    """
    img = iio.imread(path)
    return np.asarray(img, dtype=np.float64)