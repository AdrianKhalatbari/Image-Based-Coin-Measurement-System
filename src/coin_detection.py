# We here recognize the "coins" as circular in shape

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import cv2


@dataclass
class Circle:
    """Detected circle representing a coin."""
    cx: float
    cy: float
    r: float
    score: float  # detection confidence (proxy)


def detect_coins_hough(
    img01: np.ndarray,
    dp: float = 1.2,
    min_dist: float = 120.0,
    param1: float = 120.0,
    param2: float = 35.0,
    min_radius: int = 40,
    max_radius: int = 220,
) -> List[Circle]:
    """
    Detect coins using HoughCircles on a preprocessed grayscale image in [0,1].

    Parameters
    ----------
    img01 : np.ndarray
        Grayscale image normalized to [0,1].
    dp : float
        Inverse ratio of the accumulator resolution to the image resolution.
    min_dist : float
        Minimum distance between detected centers.
    param1 : float
        Canny high threshold (OpenCV uses this internally).
    param2 : float
        Accumulator threshold for circle centers (higher = fewer detections).
    min_radius, max_radius : int
        Radius range for circle detection in pixels.

    Returns
    -------
    List[Circle]
        List of detected circles.
    """
    # OpenCV expects 8-bit image for HoughCircles typically
    img_u8 = (np.clip(img01, 0.0, 1.0) * 255).astype(np.uint8)

    # Mild blur helps Canny/Hough stability
    img_blur = cv2.GaussianBlur(img_u8, (9, 9), 1.5)

    circles = cv2.HoughCircles(
        img_blur,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    out: List[Circle] = []
    if circles is None:
        return out

    circles = circles[0]  # shape: (N, 3) => x, y, r
    for (x, y, r) in circles:
        # OpenCV does not expose a direct confidence score here; we use r as proxy
        out.append(Circle(cx=float(x), cy=float(y), r=float(r), score=float(r)))

    # Sort by score descending (optional)
    out.sort(key=lambda c: c.score, reverse=True)
    return out


def suppress_near_duplicates(circles: List[Circle], min_center_dist: float = 40.0) -> List[Circle]:
    """
    Remove duplicate detections by keeping the first circle and removing others
    whose centers are too close.

    Parameters
    ----------
    circles : List[Circle]
        Candidate circles.
    min_center_dist : float
        Minimum allowed center distance between circles.

    Returns
    -------
    List[Circle]
        Filtered circles.
    """
    kept: List[Circle] = []
    for c in circles:
        ok = True
        for k in kept:
            d = np.hypot(c.cx - k.cx, c.cy - k.cy)
            if d < min_center_dist:
                ok = False
                break
        if ok:
            kept.append(c)
    return kept