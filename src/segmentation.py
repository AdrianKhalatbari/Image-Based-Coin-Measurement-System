# src/segmentation.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np


@dataclass
class SegmentResult:
    mask: np.ndarray                 # uint8 (0/1) full-size filtered mask
    mask_roi: np.ndarray             # uint8 (0/1) filtered mask in ROI coords
    mask_raw_roi: np.ndarray         # uint8 (0/1) raw mask in ROI coords (before filtering)
    roi: Tuple[int, int, int, int]   # (x, y, w, h)
    stats: List[dict]


def _to_gray(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float64)
    if img.ndim == 3:
        img = img.mean(axis=2)
    return img


def _gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    import cv2
    return cv2.GaussianBlur(img.astype(np.float32), (0, 0), sigmaX=sigma).astype(np.float64)


def _auto_roi_exclude_checkerboard(gray: np.ndarray) -> Tuple[int, int, int, int]:
    import cv2

    g = np.clip(gray, 0.0, None)
    h, w = g.shape

    scale = 0.25
    g_small = cv2.resize(g.astype(np.float32), (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    gx = cv2.Sobel(g_small, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g_small, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    col_energy = mag.mean(axis=0)
    col_energy_s = cv2.GaussianBlur(col_energy.reshape(1, -1), (0, 0), sigmaX=5).ravel()

    med = float(np.median(col_energy_s))
    mad = float(np.median(np.abs(col_energy_s - med)) + 1e-9)
    thr = med + 3.0 * mad

    candidates = np.where(col_energy_s > thr)[0]
    if len(candidates) == 0:
        x_cut_small = int(0.65 * g_small.shape[1])
    else:
        x_cut_small = int(candidates[0])
        x_cut_small = max(0, x_cut_small - int(0.03 * g_small.shape[1]))

    x_cut = int(x_cut_small / scale)

    return (0, 0, max(1, min(w, x_cut)), h)


def segment_coins(
    img: np.ndarray,
    roi: Optional[Tuple[int, int, int, int]] = None,
    bg_sigma: float = 60.0,
    denoise_sigma: float = 1.2,
    adaptive_block_size: int = 51,
    adaptive_C: int = 6,
    min_area: int = 4000,
    max_area: int = 400000,
    circularity_min: float = 0.42,
) -> SegmentResult:
    import cv2

    g = _to_gray(img)
    g = np.clip(g, 0.0, None)

    if roi is None:
        roi = _auto_roi_exclude_checkerboard(g)
    x, y, w, h = roi
    g_roi = g[y:y+h, x:x+w]

    # Residual shading removal
    bg = _gaussian_blur(g_roi, sigma=bg_sigma)
    g_norm = g_roi / (bg + 1e-12)

    # Denoise
    g_dn = _gaussian_blur(g_norm, sigma=denoise_sigma)

    # Adaptive threshold (dark -> foreground)
    g8 = cv2.normalize(g_dn, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if adaptive_block_size % 2 == 0:
        adaptive_block_size += 1
    adaptive_block_size = max(3, adaptive_block_size)

    th = cv2.adaptiveThreshold(
        g8, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        adaptive_block_size,
        adaptive_C
    )

    # Decide orientation automatically:
    # We expect background to be the majority in ROI.
    # If white pixels are > 50%, invert.
    if np.mean(th == 255) > 0.5:
        th = cv2.bitwise_not(th)

    mask_raw = (th > 0).astype(np.uint8)

    # Morph cleanup on raw mask (helps bimetal coins)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask_raw, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)

    # Fill holes
    mask_ff = mask.copy()
    hh, ww = mask_ff.shape
    tmp = np.zeros((hh + 2, ww + 2), np.uint8)
    cv2.floodFill(mask_ff, tmp, (0, 0), 1)
    holes = (mask_ff == 0).astype(np.uint8)
    mask = np.clip(mask + holes, 0, 1).astype(np.uint8)

    # Added by adrian
    # --- Extra closing to solidify bimetal coin (ring-like segmentation)
    k_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_big, iterations=1)

    # --- Remove right-edge artifacts inside ROI
    margin = int(0.05 * mask.shape[1])  # ~3% width
    mask[:, -margin:] = 0

    # Connected components
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    out_roi = np.zeros_like(mask, dtype=np.uint8)
    kept: List[dict] = []

    # small smoothing kernel for each component before circularity
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    for lab in range(1, n):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue

        comp = (labels == lab).astype(np.uint8) * 255
        comp = cv2.morphologyEx(comp, cv2.MORPH_CLOSE, k2, iterations=1)

        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        per = cv2.arcLength(cnt, True)
        if per <= 1e-6:
            continue

        circ = float(4.0 * np.pi * area / (per * per))
        if circ < circularity_min:
            continue

        out_roi[labels == lab] = 1

        bx = int(stats[lab, cv2.CC_STAT_LEFT])
        by = int(stats[lab, cv2.CC_STAT_TOP])
        bw = int(stats[lab, cv2.CC_STAT_WIDTH])
        bh = int(stats[lab, cv2.CC_STAT_HEIGHT])

        kept.append({
            "label": lab,
            "area": area,
            "circularity": circ,
            "bbox_full": (x + bx, y + by, bw, bh),
        })

    full_mask = np.zeros_like(g, dtype=np.uint8)
    full_mask[y:y+h, x:x+w] = out_roi

    # Adde by Adrian
    # --- Remove right-edge artifacts inside ROI (paper boundary / checkerboard remnants)
    margin = int(0.05 * mask.shape[1])  # 3% of ROI width ~ 70px
    mask[:, -margin:] = 0

    return SegmentResult(
        mask=full_mask,
        mask_roi=out_roi,
        mask_raw_roi=mask,
        roi=roi,
        stats=kept
    )