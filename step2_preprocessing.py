"""
Step 2: Preprocessing to Separate Coins
Implements edge-based preprocessing with bilateral filtering and morphological operations.
"""

import cv2
import numpy as np


def preprocess_image(corrected_image, verbose=False):
    """
    Preprocess image to separate coins from the background.

    Uses bilateral filtering to remove texture (like checkerboard patterns),
    CLAHE for contrast enhancement, Canny edge detection to find coin boundaries,
    and morphological operations to create solid coin masks.

    Args:
        corrected_image (np.ndarray): Intensity-corrected image from Step 1

    Returns:
        tuple: (enhanced_image, binary_mask)
            - enhanced_image: CLAHE-enhanced grayscale image
            - binary_mask: Binary mask with filled coin regions
    """
    
    # Removing the whitest parts to reduce glare effects with truncation thresholding
    _, corrected_image = cv2.threshold(corrected_image, 240, 255, cv2.THRESH_TRUNC)

    if verbose:
        print("=" * 60)
        print("STEP 2: Preprocessing to Separate Coins")
        print("=" * 60)

    if verbose:
        print("\n[2.1] Bilateral Filter (preserve edges, remove texture)...")
    # Bilateral filter removes checkerboard texture while keeping coin edges sharp
    blurred = cv2.bilateralFilter(corrected_image, 11, 100, 100)

    if verbose:
        print("[2.2] Contrast Enhancement (CLAHE)...")
    # Adaptive histogram equalization for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)

    if verbose:
        print("[2.3] Edge-based preprocessing...")
    # Canny edge detection to find coin boundaries
    edges = cv2.Canny(enhanced, 20, 80)

    # Dilate edges to connect broken edges
    kernel_edge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges_dilated = cv2.dilate(edges, kernel_edge, iterations=2)

    # Close gaps in edges to form complete boundaries
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edges_closed = cv2.morphologyEx(edges_dilated, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    if verbose:
        print("[2.4] Filling enclosed regions to create solid coin masks...")
    # Find contours and fill enclosed regions
    contours_temp, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a filled mask
    filled = np.zeros_like(edges_closed)
    for contour in contours_temp:
        area = cv2.contourArea(contour)
        if area > 3000:  # Only fill large regions (potential coins)
            cv2.drawContours(filled, [contour], -1, 255, -1)

    # Clean up the filled mask
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    filled_cleaned = cv2.morphologyEx(filled, cv2.MORPH_OPEN, kernel_clean, iterations=1)

    # Dilate slightly to ensure coins are solid
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    final = cv2.dilate(filled_cleaned, kernel_dilate, iterations=1)

    if verbose:
        print(" Preprocessing completed!")
    
    return enhanced, final