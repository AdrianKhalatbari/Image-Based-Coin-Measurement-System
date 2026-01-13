"""
Step 4: Geometric calibration
Find size of black squares in pixels and calculate mm to pixels ratio.
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt



# Thresholding function to isolate black squares
def threshold_black_squares(image, threshold=40):
    """
    Apply thresholding to isolate black squares in the image.

    Args:
        image (np.ndarray): Input grayscale image
        threshold (int): Threshold value to isolate black squares

    Returns:
        np.ndarray: Binary mask with black squares isolated
    """
    
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    
    
    _, binary_mask = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
    
    # Show detected squares image
    print("\n[4.1] Thresholding to isolate black squares...")
    plt.figure(figsize=(6, 6))
    plt.imshow(binary_mask, cmap='gray')
    plt.title('Detected Black Squares')
    plt.axis('off')
    plt.show()
    
    return binary_mask


def get_square_size(binary_mask):
    """
    Calculate the length of the black square in pixels.

    Args:
        binary_mask (np.ndarray): Binary mask with black squares isolated
        
    Returns:
        float: Length of the black square in pixels
    """
    
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        print(f"Detected black square size: width={w}px, height={h}px")
        return w, h
    else:
        return 0, 0
    
    
def mm_to_pixels_ratio(real_size_mm, pixel_size):
    """
    Calculate the mm to pixels ratio.

    Args:
        real_size_mm (float): Real-world size in millimeters
        pixel_size (float): Size in pixels

    Returns:
        float: mm to pixels ratio
    """
    if pixel_size == 0:
        return 0
    return real_size_mm / pixel_size