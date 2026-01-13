from step1_calibration import calibrate_intensity, apply_flat_field_correction, load_images_from_directory
from step2_preprocessing import preprocess_image
from step3_segmentation import segment_coins
from step4_geometric_calibration import threshold_black_squares, get_square_size, mm_to_pixels_ratio
from step5_coin_classification_and_counting import classify_coin

import cv2
import numpy as np
from pathlib import Path

SQUARE_MM_SIZE = 12.5  # Size of the black square in mm

def estim_coins(measurement, bias, dark, flat):
    """
    Estimate and classify coins in the given image.

    Args:
        measurement (np.ndarray): Raw measurement image
        bias (np.ndarray): Bias frame
        dark (np.ndarray): Dark frame
        flat (np.ndarray): Flat field frame
        
    Returns:
        dict: Dictionary with counts of each coin type
    """
    # Step 1: Apply calibration
    corrected = apply_flat_field_correction(measurement, bias, dark, flat)

    # Step 2: Preprocessing
    enhanced, binary = preprocess_image(corrected)

    # Step 3: Segmentation
    coins, segmented = segment_coins(binary, corrected)
    
    print(f"   ✓ Detected {len(coins)} coins")

    # Step 4: Geometric Calibration
    binary_squares = threshold_black_squares(corrected)
    square_width_px, square_height_px = get_square_size(binary_squares)
    
    if square_width_px == 0 or square_height_px == 0:
        raise ValueError("No black square detected for geometric calibration.")
    
    px_to_mm_width_ratio = mm_to_pixels_ratio(SQUARE_MM_SIZE, square_width_px)
    px_to_mm_height_ratio = mm_to_pixels_ratio(SQUARE_MM_SIZE, square_height_px)

    # Step 5: Coin Classification and Counting
    coin_counts = classify_coin(coins, px_to_mm_width_ratio, px_to_mm_height_ratio)

    return coin_counts