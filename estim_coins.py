from CoinImageProcessingPipeline import save_comparison_figure
from step1_calibration import calibrate_intensity, apply_flat_field_correction, load_images_from_directory
from step2_preprocessing import preprocess_image
from step3_segmentation import segment_coins
from step4_geometric_calibration import threshold_black_squares, get_square_size, mm_to_pixels_ratio
from step5_coin_classification_and_counting import classify_coin

import cv2
import numpy as np
from pathlib import Path

SQUARE_MM_SIZE = 12.5  # Size of the black square in mm
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

def estim_coins(measurement, bias, dark, flat, image_index=1, verbose=False):
    """
    Estimate and classify coins in the given image.

    Args:
        measurement (np.ndarray): Raw measurement image
        bias (np.ndarray): Bias frame
        dark (np.ndarray): Dark frame
        flat (np.ndarray): Flat field frame
        image_index (int): Index of the image being processed
    Returns:
        dict: Dictionary with counts of each coin type
    """    
    
    # Step 0: Convert to single channel grayscale if needed
    if len(measurement.shape) == 3 and measurement.shape[2] == 3:
        measurement = cv2.cvtColor(measurement.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    if len(bias.shape) == 3 and bias.shape[2] == 3:
        bias = cv2.cvtColor(bias.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        
    if len(dark.shape) == 3 and dark.shape[2] == 3:
        dark = cv2.cvtColor(dark.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        
    if len(flat.shape) == 3 and flat.shape[2] == 3:
        flat = cv2.cvtColor(flat.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    # Step 1: Apply calibration
    corrected = apply_flat_field_correction(measurement, bias, dark, flat)

    # Step 2: Preprocessing
    enhanced, binary = preprocess_image(corrected)

    # Step 3: Segmentation
    coins, segmented = segment_coins(binary, corrected)
    
    if verbose:
        print(f"\n[SAVING] Saving results to {output_dir}/...")
        cv2.imwrite(str(output_dir / f"image_{image_index + 1}_corrected.png"), corrected)
        cv2.imwrite(str(output_dir / f"image_{image_index + 1}_enhanced.png"), enhanced)
        cv2.imwrite(str(output_dir / f"image_{image_index + 1}_binary.png"), binary)
        cv2.imwrite(str(output_dir / f"image_{image_index + 1}_segmented.png"), segmented)

        # Create comparison figure
        save_comparison_figure(
            measurement, corrected, enhanced, binary, segmented,
            output_dir / f"image_{image_index + 1}_pipeline.png"
        )

        print(f"✅ Image {image_index + 1} processing completed!")
        print(f"   • Detected {len(coins)} coins")
        print(f"   • Results saved to: {output_dir}/")

    # Step 4: Geometric Calibration
    binary_squares = threshold_black_squares(measurement)
    square_width_px, square_height_px = get_square_size(binary_squares)
    
    if square_width_px == 0 or square_height_px == 0:
        raise ValueError("No black square detected for geometric calibration.")
    
    px_to_mm_width_ratio = mm_to_pixels_ratio(SQUARE_MM_SIZE, square_width_px)
    px_to_mm_height_ratio = mm_to_pixels_ratio(SQUARE_MM_SIZE, square_height_px)

    # Step 5: Coin Classification and Counting
    coin_counts = classify_coin(coins, px_to_mm_width_ratio, px_to_mm_height_ratio, verbose=verbose)

    return coin_counts