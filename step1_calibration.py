"""
Step 1: Intensity Calibration/Correction
Implements flat-field correction using bias, dark, and flat frames.
"""

import cv2
import numpy as np
from pathlib import Path


def load_images_from_directory(directory):
    """
    Load all images from a specified directory.

    Args:
        directory (Path or str): Path to a directory containing images

    Returns:
        list: List of images as numpy arrays (float32)
    """
    directory = Path(directory)
    images = []

    if not directory.exists():
        print(f"    Directory does not exist: {directory}")
        return images

    for img_path in sorted(directory.glob("*")):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img.astype(np.float32))

    return images


def compute_master_frame(images):
    """
    Compute the master frame by averaging multiple images.

    Args:
        images (list): List of images as numpy arrays

    Returns:
        np.ndarray: Average of all images (master frame)
    """
    if not images:
        return None
    return np.mean(images, axis=0).astype(np.float32)


def calibrate_intensity(data_dir="data", verbose=False):
    """
    Perform intensity calibration by computing master bias, dark, and flat frames.

    This implements the flat-field correction formula:
        Corrected = (Raw - Bias - Dark) / Flat

    Args:
        data_dir (str): Root directory containing calibration image folders

    Returns:
        tuple: (bias_frame, dark_frame, flat_frame) as numpy arrays
    """
    
    if verbose:
        print("=" * 60)
        print("STEP 1: INTENSITY CALIBRATION/CORRECTION")
        print("=" * 60)

    data_path = Path(data_dir)

    # Load and compute master bias frame
    if verbose:
        print("\n[1/3] Computing Master Bias Frame...")
        
    bias_dir = data_path / "DIIP-images-bias" / "DIIP-images" / "Bias"
    bias_images = load_images_from_directory(bias_dir)

    if bias_images:
        bias_frame = compute_master_frame(bias_images)
        
        if verbose:
            print(f"   ✓ Loaded {len(bias_images)} bias images")
            print(f"   ✓ Master bias shape: {bias_frame.shape}")
    else:
        if verbose:
            print("    No bias images found. Using zero bias.")
        bias_frame = np.zeros((100, 100), dtype=np.float32)

    # Load and compute the master dark frame
    if verbose:
        print("\n[2/3] Computing Master Dark Frame...")
    dark_dir = data_path / "DIIP-images-dark" / "DIIP-images" / "Dark"
    dark_images = load_images_from_directory(dark_dir)

    if dark_images:
        dark_frame = compute_master_frame(dark_images)
        
        if verbose:
            print(f"   ✓ Loaded {len(dark_images)} dark images")
            print(f"   ✓ Master dark shape: {dark_frame.shape}")
            
    else:
        if verbose:
            print("   No dark images found. Using zero dark.")
        dark_frame = np.zeros_like(bias_frame)

    # Load and compute the master flat frame
    if verbose:
        print("\n[3/3] Computing Master Flat Frame...")
    
    flat_dir = data_path / "DIIP-images-flat" / "DIIP-images" / "Flat"
    flat_images = load_images_from_directory(flat_dir)

    if flat_images:
        raw_flat = compute_master_frame(flat_images)
        # Subtract bias and dark from flat
        flat_frame = raw_flat - bias_frame - dark_frame
        # Avoid division by zero
        flat_frame = np.where(flat_frame < 1.0, 1.0, flat_frame)
        # Normalize flat frame
        flat_frame = flat_frame / np.mean(flat_frame)
        
        if verbose:
            print(f"   ✓ Loaded {len(flat_images)} flat images")
            print(f"   ✓ Master flat shape: {flat_frame.shape}")
            print(f"   ✓ Flat frame normalized")
    else:
        if verbose:
            print("   No flat images found. Using unity flat.")
        flat_frame = np.ones_like(bias_frame)

    if verbose:
        print("\n Intensity calibration completed!")
        
    return bias_frame, dark_frame, flat_frame


def apply_flat_field_correction(raw_image, bias_frame, dark_frame, flat_frame):
    """
    Apply flat-field correction to a raw image.

    Formula: Corrected = (Raw - Bias - Dark) / Flat

    Args:
        raw_image (np.ndarray): Raw input image
        bias_frame (np.ndarray): Master bias frame
        dark_frame (np.ndarray): Master dark frame
        flat_frame (np.ndarray): Master flat frame

    Returns:
        np.ndarray: Corrected image (uint8)
    """
    # Ensure all frames have the same shape
    if raw_image.shape != bias_frame.shape:
        bias = cv2.resize(bias_frame, (raw_image.shape[1], raw_image.shape[0]))
        dark = cv2.resize(dark_frame, (raw_image.shape[1], raw_image.shape[0]))
        flat = cv2.resize(flat_frame, (raw_image.shape[1], raw_image.shape[0]))
    else:
        bias = bias_frame
        dark = dark_frame
        flat = flat_frame

    # Apply correction: (Raw - Bias - Dark) / Flat
    corrected = (raw_image - bias - dark) / flat

    # Clip values to valid range
    corrected = np.clip(corrected, 0, 255)
    return corrected.astype(np.uint8)