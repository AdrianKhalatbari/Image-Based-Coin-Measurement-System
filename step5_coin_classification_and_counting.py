"""
Step 5: Coin classification and counting
Classify coins based on size and count them in the image.
"""

import numpy as np


DIAMETER_2E = 25.75     # Diameter of 2 Euro coin in mm
DIAMETER_1E = 23.25     # Diameter of 1 Euro coin in mm
DIAMETER_50C = 24.25    # Diameter of 50 Cent coin in mm
DIAMETER_20C = 22.25    # Diameter of 20 Cent coin in mm
DIAMETER_10C = 19.75    # Diameter of 10 Cent coin in mm
DIAMETER_5C = 21.25     # Diameter of 5 Cent coin in mm

def classify_coin(coins, px_to_mm_width_ratio, px_to_mm_height_ratio, verbose=False):
    """
    Classify coins based on their diameters.

    Args:
        coins (list): List of coin dictionaries with 'area' and 'bbox' keys
        px_to_mm_ratio (float): Conversion ratio from pixels to millimeters

    Returns:
        dict: Dictionary with counts of each coin type
    """
    if verbose:
        print("=" * 60)
        print("STEP 5: Coin classification and counting")
        print("=" * 60)
    
    coin_counts = np.zeros(6, dtype=int)
    coin_types = ['5C', '10C', '20C', '50C', '1E', '2E']

    for idx, coin in enumerate(coins):
        x, y, w, h = coin['bbox']
        diameter_height_mm = h * px_to_mm_height_ratio
        
        # Collect all diameters differences and take the minimum
        diameter_diffs = [
            abs(diameter_height_mm - DIAMETER_5C),
            abs(diameter_height_mm - DIAMETER_10C),
            abs(diameter_height_mm - DIAMETER_20C),
            abs(diameter_height_mm - DIAMETER_50C),
            abs(diameter_height_mm - DIAMETER_1E),
            abs(diameter_height_mm - DIAMETER_2E)
        ]
        
        min_diff_index = np.argmin(diameter_diffs)
        coin_counts[min_diff_index] += 1
        
        if verbose:
            print(f"   • Coin {idx + 1}: Diameter = {diameter_height_mm:.2f} mm -> Classified as {coin_types[min_diff_index]}")
        

    return coin_counts