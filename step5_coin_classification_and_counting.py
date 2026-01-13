'''
Step 5: Coin classification and counting
Classify coins based on size and count them in the image.
'''

import cv2
import numpy as np
from pathlib import Path

DIAMETER_2E = 25.75     # Diameter of 2 Euro coin in mm
DIAMETER_1E = 23.25     # Diameter of 1 Euro coin in mm
DIAMETER_50C = 24.25    # Diameter of 50 Cent coin in mm
DIAMETER_20C = 22.25    # Diameter of 20 Cent coin in mm
DIAMETER_10C = 19.75    # Diameter of 10 Cent coin in mm
DIAMETER_5C = 21.25     # Diameter of 5 Cent coin in mm

def classify_coin(coins, px_to_mm_width_ratio, px_to_mm_height_ratio):
    """
    Classify coins based on their diameters.

    Args:
        coins (list): List of coin dictionaries with 'area' and 'bbox' keys
        px_to_mm_ratio (float): Conversion ratio from pixels to millimeters

    Returns:
        dict: Dictionary with counts of each coin type
    """
    coin_counts_width = {
        '5c': 0,
        '10c': 0,
        '20c': 0,
        '50c': 0,
        '1e': 0,
        '2e': 0
    }
    coin_counts_height = coin_counts_width.copy()

    for idx, coin in enumerate(coins):
        x, y, w, h = coin['bbox']
        diameter_width_mm = w * px_to_mm_width_ratio
        diameter_height_mm = h * px_to_mm_height_ratio
        
        # Collect all diameters differences and take the minimum
        width_diameter_diffs = {
            '2e': abs(diameter_width_mm - DIAMETER_2E),
            '1e': abs(diameter_width_mm - DIAMETER_1E),
            '50c': abs(diameter_width_mm - DIAMETER_50C),
            '20c': abs(diameter_width_mm - DIAMETER_20C),
            '10c': abs(diameter_width_mm - DIAMETER_10C),
            '5c': abs(diameter_width_mm - DIAMETER_5C)
        }
        
        classified_coin = min(width_diameter_diffs, key=width_diameter_diffs.get)
        coin_counts_width[classified_coin] += 1
        
        
        # Height based classification
        height_diameter_diffs = {
            '2e': abs(diameter_height_mm - DIAMETER_2E),
            '1e': abs(diameter_height_mm - DIAMETER_1E),
            '50c': abs(diameter_height_mm - DIAMETER_50C),
            '20c': abs(diameter_height_mm - DIAMETER_20C),
            '10c': abs(diameter_height_mm - DIAMETER_10C),
            '5c': abs(diameter_height_mm - DIAMETER_5C)
        }
        
        classified_coin_height = min(height_diameter_diffs, key=height_diameter_diffs.get)
        coin_counts_height[classified_coin_height] += 1
        
        
        print(f"   • Coin {idx + 1}: Diameter = {diameter_width_mm:.2f} mm -> Classified as {classified_coin}")
        print(f"   • Coin {idx + 1}: Diameter = {diameter_height_mm:.2f} mm -> Classified as {classified_coin_height}")
        

    return coin_counts_width, coin_counts_height