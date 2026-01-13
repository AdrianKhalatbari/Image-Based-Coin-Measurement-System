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
    coin_counts = {
        '5c': 0,
        '10c': 0,
        '20c': 0,
        '50c': 0,
        '1e': 0,
        '2e': 0
    }

    for coin in coins:
        x, y, w, h = coin['bbox']
        diameter_mm = (w * px_to_mm_width_ratio + h * px_to_mm_height_ratio) / 2
        
        # Collect all diameters differences and take the minimum
        diameter_diffs = {
            '2e': abs(diameter_mm - DIAMETER_2E),
            '1e': abs(diameter_mm - DIAMETER_1E),
            '50c': abs(diameter_mm - DIAMETER_50C),
            '20c': abs(diameter_mm - DIAMETER_20C),
            '10c': abs(diameter_mm - DIAMETER_10C),
            '5c': abs(diameter_mm - DIAMETER_5C)
        }
        
        classified_coin = min(diameter_diffs, key=diameter_diffs.get)
        coin_counts[classified_coin] += 1
        

    return coin_counts