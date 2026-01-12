"""
Step 3: Segmentation (Coin Extraction)
Detects and extracts individual coins using contour analysis and shape filtering.
"""

import cv2
import numpy as np


def segment_coins(binary_image, original_image, image_name=""):
    """
    Detect and segment individual coins from binary mask.

    Uses contour detection and filters based on area, aspect ratio, circularity,
    solidity, and extent to identify coin-shaped objects.

    Args:
        binary_image (np.ndarray): Binary mask from Step 2
        original_image (np.ndarray): Original grayscale image for visualization
        image_name (str): Name of the image being processed (for logging)

    Returns:
        tuple: (coins, segmented_image)
            - coins: List of dictionaries containing coin properties
            - segmented_image: Color image with detected coins highlighted
    """
    print("\n[3.1] Finding contours...")
    contours, hierarchy = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    print(f"   ✓ Found {len(contours)} contours")

    print("[3.2] Filtering coin contours...")
    coins = []
    rejected_coins = []

    # Calculate dynamic area thresholds based on image size
    image_area = binary_image.shape[0] * binary_image.shape[1]
    min_area = image_area * 0.006  # 0.6% of image
    max_area = image_area * 0.12  # 12% of image

    print(f"   📸 Image: {image_name}")
    print(f"   📏 Image area: {image_area}")
    print(f"   📏 Min coin area: {min_area:.0f}")
    print(f"   📏 Max coin area: {max_area:.0f}")

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)

        if min_area < area < max_area:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Calculate shape metrics
            aspect_ratio = float(w) / h if h > 0 else 0

            # Circularity: 4π*Area/Perimeter² (1.0 for perfect circle)
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0

            # Solidity: Area/ConvexHullArea (measures compactness)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0

            # Extent: Area/BoundingBoxArea (how well it fills bounding box)
            bbox_area = w * h
            extent = area / bbox_area if bbox_area > 0 else 0

            # Flexible filtering criteria for imperfect coins
            passes_shape = (
                    (0.5 < aspect_ratio < 1.8) and  # Reasonably round
                    (circularity > 0.3 or solidity > 0.75 or extent > 0.65)
            )

            if passes_shape:
                coins.append({
                    'id': len(coins),
                    'contour': contour,
                    'area': area,
                    'bbox': (x, y, w, h),
                    'circularity': circularity,
                    'aspect_ratio': aspect_ratio,
                    'solidity': solidity,
                    'extent': extent,
                    'center': (x + w // 2, y + h // 2)
                })
                print(f"   ✓ Coin {len(coins)}: area={area:.0f}, circ={circularity:.2f}, "
                      f"aspect={aspect_ratio:.2f}, solid={solidity:.2f}, extent={extent:.2f}")
            else:
                rejected_coins.append({'area': area})
                if area > min_area * 1.2:
                    reason = []
                    if not (0.5 < aspect_ratio < 1.8):
                        reason.append(f"aspect={aspect_ratio:.2f}")
                    if not (circularity > 0.3 or solidity > 0.75 or extent > 0.65):
                        reason.append(f"shape metrics too low")
                    print(f"   ✗ REJECTED: area={area:.0f}, {', '.join(reason)}")

        elif area < min_area and area > min_area * 0.4:
            print(f"   ⚠ Too small: area={area:.0f} (need >{min_area:.0f})")
        elif area >= max_area and area < max_area * 2.0:
            print(f"   ⚠ Too large (merged coins?): area={area:.0f} (need <{max_area:.0f})")

    print(f"   ✓ Detected {len(coins)} coins after filtering")
    if rejected_coins:
        print(f"   ✗ Rejected {len(rejected_coins)} potential coins")

    # Create visualization
    segmented_image = create_visualization(original_image, coins)

    print("✅ Segmentation completed!")
    return coins, segmented_image


def create_visualization(original_image, coins):
    """
    Create visualization with detected coins highlighted.

    Draws contours, bounding boxes, center points, and labels on the image.

    Args:
        original_image (np.ndarray): Original grayscale image
        coins (list): List of detected coin dictionaries

    Returns:
        np.ndarray: Color image (BGR) with annotations
    """
    segmented_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)

    for idx, coin in enumerate(coins):
        x, y, w, h = coin['bbox']
        center = coin['center']

        # Draw contour in green
        cv2.drawContours(segmented_image, [coin['contour']], -1, (0, 255, 0), 3)

        # Draw bounding box in blue
        cv2.rectangle(segmented_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Draw center point in red
        cv2.circle(segmented_image, center, 7, (0, 0, 255), -1)

        # Add label
        label = f"Coin {idx + 1}"
        cv2.putText(segmented_image, label,
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return segmented_image