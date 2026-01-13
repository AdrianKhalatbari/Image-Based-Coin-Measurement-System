from step1_calibration import load_images_from_directory, calibrate_intensity
from estim_coins import estim_coins

from pathlib import Path


def main():
    measurements_dir1 = Path("data/DIIP-images-measurements-1/DIIP-images/Measurements")
    measurements_dir2 = Path("data/DIIP-images-measurements-2/DIIP-images/Measurements")

    # Load calibration frames
    bias_frame, dark_frame, flat_frame = calibrate_intensity("data")
    

    # Load measurement images
    measurement_images = load_images_from_directory(measurements_dir1)
    measurement_images += load_images_from_directory(measurements_dir2)
    
    
    # Print how many images were loaded
    print(f"Loaded {len(measurement_images)} measurement images.")
    
    coin_counts = {}
    for i, measurement in enumerate(measurement_images):
        print(f"\nProcessing image {i+1}: {measurement.shape}")
        coin_counts[i] = estim_coins(measurement, bias_frame, dark_frame, flat_frame, i)
        print(f"Coin counts for image {i+1}: {coin_counts[i]}")
        
    print("\nAll images processed. Summary of coin counts:")
    for img_idx, counts in coin_counts.items():
        print(f"Image {img_idx + 1}: {counts}")
        
if __name__ == "__main__":
    main()