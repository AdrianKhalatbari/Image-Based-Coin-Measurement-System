from step1_calibration import load_images_from_directory, calibrate_intensity
from estim_coins import estim_coins

from pathlib import Path


def main():
    measurements_dir = Path("data/DIIP-images-measurements-2/DIIP-images/Measurements")

    # Load calibration frames
    bias_frame, dark_frame, flat_frame = calibrate_intensity("data")
    

    # Load measurement images
    measurement_images = load_images_from_directory(measurements_dir)
    
    # Print how many images were loaded
    print(f"Loaded {len(measurement_images)} measurement images.")
    
    for i, measurement in enumerate(measurement_images):
        print(f"\nProcessing image {i+1}: {measurement.shape}")
        coin_counts_width, coin_counts_height = estim_coins(measurement, bias_frame, dark_frame, flat_frame, i)
        print(f"Coin counts for image {i+1} (width): {coin_counts_width}")
        print(f"Coin counts for image {i+1} (height): {coin_counts_height}")
        
        
if __name__ == "__main__":
    main()