from step1_calibration import load_images_from_directory
from estim_coins import estim_coins

from pathlib import Path


def main():
    measurements_dir = Path("data/DIIP-images-measurements-1/DIIP-images/Measurements")
    bias_dir = Path("data/DIIP-images-bias/DIIP-images/Bias")
    dark_dir = Path("data/DIIP-images-dark/DIIP-images/Dark")
    flat_dir = Path("data/DIIP-images-flat/DIIP-images/Flat")

    # Load calibration frames
    bias_images = load_images_from_directory(bias_dir)
    dark_images = load_images_from_directory(dark_dir)
    flat_images = load_images_from_directory(flat_dir)

    # Load measurement images
    measurement_images = load_images_from_directory(measurements_dir)
    
    # Print how many images were loaded
    print(f"Loaded {len(bias_images)} bias images.")
    print(f"Loaded {len(dark_images)} dark images.")
    print(f"Loaded {len(flat_images)} flat images.")
    print(f"Loaded {len(measurement_images)} measurement images.")
    
    for i, measurement in enumerate(measurement_images):
        print(f"\nProcessing image {i}: {measurement.shape}")
        coin_counts = estim_coins(measurement, bias_images[0], dark_images[0], flat_images[0])
        print(f"Coin counts for image {i}: {coin_counts}")
        
        
if __name__ == "__main__":
    main()