
"""
Coin Image Processing Pipeline - Main Script
Integrates all three steps: Calibration, Preprocessing, and Segmentation
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Import the three processing steps
from step1_calibration import calibrate_intensity, apply_flat_field_correction, load_images_from_directory
from step2_preprocessing import preprocess_image
from step3_segmentation import segment_coins


def save_comparison_figure(raw, corrected, enhanced, binary, segmented, output_path):
    """
    Create and save a comparison figure showing all processing steps.

    Args:
        raw (np.ndarray): Raw input image
        corrected (np.ndarray): Intensity-corrected image
        enhanced (np.ndarray): CLAHE-enhanced image
        binary (np.ndarray): Binary mask
        segmented (np.ndarray): Final segmented result
        output_path (Path): Path to save the figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Coin Processing Pipeline', fontsize=16, fontweight='bold')

    axes[0, 0].imshow(raw, cmap='gray')
    axes[0, 0].set_title('1. Raw Image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(corrected, cmap='gray')
    axes[0, 1].set_title('2. Intensity Corrected')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(enhanced, cmap='gray')
    axes[0, 2].set_title('3. Enhanced (CLAHE)')
    axes[0, 2].axis('off')

    axes[1, 0].imshow(binary, cmap='gray')
    axes[1, 0].set_title('4. Binary (Edge-based)')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('5. Segmented Coins')
    axes[1, 1].axis('off')

    # Hide the last subplot
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def process_measurement_images(measurements_dir, output_dir, bias_frame, dark_frame, flat_frame):
    """
    Process all measurement images in a directory.

    Applies all three steps to each image and saves results.

    Args:
        measurements_dir (Path): Directory containing measurement images
        output_dir (Path): Directory to save results
        bias_frame (np.ndarray): Master bias frame from calibration
        dark_frame (np.ndarray): Master dark frame from calibration
        flat_frame (np.ndarray): Master flat frame from calibration
    """
    measurements_dir = Path(measurements_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"\n{'=' * 60}")
    print(f"Processing images from: {measurements_dir.name}")
    print(f"{'=' * 60}")

    # Load measurement images
    image_paths = sorted(measurements_dir.glob("*"))
    image_paths = [p for p in image_paths if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']]

    if not image_paths:
        print(f"⚠ No images found in {measurements_dir}")
        return

    print(f"Found {len(image_paths)} measurement images\n")

    for idx, img_path in enumerate(image_paths):
        print(f"\n{'─' * 60}")
        print(f"Processing Image {idx + 1}/{len(image_paths)}: {img_path.name}")
        print(f"{'─' * 60}")

        # Load raw image
        raw_image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if raw_image is None:
            print(f"   ⚠ Failed to load image: {img_path.name}")
            continue

        raw_image = raw_image.astype(np.float32)

        # Step 1: Apply calibration
        print("\n[STEP 1] Applying intensity correction...")
        corrected = apply_flat_field_correction(raw_image, bias_frame, dark_frame, flat_frame)

        # Step 2: Preprocessing
        print("\n[STEP 2] Preprocessing...")
        enhanced, binary = preprocess_image(corrected)

        # Step 3: Segmentation
        print("\n[STEP 3] Segmentation...")
        coins, segmented = segment_coins(binary, corrected, img_path.name)

        # Save results
        print(f"\n[SAVING] Saving results to {output_dir}/...")
        cv2.imwrite(str(output_dir / f"image_{idx + 1}_corrected.png"), corrected)
        cv2.imwrite(str(output_dir / f"image_{idx + 1}_enhanced.png"), enhanced)
        cv2.imwrite(str(output_dir / f"image_{idx + 1}_binary.png"), binary)
        cv2.imwrite(str(output_dir / f"image_{idx + 1}_segmented.png"), segmented)

        # Create comparison figure
        save_comparison_figure(
            raw_image, corrected, enhanced, binary, segmented,
            output_dir / f"image_{idx + 1}_pipeline.png"
        )

        print(f"✅ Image {idx + 1} processing completed!")
        print(f"   • Detected {len(coins)} coins")
        print(f"   • Results saved to: {output_dir}/")


def main():
    """
    Main function to run the complete coin processing pipeline.

    Executes all three steps:
    1. Intensity calibration (flat-field correction)
    2. Preprocessing (coin separation)
    3. Segmentation (coin extraction)
    """
    print("\n" + "=" * 60)
    print("COIN IMAGE PROCESSING PIPELINE")
    print("=" * 60)

    data_dir = "data"

    # Step 1: Perform intensity calibration
    bias_frame, dark_frame, flat_frame = calibrate_intensity(data_dir)

    # Process measurements from folder 1
    measurements_dir_1 = Path(data_dir) / "DIIP-images-measurements-1" / "DIIP-images" / "Measurements"
    if measurements_dir_1.exists():
        process_measurement_images(
            measurements_dir_1,
            "output/measurements_1",
            bias_frame, dark_frame, flat_frame
        )

    # Process measurements from folder 2
    measurements_dir_2 = Path(data_dir) / "DIIP-images-measurements-2" / "DIIP-images" / "Measurements"
    if measurements_dir_2.exists():
        process_measurement_images(
            measurements_dir_2,
            "output/measurements_2",
            bias_frame, dark_frame, flat_frame
        )

    print("\n" + "=" * 60)
    print("ALL PROCESSING COMPLETED! ✅")
    print("=" * 60)
    print("\nCheck the 'output' folder for results.")


if __name__ == "__main__":
    main()