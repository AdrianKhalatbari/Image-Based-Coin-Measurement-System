Image-Based Coin Measurement System
Course: BM40A1201 Digital Imaging and Image Preprocessing

Students:
- Mohammad Khalatbari Mokaram, 003235474
- Antonio Oliva, 003247116

Description:
This project implements an image-based measurement system to estimate
the number of coins in an image using calibrated imaging data.

Files:
* CoinImageProcessingPipeline.py
  This file is the primary pipeline script that manages the entire coin processing procedure.
  Processes every measurement image from both dataset files and contains all processing stages.
  Saves all intermediate and final results to the output directory and creates comparison figures that show each processing step.

* step1_calibration.py
  In this file we implement intensity calibration using flat-field correction formula
  Corrected = (Raw - Bias - Dark) / Flat. It loads bias, dark, and flat-field
  images from data directories (source data downloaded from Moodle), computes master frames
  by averaging, and applies correction to remove sensor artifacts and uneven illumination.

* step2_preprocessing.py
  In this file we implement edge-based preprocessing to separate coins from textured backgrounds.
  Uses bilateral filtering to remove checkerboard patterns while preserving edges,
  applies CLAHE for contrast enhancement, performs Canny edge detection, and uses
  morphological operations (dilation, closing, filling) to create solid coin masks.

* step3_segmentation.py
  In this file we try to detect and extract individual coins using contour analysis and multi-criteria
  shape filtering. The function calculates geometric features (circularity, aspect ratio,
  solidity, extent) for each contour and applies flexible filtering criteria to
  identify coin-shaped objects. Creates visualizations with contours, bounding
  boxes, centers, and labels.

* step4_geometric_calibration.py
  In this file we isolate the black squares from the raw image (easiest for thresholding
  only the squares), find the size in pixels of one square, chosen using maximum area, and
  calculate the ratio using the reference measure given in the project specification
  (square size = 12.5mm x 12.5mm).

* step5_coin_classification_and_counting.py
  This is the main file for coin classification and counting. The measurements of the
  contours (detected coins) found during step 3 are converted from pixels to millimeters
  using the ratio calculated in step 4. Then, each coin is assigned to one coin type based
  on the coin diameter, taken as the height of the contour that contains the coin. The value
  is compared to reference values given in the project specification.

* requirements.txt
  This is a list of all Python library dependencies with minimum versions required to run
  the project.

* data/
  This directory contains input calibration and measurement images.
  It is not in the final repository due to project description.
  
  IMPORTANT - Data Directory Structure:
  When extracting the source files downloaded from Moodle, place them in the data directory
  following this structure:
  data/DIIP-images-{category}/DIIP-images/{category-name}
  
  For example:
  - data/DIIP-images-bias/DIIP-images/Bias
  - data/DIIP-images-dark/DIIP-images/Dark
  - data/DIIP-images-flat/DIIP-images/Flat
  - data/DIIP-images-measurements-1/DIIP-images/Measurements
  - data/DIIP-images-measurements-2/DIIP-images/Measurements
  
  This is the same directory structure that results when you extract the source files
  directly from Moodle into the data folder.

* output/
  Stores all generated results including corrected images, enhanced images,
  binary masks, segmented images with annotations, and pipeline comparison figures
  for both measurement sets.

How to run:
Install dependencies:
`pip install -r requirements.txt`
Run:
`python main.py`.
The script will automatically process
all images and give the coins classification as output.

Notes:
The use of Python instead of MATLAB was discussed and approved
by the course TA.
