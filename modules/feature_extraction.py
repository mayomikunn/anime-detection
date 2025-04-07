import os
import cv2
import numpy as np


def extract_colour_histogram(img, bins=(8, 8, 8)):

    # Extracts a colour histogram from a preprocessed image.

    # img: Preprocessed image.
    # bins: Number of bins per channel (default: 8 per channel).
    # return: Flattened and normalized colour histogram as a NumPy array.


    # Convert BGR to HSV for better colour representation
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Compute histogram for each channel (H, S, V)
    hist = cv2.calcHist([img_hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])

    # Normalize and flatten the histogram
    hist = cv2.normalize(hist, hist).flatten()

    return hist


def extract_preprocessed_images(preprocessed_dir, output_dir="./colour_histograms"):

    # Processes all preprocessed images to extract and save colour histograms.

    # preprocessed_dir: Path to the preprocessed images directory.
    # output_dir: Directory where histograms will be stored.

    os.makedirs(output_dir, exist_ok=True)

    image_paths = []
    for root, _, files in os.walk(preprocessed_dir):
        for file in files:
            image_paths.append(os.path.join(root, file))

    total_images = len(image_paths)

    print(f"Processing {total_images} preprocessed images for colour feature extraction...")

    for idx, image_path in enumerate(image_paths, 1):
        print(f"\rProcessing image {idx}/{total_images}", end="", flush=True)

        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not load image at {image_path}")
            continue

        hist = extract_colour_histogram(image)
        if hist is not None:
            # Save histogram as a .npy file (NumPy array)
            rel_path = os.path.relpath(image_path, start="./processed_images")
            anime_name = os.path.dirname(rel_path).split(os.sep)[0]  # subdir name = anime name
            image_name = os.path.splitext(os.path.basename(image_path))[0]

            # Use a consistent filename format: "animeName_imageName.npy"
            filename = f"{anime_name}__{image_name}.npy"
            np.save(os.path.join(output_dir, filename), hist)

    print("\nColour feature extraction complete.")





