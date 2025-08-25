import os
import cv2
import numpy as np
from matplotlib import pyplot as plt


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

    os.makedirs(output_dir, exist_ok=True)

    image_paths = []
    for root, _, files in os.walk(preprocessed_dir):
        for file in files:
            image_paths.append(os.path.join(root, file))

    total_images = len(image_paths)

    print(f"Processing {total_images} preprocessed images for colour and SIFT feature extraction...")

    for idx, image_path in enumerate(image_paths, 1):
        print(f"\rProcessing image {idx}/{total_images}", end="", flush=True)

        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not load image at {image_path}")
            continue

        # Always extract these
        rel_path = os.path.relpath(image_path, start="./processed_images")
        anime_name = os.path.dirname(rel_path).split(os.sep)[0]
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        filename = f"{anime_name}__{image_name}.npy"

        hist = extract_colour_histogram(image)
        sift_desc = extract_sift_descriptors(image)

        if hist is not None:
            np.save(os.path.join(output_dir, filename), hist)

        if sift_desc is not None:
            sift_dir = "./sift_descriptors"
            os.makedirs(sift_dir, exist_ok=True)
            np.save(os.path.join(sift_dir, filename), sift_desc)

    print("\nColour and SIFT feature extraction complete.")



def extract_canny_edges(img):
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 100)

    cv2.imwrite("processed_edges.jpg", edges)

    return edges


def display_image_and_sift_keypoints(image_path):
    """
    Loads an input image, computes its SIFT keypoints,
    and displays the original image alongside the image with keypoints drawn.

    Parameters:
        image_path: Path to the input image.
    """
    # Load the input image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return

    # Convert to grayscale for SIFT
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create SIFT detector and compute keypoints
    sift = cv2.SIFT_create()
    keypoints, _ = sift.detectAndCompute(gray, None)

    # Draw keypoints on the image
    keypoints_image = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Convert images to RGB for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    keypoints_rgb = cv2.cvtColor(keypoints_image, cv2.COLOR_BGR2RGB)

    # Display side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.imshow(image_rgb)
    ax1.set_title("Original Image")
    ax1.axis("off")

    ax2.imshow(keypoints_rgb)
    ax2.set_title("SIFT Keypoints")
    ax2.axis("off")

    plt.tight_layout()
    plt.show()








def extract_sift_descriptors(img):
    """
    Extracts SIFT descriptors from an image.

    :param img: Preprocessed image
    :return: descriptors (numpy array) or None if not found
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return descriptors
