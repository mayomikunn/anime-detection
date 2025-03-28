import os
import cv2

from preprocessing import preprocess_image


def get_all_image_paths(base_dir):

    # Recursively collects all image file paths from the given base directory.

    image_paths = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            image_path = os.path.join(root, file)
            image_paths.append(image_path)
    return image_paths


def process_single_image(image_path):

    # Processes a single image by applying preprocessing.
    # Returns a tuple of (image_path, processed_image) if successful, otherwise None.
    processed_image = preprocess_image(image_path)
    return (image_path, processed_image) if processed_image is not None else None


def load_database_images():
    base_dir = "C:/Users/mayom/Downloads/archive/data/anime_images"
    processed_parent = "./processed_images"

    # Create the parent folder for processed images if it doesn't exist
    os.makedirs(processed_parent, exist_ok=True)

    # Get a list of all immediate subdirectories in base_dir (for progress reporting)
    all_dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    total_dirs = len(all_dirs)

    # Walk through all directories under base_dir
    for subdir_index, (root, dirs, files) in enumerate(os.walk(base_dir), 1):
        # Print current subdirectory progress on the same line
        print(f"\rProcessing subdirectory {subdir_index}/{total_dirs}: {root} ", end="", flush=True)

        # Compute the relative path from the base directory
        rel_path = os.path.relpath(root, base_dir)
        # Create a corresponding destination folder under the processed_parent directory
        dest_dir = os.path.join(processed_parent, rel_path)
        os.makedirs(dest_dir, exist_ok=True)

        # Process each file in the current directory
        for file in files:
            image_path = os.path.join(root, file)
            processed_image = preprocess_image(image_path)
            if processed_image is not None:
                # Save the processed image in the corresponding destination folder
                dest_path = os.path.join(dest_dir, file)
                cv2.imwrite(dest_path, processed_image)

    print()  # Move to the next line after finishing



